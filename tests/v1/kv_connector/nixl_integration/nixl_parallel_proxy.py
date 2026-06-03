# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallel disagg proxy for the NIXL connector.

Unlike ``toy_proxy_server.py`` (which ``await``s the full prefill before
contacting the decoder), this proxy dispatches the prefiller and decoder
**in parallel** -- the same strategy as
``examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py``.

Why this is possible
--------------------
The decoder needs the prefiller's *static* identity (NIXL ``engine_id`` and
side-channel ``host``/``port``) to perform its handshake, plus a shared
``transfer_id`` to correlate the request. It does **not** need the prefiller's
per-request block ids: with the parallel control plane the prefiller resolves
its own source blocks by ``transfer_id`` (and buffers a write request that
arrives before its prefill finishes).

The prefiller's static identity is generated once at startup, so we learn it
**lazily** from the first request's response (which still carries
``kv_transfer_params``), cache it, and dispatch every subsequent request to P
and D concurrently.

Usage (mirrors toy_proxy_server.py flags):
  python3 nixl_parallel_proxy.py --port 8200 \
      --prefiller-hosts $PHOST --prefiller-ports 30000 \
      --decoder-hosts $DHOST --decoder-ports 30001
"""

import argparse
import asyncio
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = []
    app.state.decode_clients = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
                # Static NIXL identity of this prefiller, learned lazily from
                # the first response. Guarded by ``identity_ready``.
                "identity": None,
                "identity_ready": asyncio.Event(),
                "identity_lock": asyncio.Lock(),
            }
        )

    for i, (host, port) in enumerate(global_args.decoder_instances):
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()
    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--prefiller-hosts", "--prefiller-host", type=str, nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def get_next_client(app, service_type: str):
    if service_type == "prefill":
        idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[idx]
    elif service_type == "decode":
        idx = next(app.state.decode_iterator)
        return app.state.decode_clients[idx]
    raise ValueError(f"Unknown service type: {service_type}")


def _prefill_params(transfer_id: str) -> dict:
    return {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        # Shared id so P can register its own blocks for this request and
        # match the decoder's write request.
        "transfer_id": transfer_id,
    }


def _decode_params(transfer_id: str, identity: dict) -> dict:
    # Parallel path: give D the prefiller's STATIC identity + transfer_id, but
    # NOT per-request blocks (P resolves those locally by transfer_id).
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_engine_id": identity["remote_engine_id"],
        "remote_host": identity["remote_host"],
        "remote_port": identity["remote_port"],
        "tp_size": identity.get("tp_size", 1),
        "transfer_id": transfer_id,
    }


async def send_request_to_prefill(
    client_info: dict, endpoint: str, req_data: dict, request_id: str, transfer_id: str
):
    """POST a prefill-only request to P. Returns the parsed JSON response."""
    req_data = dict(req_data)
    req_data["kv_transfer_params"] = _prefill_params(transfer_id)
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    req_data.pop("min_tokens", None)
    req_data.pop("min_completion_tokens", None)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    response = await client_info["client"].post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    await response.aread()
    return response.json()


async def _fire_prefill(
    client_info: dict, endpoint: str, req_data: dict, request_id: str, transfer_id: str
):
    """Fire-and-forget prefill: we don't need P's response on the parallel
    path, but we must not let an exception go unretrieved."""
    try:
        await send_request_to_prefill(
            client_info, endpoint, req_data, request_id, transfer_id
        )
    except Exception as e:
        logger.warning("Background prefill for %s failed: %s", request_id, e)


def _capture_identity(prefill_client_info: dict, response_json: dict) -> bool:
    """Extract the prefiller's static NIXL identity from a response. Returns
    True if identity is now known."""
    kvt = (response_json or {}).get("kv_transfer_params") or {}
    engine_id = kvt.get("remote_engine_id")
    host = kvt.get("remote_host")
    port = kvt.get("remote_port")
    if engine_id is None or host is None or port is None:
        return False
    prefill_client_info["identity"] = {
        "remote_engine_id": engine_id,
        "remote_host": host,
        "remote_port": port,
        "tp_size": kvt.get("tp_size", 1),
    }
    prefill_client_info["identity_ready"].set()
    return True


async def stream_decode_response(
    decode_client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    async with decode_client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        transfer_id = f"xfer-{request_id}"

        prefill_client_info = get_next_client(request.app, "prefill")
        decode_client_info = get_next_client(request.app, "decode")

        if prefill_client_info["identity_ready"].is_set():
            # Parallel path: fire prefill and don't wait for it. Pass a copy so
            # the decode params we set below can't race the background task.
            asyncio.create_task(
                _fire_prefill(
                    prefill_client_info, api, dict(req_data), request_id, transfer_id
                )
            )
            req_data["kv_transfer_params"] = _decode_params(
                transfer_id, prefill_client_info["identity"]
            )
        else:
            # Lazy identity capture: the first request per prefiller runs
            # sequentially to learn its static NIXL identity. A lock ensures
            # only one request pays this cost; the rest wait then go parallel.
            async with prefill_client_info["identity_lock"]:
                if prefill_client_info["identity_ready"].is_set():
                    asyncio.create_task(
                        _fire_prefill(
                            prefill_client_info,
                            api,
                            dict(req_data),
                            request_id,
                            transfer_id,
                        )
                    )
                    req_data["kv_transfer_params"] = _decode_params(
                        transfer_id, prefill_client_info["identity"]
                    )
                else:
                    resp = await send_request_to_prefill(
                        prefill_client_info, api, req_data, request_id, transfer_id
                    )
                    if _capture_identity(prefill_client_info, resp):
                        req_data["kv_transfer_params"] = _decode_params(
                            transfer_id, prefill_client_info["identity"]
                        )
                    else:
                        # Couldn't learn identity (e.g. full prefix hit on P).
                        # Fall back to sequential semantics for this request by
                        # forwarding P's own params; retry capture next time.
                        kvt = (resp or {}).get("kv_transfer_params") or {}
                        if kvt:
                            req_data["kv_transfer_params"] = kvt

        async def generate_stream():
            async for chunk in stream_decode_response(
                decode_client_info, api, req_data, request_id=request_id
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
