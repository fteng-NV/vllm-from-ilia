# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Push-mode disaggregated prefilling proxy demo.

Companion to ``disagg_proxy_demo.py`` (pull mode). The client-facing API is
the same; the difference is in how P and D coordinate the KV transfer:

* Pull mode: proxy forwards P's ``kv_transfer_params`` (including
  ``remote_block_ids``) to D, and D pulls KV from P via NIXL READ.
* Push mode: proxy hands D **only** P's coordinates
  (``remote_engine_id``, ``remote_host``, ``remote_port``, ``tp_size``,
  ``pp_size``) and the shared ``remote_request_id``. D registers its locally
  allocated blocks with P over a NIXL notification; P then pushes the KV to D via
  NIXL WRITE.

Launch multiple vLLM instances configured with ``NixlPushConnector`` and
matching ``engine_id`` / ``side_channel_port``, then start this proxy:

  python3 examples/disaggregated/disaggregated_serving/\
disagg_proxy_pushconnector_demo.py \
       --model $model_name \
       --prefill localhost:8100 \
       --decode  localhost:8200 \
       --prefill-engine-id prefill-engine-001 \
       --prefill-kv-host  10.0.0.1 \
       --prefill-side-channel-port 5600 \
       --prefill-tp-size 1 \
       --prefill-pp-size 1 \
       --port 8000
"""

import argparse
import asyncio
import contextlib
import ipaddress
import itertools
import json
import logging
import os
import sys
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable

import aiohttp
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class SchedulingPolicy(ABC):
    @abstractmethod
    def schedule(self, cycler: itertools.cycle):
        raise NotImplementedError("Scheduling Proxy is not set.")


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    def schedule(self, cycler: itertools.cycle) -> str:
        return next(cycler)


class PushProxy:
    """Push-mode proxy.

    The structure mirrors the pull-mode ``Proxy`` in
    ``disagg_proxy_demo.py``: an APIRouter with ``/v1/completions``,
    ``/v1/chat/completions``, ``/status`` and ``/instances/add``, plus
    round-robin scheduling across multiple P / D instances.

    Push-specific differences are confined to the request-handling
    methods (``create_completion`` / ``create_chat_completion``):

    * D's ``kv_transfer_params`` is built from CLI-provided P
      coordinates instead of being derived from P's response.
    * P and D requests are issued concurrently — D registers blocks and
      waits while P prefills and pushes.
    """

    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        scheduling_policy: SchedulingPolicy,
        prefill_engine_id: str,
        prefill_kv_host: str,
        prefill_side_channel_port: int,
        prefill_tp_size: int,
        prefill_pp_size: int,
        concurrent_prefill_decode: bool = False,
        custom_create_completion: Callable[[Request], StreamingResponse] | None = None,
        custom_create_chat_completion: Callable[[Request], StreamingResponse]
        | None = None,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.scheduling_policy = scheduling_policy
        self.concurrent_prefill_decode = concurrent_prefill_decode

        # Push-mode metadata: D needs P's coordinates up-front. Pull mode
        # learns these from P's response; push mode uses CLI args because
        # D issues its registration before P responds.
        self.push_metadata = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": prefill_engine_id,
            "remote_host": prefill_kv_host,
            "remote_port": prefill_side_channel_port,
            "tp_size": prefill_tp_size,
            "pp_size": prefill_pp_size,
        }

        self.custom_create_completion = custom_create_completion
        self.custom_create_chat_completion = custom_create_chat_completion
        self.router = APIRouter()
        self.setup_routes()

    # ── routes ──────────────────────────────────────────────────────── #

    def setup_routes(self):
        self.router.post(
            "/v1/completions", dependencies=[Depends(self.validate_json_request)]
        )(
            self.custom_create_completion
            if self.custom_create_completion
            else self.create_completion
        )
        self.router.post(
            "/v1/chat/completions", dependencies=[Depends(self.validate_json_request)]
        )(
            self.custom_create_chat_completion
            if self.custom_create_chat_completion
            else self.create_chat_completion
        )
        self.router.get("/status", response_class=JSONResponse)(self.get_status)

    async def validate_json_request(self, raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail="Unsupported Media Type: Only 'application/json' is allowed",
            )

    # ── HTTP forwarding ─────────────────────────────────────────────── #

    async def forward_request(
        self, url, data, headers, use_chunked=True, raise_on_4xx=False
    ):
        """Forward a request and stream the response body.

        Args:
            raise_on_4xx: When True, treat 4xx responses as errors and raise
                HTTPException instead of yielding the body. Use this for
                prefill (P) requests where a 4xx means the NIXL Write will
                never happen. Leave False (default) for decode (D) requests
                where 4xx bodies should be forwarded to the client as-is.
        """
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                async with session.post(
                    url=url, json=data, headers=headers
                ) as response:
                    if 400 <= response.status < 500 and raise_on_4xx:
                        error_content = await response.text()
                        with contextlib.suppress(json.JSONDecodeError):
                            error_content = json.loads(error_content)
                        logger.error(
                            "Request failed with status %s: %s",
                            response.status,
                            error_content,
                        )
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
                    if 200 <= response.status < 300 or 400 <= response.status < 500:
                        if use_chunked:
                            async for chunk_bytes in response.content.iter_chunked(
                                1024
                            ):
                                yield chunk_bytes
                        else:
                            yield await response.read()
                    else:
                        error_content = await response.text()
                        with contextlib.suppress(json.JSONDecodeError):
                            error_content = json.loads(error_content)
                        logger.error(
                            "Request failed with status %s: %s",
                            response.status,
                            error_content,
                        )
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError occurred: %s", str(e))
                raise HTTPException(
                    status_code=502,
                    detail="Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    def schedule(self, cycler: itertools.cycle) -> str:
        return self.scheduling_policy.schedule(cycler)

    async def get_status(self):
        return {
            "mode": "push",
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
            "prefill_engine_id": self.push_metadata["remote_engine_id"],
            "prefill_kv_host": self.push_metadata["remote_host"],
            "prefill_side_channel_port": self.push_metadata["remote_port"],
            "prefill_tp_size": self.push_metadata["tp_size"],
            "prefill_pp_size": self.push_metadata["pp_size"],
        }

    # ── push-mode request handling ──────────────────────────────────── #

    def _build_decode_kv_params(self, request_id: str) -> dict:
        """Push-mode kv_transfer_params for D.

        ``remote_block_ids`` is intentionally omitted: D allocates its
        own blocks and registers them with P; P determines the
        prefill-side block IDs and ships them via the WRITE.
        """
        params = self.push_metadata.copy()
        params["remote_request_id"] = request_id
        return params

    def _common_headers(self, request_id: str) -> dict:
        h = {"X-Request-Id": request_id}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    async def _push_completion(self, raw_request: Request, path: str):
        """Shared body for /v1/completions and /v1/chat/completions.

        Push mode fires P and D concurrently:
          * P runs a normal prefill (max_tokens=1, do_remote_decode=True).
          * D runs the decode (do_remote_prefill=True, no remote_block_ids).

        D blocks waiting for P's WRITE; the response streamed back to the
        client is the decode output from D.
        """
        request = await raw_request.json()
        request_id = str(uuid.uuid4())

        # Prefill leg (max_tokens=1, signals P to keep KV around for D).
        prefill_request = request.copy()
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        prefill_request["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }

        # Decode leg (push mode: no remote_block_ids).
        decode_request = request.copy()
        decode_request["kv_transfer_params"] = self._build_decode_kv_params(request_id)

        prefill_instance = self.schedule(self.prefill_cycler)
        decode_instance = self.schedule(self.decode_cycler)
        headers = self._common_headers(request_id)

        if self.concurrent_prefill_decode:
            return await self._push_completion_concurrent(
                path,
                prefill_instance,
                decode_instance,
                prefill_request,
                decode_request,
                headers,
                request_id,
            )

        # Sequential mode (default): wait for P's response before firing D.
        async for _ in self.forward_request(
            f"http://{prefill_instance}{path}", prefill_request, headers
        ):
            continue

        generator = self.forward_request(
            f"http://{decode_instance}{path}", decode_request, headers
        )
        return StreamingResponse(generator, media_type="application/json")

    async def _push_completion_concurrent(
        self,
        path: str,
        prefill_instance: str,
        decode_instance: str,
        prefill_request: dict,
        decode_request: dict,
        headers: dict,
        request_id: str,
    ) -> StreamingResponse:
        """Concurrent P+D dispatch with prefill failure propagation.

        ``asyncio.wait`` races P's task against D's first response chunk and
        handles two failure windows:

        * **P fails before D produces output** (connection refused, mid-prefill
          crash, etc.): ``p_exc`` is set, D's request is cancelled, and a 502
          is returned immediately — avoiding a multi-second wait on D's
          watchdog timeout.

        * **P fails after D has started streaming**: the client already has
          tokens; no graceful abort is possible. The error is logged and D's
          stream continues until D's own watchdog fires.
        """
        p_exc: list[BaseException] = []

        async def _run_prefill() -> None:
            try:
                async for _ in self.forward_request(
                    f"http://{prefill_instance}{path}",
                    prefill_request,
                    headers,
                    raise_on_4xx=True,
                ):
                    pass
            except Exception as exc:
                p_exc.append(exc)
                logger.error(
                    "Prefill request failed (request_id=%s, instance=%s): %s",
                    request_id,
                    prefill_instance,
                    exc,
                )

        p_task = asyncio.create_task(_run_prefill())

        # Race P's task against D's first response chunk.
        d_iter = self.forward_request(
            f"http://{decode_instance}{path}", decode_request, headers
        ).__aiter__()

        first_chunk_task: asyncio.Task = asyncio.create_task(
            d_iter.__anext__()  # type: ignore[arg-type]
        )

        done, _ = await asyncio.wait(
            [p_task, first_chunk_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # P failed — abort D regardless of whether D has produced
        # output yet. We cannot reliably tell whether the NIXL Write succeeded
        # when P's HTTP request failed, so aborting is the safe choice.
        if p_exc:
            first_chunk_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await first_chunk_task
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Prefill failed, decode aborted "
                    f"(request_id={request_id}): {p_exc[0]}"
                ),
            )

        # D has its first chunk, or P succeeded before D (normal fast-P case).
        # Stream remaining chunks; clean up p_task on exit.
        async def _stream_rest() -> ...:
            try:
                # Await first_chunk_task: returns immediately if D already has
                # a chunk ready; waits if P finished before D (normal case).
                try:
                    yield await first_chunk_task
                except StopAsyncIteration:
                    return
                # Stream the rest.
                async for chunk in d_iter:
                    if p_exc:
                        # P failed after streaming started — log
                        # once and continue; client already has partial output.
                        logger.warning(
                            "Prefill failed while decode is streaming "
                            "(request_id=%s): %s — D watchdog will clean up.",
                            request_id,
                            p_exc[0],
                        )
                        p_exc.clear()  # suppress repeated log per chunk
                    yield chunk
            finally:
                if not p_task.done():
                    p_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await p_task

        return StreamingResponse(_stream_rest(), media_type="application/json")

    async def create_completion(self, raw_request: Request):
        try:
            return await self._push_completion(raw_request, "/v1/completions")
        except HTTPException:
            raise
        except Exception:
            exc_info = sys.exc_info()
            print("Error occurred in disagg push proxy server")
            print(exc_info)
            raise

    async def create_chat_completion(self, raw_request: Request):
        try:
            return await self._push_completion(raw_request, "/v1/chat/completions")
        except HTTPException:
            raise
        except Exception:
            exc_info = sys.exc_info()
            error_messages = [str(e) for e in exc_info if e]
            print("Error occurred in disagg push proxy server")
            print(error_messages)
            return StreamingResponse(
                content=iter(error_messages), media_type="text/event-stream"
            )


class PushProxyServer:
    def __init__(
        self,
        args: argparse.Namespace,
        scheduling_policy: SchedulingPolicy | None = None,
        create_completion: Callable[[Request], StreamingResponse] | None = None,
        create_chat_completion: Callable[[Request], StreamingResponse] | None = None,
    ):
        self.validate_parsed_serve_args(args)
        self.port = args.port
        self.proxy_instance = PushProxy(
            prefill_instances=[] if args.prefill is None else args.prefill,
            decode_instances=[] if args.decode is None else args.decode,
            model=args.model,
            scheduling_policy=(
                scheduling_policy
                if scheduling_policy is not None
                else RoundRobinSchedulingPolicy()
            ),
            prefill_engine_id=args.prefill_engine_id,
            prefill_kv_host=args.prefill_kv_host,
            prefill_side_channel_port=args.prefill_side_channel_port,
            prefill_tp_size=args.prefill_tp_size,
            prefill_pp_size=args.prefill_pp_size,
            concurrent_prefill_decode=args.concurrent_prefill_decode,
            custom_create_completion=create_completion,
            custom_create_chat_completion=create_chat_completion,
        )

    def validate_parsed_serve_args(self, args: argparse.Namespace):
        if not args.prefill:
            raise ValueError("Please specify at least one prefill node.")
        if not args.decode:
            raise ValueError("Please specify at least one decode node.")
        if not args.prefill_engine_id:
            raise ValueError(
                "--prefill-engine-id is required in push mode (it must match "
                "the engine_id passed to the prefill vLLM instance via "
                "--kv-transfer-config)."
            )
        if not args.prefill_kv_host:
            raise ValueError(
                "--prefill-kv-host is required in push mode (the IP / host "
                "that the prefill vLLM advertises on its NIXL side channel)."
            )
        self.validate_instances(args.prefill)
        self.validate_instances(args.decode)

    def validate_instances(self, instances: list):
        for instance in instances:
            if len(instance.split(":")) != 2:
                raise ValueError(f"Invalid instance format: {instance}")
            host, port = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port)
                if not (0 < port < 65536):
                    raise ValueError(f"Invalid port number in instance: {instance}")
            except Exception as e:
                raise ValueError(f"Invalid instance {instance}: {str(e)}") from e

    def run_server(self):
        app = FastAPI()
        app.include_router(self.proxy_instance.router)
        config = uvicorn.Config(app, port=self.port, loop="uvloop")
        server = uvicorn.Server(config)
        server.run()


def parse_args():
    parser = argparse.ArgumentParser("vLLM disaggregated push-mode proxy server.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")

    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )

    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number",
    )

    # Push-mode specific: P's coordinates that D needs in advance.
    parser.add_argument(
        "--prefill-engine-id",
        type=str,
        required=True,
        help=(
            "engine_id of the prefill vLLM instance (must match "
            "--kv-transfer-config engine_id on the prefill server)"
        ),
    )
    parser.add_argument(
        "--prefill-kv-host",
        type=str,
        required=True,
        help=(
            "IP / host the prefill vLLM advertises on its NIXL side "
            "channel (VLLM_NIXL_SIDE_CHANNEL_HOST)"
        ),
    )
    parser.add_argument(
        "--prefill-side-channel-port",
        type=int,
        default=5600,
        help="NIXL side channel port on the prefill node "
        "(VLLM_NIXL_SIDE_CHANNEL_PORT, default 5600)",
    )
    parser.add_argument(
        "--prefill-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size of the prefill vLLM instance",
    )
    parser.add_argument(
        "--prefill-pp-size",
        type=int,
        default=1,
        help="Pipeline parallel size of the prefill vLLM instance",
    )
    parser.add_argument(
        "--concurrent-prefill-decode",
        action="store_true",
        default=False,
        help=(
            "Fire prefill (P) and decode (D) requests concurrently instead of "
            "waiting for P's HTTP response before sending D's request. "
            "Hides P's HTTP round-trip latency and overlaps D's block "
            "allocation with P's prefill compute. Default: sequential (False)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    proxy_server = PushProxyServer(args=args)
    proxy_server.run_server()
