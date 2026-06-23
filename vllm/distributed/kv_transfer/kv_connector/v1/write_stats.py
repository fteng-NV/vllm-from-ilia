# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead instrumentation for the P-side KV-transfer path.

Shared by the NIXL and Mooncake connectors. Decomposes per-request latency on
the producer (prefill) side into four orthogonal stages:

  * pending_ms -- request arrived at P, waiting for P's own prefill to finish
                  (parallel control plane only; ~0 for the sequential path)
  * wait_ms    -- queue residence: enqueued -> picked up by a sender worker
  * prepare_ms -- per-connector overhead around the transfer that is NOT the
                  transport itself (descriptor build + issue + any polling)
  * xfer_ms    -- the *true* transport time (NIXL telemetry xferDuration /
                  Mooncake batch_transfer_sync_write duration)

`xfer_ms` is intended to be apples-to-apples across backends (pure transport);
`prepare_ms` captures each backend's non-transport overhead.

Samples are accumulated into fixed in-memory histograms and a windowed summary
is logged every `interval_s`, so the hot path never does per-request logging/IO
(which would itself perturb the GIL / latency we are measuring).

Enabled via kv_connector_extra_config "write_stats" (default false).
"""

import threading
import time

# Log-spaced millisecond bucket edges (upper-inclusive); last bucket is overflow.
_MS_EDGES = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)


class _Hist:
    """Fixed-bucket histogram with count/sum/min/max and percentile estimate."""

    __slots__ = ("edges", "counts", "n", "s", "mn", "mx")

    def __init__(self, edges: tuple[float, ...]):
        self.edges = edges
        self.reset()

    def add(self, v: float) -> None:
        self.n += 1
        self.s += v
        if v < self.mn:
            self.mn = v
        if v > self.mx:
            self.mx = v
        i = 0
        edges = self.edges
        while i < len(edges) and v > edges[i]:
            i += 1
        self.counts[i] += 1

    def mean(self) -> float:
        return self.s / self.n if self.n else 0.0

    def pct(self, p: float) -> float:
        """Approximate percentile = upper edge of the crossing bucket, clamped
        to the observed max (so coarse buckets never report a value > max)."""
        if self.n == 0:
            return 0.0
        target = p * self.n
        cum = 0
        for i, c in enumerate(self.counts):
            cum += c
            if cum >= target:
                edge = self.edges[i] if i < len(self.edges) else self.mx
                return min(edge, self.mx)
        return self.mx

    def reset(self) -> None:
        self.counts = [0] * (len(self.edges) + 1)
        self.n = 0
        self.s = 0.0
        self.mn = float("inf")
        self.mx = 0.0


class WritePathStats:
    """Thread-safe accumulator. Call record() from the worker/sender threads and
    maybe_dump() from a single always-alive loop (NIXL poller / Mooncake sender
    worker). `name` is the log prefix, e.g. "nixl-wstats" or "mc-wstats"."""

    def __init__(
        self,
        logger,
        name: str = "wstats",
        tag: str = "",
        interval_s: float = 5.0,
    ):
        self._logger = logger
        self._name = name
        self._tag = tag
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._pending = _Hist(_MS_EDGES)
        self._wait = _Hist(_MS_EDGES)
        self._prepare = _Hist(_MS_EDGES)
        self._xfer = _Hist(_MS_EDGES)
        self._batch: dict[int, int] = {}
        self._buffered = 0  # requests that waited for P's prefill (pending > 0)
        self._last_dump = time.perf_counter()

    def record(
        self,
        pending_ms: float,
        wait_ms: float,
        prepare_ms: float,
        xfer_ms: float | None,
        batch_size: int,
    ) -> None:
        with self._lock:
            self._pending.add(pending_ms)
            if pending_ms > 0.0:
                self._buffered += 1
            self._wait.add(wait_ms)
            self._prepare.add(prepare_ms)
            if xfer_ms is not None:
                self._xfer.add(xfer_ms)
            self._batch[batch_size] = self._batch.get(batch_size, 0) + 1

    def maybe_dump(self) -> None:
        now = time.perf_counter()
        line = None
        with self._lock:
            if now - self._last_dump >= self._interval_s and self._wait.n > 0:
                self._last_dump = now
                p, w, pr, x = self._pending, self._wait, self._prepare, self._xfer
                batch = dict(sorted(self._batch.items()))
                buffered_pct = 100.0 * self._buffered / w.n if w.n else 0.0
                line = (
                    "[%s%s] n=%d buffered=%.1f%% batch=%s "
                    "pending_ms[mean=%.2f p99=%.2f max=%.2f] "
                    "wait_ms[mean=%.2f p99=%.2f max=%.2f] "
                    "prepare_ms[mean=%.2f p99=%.2f max=%.2f] "
                    "xfer_ms[mean=%.2f p99=%.2f max=%.2f]"
                    % (
                        self._name, self._tag, w.n, buffered_pct, batch,
                        p.mean(), p.pct(0.99), p.mx,
                        w.mean(), w.pct(0.99), w.mx,
                        pr.mean(), pr.pct(0.99), pr.mx,
                        x.mean(), x.pct(0.99), x.mx,
                    )
                )
                self._pending.reset()
                self._wait.reset()
                self._prepare.reset()
                self._xfer.reset()
                self._batch = {}
                self._buffered = 0
        if line is not None:
            self._logger.info(line)


class RecvPathStats:
    """Low-overhead instrumentation for the D-side (consumer) KV-receive path.

    Shared by the NIXL and Mooncake connectors. Decomposes the per-request
    receive latency, measured at the *common engine interface* (start_load_kv ->
    get_finished) so the two connectors are compared on equal footing despite
    different internal mechanisms (NIXL polls notifs; Mooncake awaits a ZMQ
    response). Three timestamps, all taken with perf_counter() in the same
    decode-worker process:

      * t_start  -- D registers the request to receive (start_load_kv / receive_kv)
      * t_known  -- the connector's background mechanism first learns the KV is
                    done (NIXL: WRITE_DONE notif handled; Mooncake: ZMQ response
                    processed)
      * t_report -- the request is first surfaced to the engine via get_finished

    Derived stages:
      * recv_wait_ms   = t_known  - t_start   (wait for P prefill + transfer +
                         raw detection; ~equal across backends -> engine-bound)
      * report_lag_ms  = t_report - t_known   (detection -> engine; this is where
                         "poll vs event-driven" shows up)
      * recv_total_ms  = t_report - t_start   (full D-side-visible receive)

    Same enable switch / dump cadence as WritePathStats:
    kv_connector_extra_config "write_stats" (+ "write_stats_interval_s").
    """

    def __init__(
        self,
        logger,
        name: str = "recv",
        tag: str = "",
        interval_s: float = 5.0,
    ):
        self._logger = logger
        self._name = name
        self._tag = tag
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._wait = _Hist(_MS_EDGES)
        self._lag = _Hist(_MS_EDGES)
        self._total = _Hist(_MS_EDGES)
        self._last_dump = time.perf_counter()

    def record(
        self,
        recv_wait_ms: float,
        report_lag_ms: float,
        recv_total_ms: float,
    ) -> None:
        with self._lock:
            self._wait.add(recv_wait_ms)
            self._lag.add(report_lag_ms)
            self._total.add(recv_total_ms)

    def maybe_dump(self) -> None:
        now = time.perf_counter()
        line = None
        with self._lock:
            if now - self._last_dump >= self._interval_s and self._total.n > 0:
                self._last_dump = now
                w, lg, t = self._wait, self._lag, self._total
                line = (
                    "[%s%s] n=%d "
                    "recv_wait_ms[mean=%.2f p99=%.2f max=%.2f] "
                    "report_lag_ms[mean=%.2f p99=%.2f max=%.2f] "
                    "recv_total_ms[mean=%.2f p99=%.2f max=%.2f]"
                    % (
                        self._name, self._tag, t.n,
                        w.mean(), w.pct(0.99), w.mx,
                        lg.mean(), lg.pct(0.99), lg.mx,
                        t.mean(), t.pct(0.99), t.mx,
                    )
                )
                self._wait.reset()
                self._lag.reset()
                self._total.reset()
        if line is not None:
            self._logger.info(line)


class SchedAdmissionStats:
    """Prefill-side scheduler admission instrumentation (backend-agnostic).

    Answers "why does one connector split the prefill of a concurrent burst
    across steps more than the other?" by measuring, at the *engine scheduler*
    (the common code both connectors share):

      * new_reqs_per_step -- how many fresh requests the scheduler admitted from
        the WAITING queue in a given step (= len(scheduled_new_reqs)). On a
        prefill instance these are prefills; a burst of N that lands in ONE step
        shows up as new_reqs_per_step=N (aligned), while N split across steps
        shows up as several steps of =1 (split).
      * inter_arrival_ms -- gap between consecutive add_request() calls; large
        gaps mean the requests reached the engine staggered (proxy/dispatch),
        small gaps + still-split point at step-cadence instead.

    Gated by env VLLM_SCHED_ADMISSION_STATS (the core scheduler does not see the
    connector's kv_connector_extra_config). Dump cadence: every `interval_s`.
    Only steps that actually admit something (count>0) are recorded, so idle
    polling steps don't swamp the distribution.
    """

    def __init__(
        self,
        logger,
        tag: str = "",
        interval_s: float = 5.0,
        gap_max_ms: float = 500.0,
    ):
        self._logger = logger
        self._tag = tag
        self._interval_s = interval_s
        # Only inter-arrival gaps below this are counted, so the large idle gaps
        # BETWEEN bench runs (10-40s) don't swamp the within-burst gap we want.
        self._gap_max_ms = gap_max_ms
        self._lock = threading.Lock()
        self._per_step: dict[int, int] = {}  # new_reqs_per_step -> occurrences
        self._gap = _Hist(_MS_EDGES)  # within-burst inter-arrival gap (ms)
        self._n_burst_starts = 0  # arrivals after an idle gap (>= gap_max_ms)
        self._last_add: float | None = None
        # Optional arrival-probe (only populated if record_arrival is given the
        # request's frontend arrival_time): decomposes the proxy->add_request gap
        # into "frontend-ingress gap" (arr_gap) and "frontend->engine latency"
        # (pickup_lag), to localize whether a burst's stagger appears upstream of
        # the APIServer or in the APIServer->EngineCore hop.
        self._arr_gap = _Hist(_MS_EDGES)  # gap between consecutive arrival_time
        self._pickup_lag = _Hist(_MS_EDGES)  # arrival_time -> engine pickup (ms)
        self._last_arrival: float | None = None  # prev request's arrival_time (wall)
        # Optional add_request-duration probe: tests whether nixl's larger
        # engine-pickup gap (burst_gap) is caused by a slow add_request (e.g. a
        # connector scheduler-side hook holding up the next request's pickup).
        self._add_req = _Hist(_MS_EDGES)  # full scheduler.add_request() duration
        self._on_new = _Hist(_MS_EDGES)  # connector.on_new_request() portion
        # Per-step busy-window probe: how long the EngineCore main thread is
        # busy per engine step (step_fn = schedule + execute dispatch + waiting
        # for worker output). This is the window during which the input-ingest
        # thread can't hand a freshly-arrived sibling request to the scheduler,
        # so a longer step => bigger chance the 2nd request of a burst misses
        # this step and is split off. build_meta = the connector scheduler-side
        # build_connector_meta() portion of that step.
        self._step = _Hist(_MS_EDGES)
        self._build_meta = _Hist(_MS_EDGES)
        self._last_dump = time.perf_counter()

    def record_arrival(self, arrival_time: float | None = None) -> None:
        """Call from add_request(): record the gap since the previous arrival,
        but only if it is small enough to be within the same burst (gaps >=
        gap_max_ms are idle-between-runs and counted separately as burst starts).
        This is the key metric to tell apart 'requests reach the engine
        staggered' (large within-burst gap) from 'they arrive together but
        schedule() splits them' (tiny within-burst gap).

        `arrival_time` (optional) is the request's frontend ingress timestamp
        (`time.time()` set in the APIServer, carried on the request). When given,
        we also record pickup_lag (arrival_time -> this add_request, i.e.
        frontend+IPC latency) and arr_gap (gap between consecutive arrival_time).
        Comparing arr_gap vs burst_gap tells us whether a burst's stagger is
        already present at frontend ingress or appears only at engine pickup."""
        now = time.perf_counter()
        with self._lock:
            if self._last_add is not None:
                gap_ms = (now - self._last_add) * 1000.0
                if gap_ms < self._gap_max_ms:
                    self._gap.add(gap_ms)
                else:
                    self._n_burst_starts += 1
            self._last_add = now

            if arrival_time is not None:
                # Same host + wall clock on both ends -> direct subtraction OK.
                self._pickup_lag.add(max(0.0, (time.time() - arrival_time) * 1000.0))
                if self._last_arrival is not None:
                    ag = (arrival_time - self._last_arrival) * 1000.0
                    if 0.0 <= ag < self._gap_max_ms:
                        self._arr_gap.add(ag)
                self._last_arrival = arrival_time

    def record_add_timing(self, add_req_ms: float, on_new_ms: float) -> None:
        """Call at the end of add_request(): full add_request duration and the
        connector.on_new_request() portion. Tests whether nixl's larger
        burst_gap comes from a slow add_request vs elsewhere in the engine loop."""
        with self._lock:
            self._add_req.add(add_req_ms)
            self._on_new.add(on_new_ms)

    def record_step_ms(self, step_ms: float) -> None:
        """Call after each _process_engine_step's step_fn(): the per-step busy
        window. Compare nixl vs mooncake -- a longer step is a bigger window for
        the 2nd burst request to miss this step (=> split)."""
        with self._lock:
            self._step.add(step_ms)

    def record_build_meta_ms(self, build_meta_ms: float) -> None:
        """Call around connector.build_connector_meta(): the connector
        scheduler-side per-step portion (a candidate for nixl's longer step)."""
        with self._lock:
            self._build_meta.add(build_meta_ms)

    def record_admission(self, n_new: int) -> None:
        """Call at the end of schedule(): n_new = len(scheduled_new_reqs)."""
        if n_new <= 0:
            return
        with self._lock:
            self._per_step[n_new] = self._per_step.get(n_new, 0) + 1

    def maybe_dump(self) -> None:
        now = time.perf_counter()
        line = None
        with self._lock:
            steps = sum(self._per_step.values())
            if now - self._last_dump >= self._interval_s and steps > 0:
                self._last_dump = now
                g = self._gap
                ag = self._arr_gap
                pl = self._pickup_lag
                ar = self._add_req
                on = self._on_new
                st = self._step
                bm = self._build_meta
                dist = dict(sorted(self._per_step.items()))
                # burst_gap_ms = gap between requests WITHIN a burst at engine
                # pickup (idle gaps between runs excluded). arr_gap_ms = same gap
                # but at FRONTEND ingress (arrival_time); pickup_lag_ms =
                # arrival_time -> engine pickup. burst_starts = #arrivals after idle.
                line = (
                    "[sched-admit%s] admit_steps=%d new_reqs_per_step=%s "
                    "burst_starts=%d burst_gap_ms[n=%d mean=%.2f p99=%.2f max=%.2f] "
                    "arr_gap_ms[n=%d mean=%.2f p99=%.2f max=%.2f] "
                    "pickup_lag_ms[n=%d mean=%.2f p99=%.2f max=%.2f] "
                    "add_req_ms[n=%d mean=%.2f p99=%.2f max=%.2f] "
                    "on_new_req_ms[mean=%.2f p99=%.2f max=%.2f] "
                    "step_ms[n=%d mean=%.2f p99=%.2f max=%.2f] "
                    "build_meta_ms[n=%d mean=%.2f p99=%.2f max=%.2f]"
                    % (
                        self._tag, steps, dist,
                        self._n_burst_starts,
                        g.n, g.mean(), g.pct(0.99), g.mx,
                        ag.n, ag.mean(), ag.pct(0.99), ag.mx,
                        pl.n, pl.mean(), pl.pct(0.99), pl.mx,
                        ar.n, ar.mean(), ar.pct(0.99), ar.mx,
                        on.mean(), on.pct(0.99), on.mx,
                        st.n, st.mean(), st.pct(0.99), st.mx,
                        bm.n, bm.mean(), bm.pct(0.99), bm.mx,
                    )
                )
                self._per_step = {}
                self._gap.reset()
                self._arr_gap.reset()
                self._pickup_lag.reset()
                self._add_req.reset()
                self._on_new.reset()
                self._step.reset()
                self._build_meta.reset()
                self._n_burst_starts = 0
        if line is not None:
            self._logger.info(line)
