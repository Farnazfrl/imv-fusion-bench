"""Runs training jobs, sequentially or in parallel worker processes.

Prepared data (splits + imputations) is kept in a small LRU cache per process, so the trials of
a study do not rebuild it. With --workers n > 1 the jobs of a batch are spread over n long-lived
processes (job i -> worker i % n); each gets (available CPUs) // n threads. Jobs are always
ordered seed-innermost, so every worker keeps seeing the same seed and reuses its cache.
"""

import atexit
import itertools
import math
import multiprocessing as mp
import os
import queue
import traceback
from collections import OrderedDict

import torch

from imvbench import config


class Runner:
    def __init__(self, epochs: int, batch_size: int, cache_size: int = 8,
                 workers: int = 1, threads: int = None):
        self.epochs, self.batch_size, self.cache_size = epochs, batch_size, cache_size
        self.cache: "OrderedDict[tuple, object]" = OrderedDict()
        self.pool = None
        if workers > 1:
            cpus = config.available_cpus()
            threads = threads or max(1, cpus // workers)
            self.pool = _Pool(workers, dict(epochs=epochs, batch_size=batch_size,
                                            cache_size=max(1, math.ceil(cache_size / workers))), threads)
            print(f"{workers} worker processes, {threads} threads each", flush=True)

    # ---- data ------------------------------------------------------------------------
    def data(self, dataset: str, imputer: str, rate: float, seed: int):
        from imvbench.data import build_data
        from imvbench.train import set_seed
        from imvbench import base_imputer
        base = base_imputer(imputer)                 # "cpm+mask" uses the data of "cpm"
        key = (dataset, "zero" if base == "drop" else base, float(rate), int(seed))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        set_seed(seed)                       # imputer training is seeded as well
        value = build_data(dataset, key[1], rate, seed, device=config.DEVICE)
        self.cache[key] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    # ---- one job ---------------------------------------------------------------------
    def run(self, dataset, fusion, imputer, rate, seed, hidden, lr, weight_decay, test=False):
        from imvbench.train import run
        data = self.data(dataset, imputer, rate, seed)
        return run(data, fusion, imputer, hidden, lr, weight_decay, self.epochs, self.batch_size,
                   test=test, device=config.DEVICE)

    # ---- many jobs -------------------------------------------------------------------
    def map(self, jobs):
        """Yield run(**job) for every job, in order. Sequentially each result comes as soon as
        it is ready (so a trial can be pruned early); in parallel all jobs run at once."""
        if self.pool is None:
            for job in jobs:
                yield self.run(**job)
        else:
            yield from self.pool.map(_run_job, jobs)

    def close(self):
        if self.pool is not None:
            self.pool.close()


def _run_job(runner, **job):
    return runner.run(**job)


# --------------------------------------------------------------------------------------
# Worker processes
# --------------------------------------------------------------------------------------
def _worker_main(in_q, out_q, runner_kwargs, threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    runner = Runner(**runner_kwargs)
    while True:
        task = in_q.get()
        if task is None:
            break
        tag, fn, kwargs = task
        try:
            out_q.put((tag, True, fn(runner, **kwargs)))
        except BaseException:
            out_q.put((tag, False, traceback.format_exc()))


class _Pool:
    def __init__(self, n, runner_kwargs, threads):
        ctx = mp.get_context("spawn")
        self.out_q = ctx.Queue()
        self.in_qs = [ctx.Queue() for _ in range(n)]
        self.procs = [ctx.Process(target=_worker_main, daemon=True,
                                  args=(q, self.out_q, runner_kwargs, threads)) for q in self.in_qs]
        for p in self.procs:
            p.start()
        self._batches = itertools.count()
        atexit.register(self.close)

    def map(self, fn, jobs):
        batch = next(self._batches)
        for i, job in enumerate(jobs):
            self.in_qs[i % len(self.in_qs)].put(((batch, i), fn, job))
        results, pending = [None] * len(jobs), len(jobs)
        while pending:
            try:
                (b, i), ok, value = self.out_q.get(timeout=30)
            except queue.Empty:
                dead = [p for p in self.procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"A worker process died (exit code {dead[0].exitcode}); "
                                       "most likely out of memory: raise --mem or lower --workers.")
                continue
            if b != batch:
                continue
            if not ok:
                raise RuntimeError("Error in a worker process:\n" + value)
            results[i] = value
            pending -= 1
        return results

    def close(self):
        for q in self.in_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
