"""Command line interface:  python -m imvbench <command> [options]

  tasks       list the (dataset, fusion, imputer) configurations and their task indices
  precompute  train / cache the CPM-Nets and DCP imputers for all rates and seeds
  tune        independent Optuna search per configuration
  evaluate    retrain each tuned configuration on the evaluation seeds and test it
  run         tune, then evaluate
  status      progress of every study and evaluation
  report      tables and paired tests -> results/report.md, results/cells.csv
"""

import argparse
import itertools
import os
import sys

from imvbench import ALL_IMPUTERS, FUSIONS, config, configurations
from imvbench.datasets import DATASETS

DEFAULT_DATASETS = ["har", "sensit", "modelnet", "youtube", "mosei"]


def selected(args):
    confs = configurations(args.datasets)
    if getattr(args, "fusions", None):
        confs = [c for c in confs if c[1] in args.fusions]
    if getattr(args, "imputers", None):
        confs = [c for c in confs if c[2] in args.imputers]
    if getattr(args, "task_index", None) is not None:
        if not 0 <= args.task_index < len(confs):
            sys.exit(f"--task-index {args.task_index} out of range (0..{len(confs) - 1})")
        confs = [confs[args.task_index]]
    return confs


def make_runner(args, cache_size=None):
    from imvbench.runner import Runner
    return Runner(args.epochs, args.batch_size, cache_size or args.cache_size,
                  getattr(args, "workers", 1), getattr(args, "threads", None))


# --------------------------------------------------------------------------------------
def cmd_tasks(args):
    for i, (d, f, imp) in enumerate(selected(args)):
        print(f"{i:3d}  {d:10s} {f:6s} {imp}")


def cmd_precompute(args):
    from imvbench.tune import TUNE_SEEDS, TUNE_RATES
    from imvbench.evaluate import EVAL_SEEDS, EVAL_RATES
    rates = sorted(set(args.rates or (TUNE_RATES + EVAL_RATES)))
    seeds = args.seeds or (TUNE_SEEDS + EVAL_SEEDS)
    pairs = list(itertools.product(args.datasets, args.imputers))
    if args.task_index is not None:
        pairs = [pairs[args.task_index]]
    runner = make_runner(args, cache_size=1)
    for dataset, imputer in pairs:
        for rate in rates:
            for seed in seeds:
                runner.data(dataset, imputer, rate, seed)
                print(f"[{dataset}/{imputer}] p={rate:g} seed={seed} ready", flush=True)


def cmd_tune(args):
    from imvbench.tune import tune
    runner = make_runner(args)
    for d, f, imp in selected(args):
        tune(runner, d, f, imp, args.trials, args.tune_rates, args.tune_seeds)
    runner.close()


def cmd_evaluate(args):
    from imvbench.evaluate import evaluate
    runner = make_runner(args)
    for d, f, imp in selected(args):
        evaluate(runner, d, f, imp, args.eval_rates, args.eval_seeds)
    runner.close()


def cmd_run(args):
    from imvbench.evaluate import evaluate
    from imvbench.tune import tune
    runner = make_runner(args)
    for d, f, imp in selected(args):
        tune(runner, d, f, imp, args.trials, args.tune_rates, args.tune_seeds)
        evaluate(runner, d, f, imp, args.eval_rates, args.eval_seeds)
    runner.close()


def cmd_status(args):
    from imvbench.evaluate import eval_path, read_runs
    from imvbench.tune import finished_trials, get_study
    n_eval = len(args.eval_rates) * len(args.eval_seeds)
    print(f"{'#':>3}  {'dataset':10s} {'fusion':6s} {'imputer':9s} {'trials':>9s} {'eval':>7s}  best val")
    for i, (d, f, imp) in enumerate(selected(args)):
        study = get_study(d, f, imp, create=False)
        n = finished_trials(study)
        try:
            best = f"{study.best_value:.2f}" if study else ""
        except ValueError:
            best = ""
        ne = len(read_runs(eval_path(d, f, imp)))
        print(f"{i:3d}  {d:10s} {f:6s} {imp:9s} {n:4d}/{args.trials:<4d} {ne:3d}/{n_eval:<3d}  {best}")


def cmd_report(args):
    from imvbench.report import build_report
    print(build_report(args.datasets))
    print(f"\nWritten to {os.path.join(config.RESULTS_DIR, 'report.md')} and cells.csv")


# --------------------------------------------------------------------------------------
def main(argv=None):
    from imvbench.evaluate import EVAL_RATES, EVAL_SEEDS
    from imvbench.tune import N_TRIALS, TUNE_RATES, TUNE_SEEDS

    p = argparse.ArgumentParser(prog="imvbench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, select=True, train=True):
        sp.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
        if select:
            sp.add_argument("--fusions", nargs="+", choices=FUSIONS)
            sp.add_argument("--imputers", nargs="+", choices=ALL_IMPUTERS,
                            help="e.g. zero cpm+mask drop")
            sp.add_argument("--task-index", type=int, help="run only this configuration (e.g. the index of an array job)")
        sp.add_argument("--trials", type=int, default=N_TRIALS)
        sp.add_argument("--tune-rates", nargs="+", type=float, default=TUNE_RATES)
        sp.add_argument("--tune-seeds", nargs="+", type=int, default=TUNE_SEEDS)
        sp.add_argument("--eval-rates", nargs="+", type=float, default=EVAL_RATES)
        sp.add_argument("--eval-seeds", nargs="+", type=int, default=EVAL_SEEDS)
        if train:
            sp.add_argument("--epochs", type=int, default=20)
            sp.add_argument("--batch-size", type=int, default=32)
            sp.add_argument("--workers", type=int, default=1, help="parallel processes (one per seed)")
            sp.add_argument("--threads", type=int, help="threads per worker (default: CPUs / workers)")
            sp.add_argument("--cache-size", type=int, default=8, help="prepared datasets kept in memory")
        sp.add_argument("--verbose", action="store_true")

    common(sub.add_parser("tasks", help="list configurations"), train=False)
    common(sub.add_parser("tune", help="hyperparameter search"))
    common(sub.add_parser("evaluate", help="final evaluation"))
    common(sub.add_parser("run", help="tune + evaluate"))
    common(sub.add_parser("status", help="progress"), train=False)
    rp = sub.add_parser("report", help="tables and tests")
    rp.add_argument("--datasets", nargs="+")
    pp = sub.add_parser("precompute", help="train the CPM-Nets / DCP imputers")
    pp.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    pp.add_argument("--imputers", nargs="+", default=["cpm", "dcp"], choices=["cpm", "dcp"])
    pp.add_argument("--rates", nargs="+", type=float)
    pp.add_argument("--seeds", nargs="+", type=int)
    pp.add_argument("--task-index", type=int, help="only this (dataset, imputer) pair")
    pp.add_argument("--epochs", type=int, default=20)
    pp.add_argument("--batch-size", type=int, default=32)
    pp.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)
    for d in getattr(args, "datasets", None) or []:
        if d not in DATASETS:
            p.error(f"unknown dataset '{d}' (registered: {', '.join(sorted(DATASETS))})")
    if not getattr(args, "verbose", False):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    {"tasks": cmd_tasks, "precompute": cmd_precompute, "tune": cmd_tune, "evaluate": cmd_evaluate,
     "run": cmd_run, "status": cmd_status, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
