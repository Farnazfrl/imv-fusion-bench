"""Independent hyperparameter search for every (dataset, fusion, imputer) configuration.

Search space   hidden width h in {32, 64, 128, 256, 512}
               learning rate in [1e-4, 1e-2]       (log-uniform)
               weight decay  in [1e-6, 1e-1]       (log-uniform, AdamW)
Objective      mean best-validation accuracy over the tuning missing rates (0.3, 0.7) and the
               tuning seeds (7, 11, 13), which are disjoint from the evaluation seeds.
Search         Optuna TPE (seed 0), 60 trials, median pruning after the first missing rate.

Each study lives in its own SQLite file results/studies/<dataset>/<fusion>_<imputer>.db, so
parallel array tasks never write to the same file and an interrupted study resumes.
"""

import os

import optuna

from imvbench import config

HIDDEN_CHOICES = [32, 64, 128, 256, 512]
LR_RANGE = (1e-4, 1e-2)
WD_RANGE = (1e-6, 1e-1)
TUNE_SEEDS = [7, 11, 13]
TUNE_RATES = [0.3, 0.7]
N_TRIALS = 60
SAMPLER_SEED = 0
FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def study_name(dataset, fusion, imputer):
    return f"imvb__{dataset}__{fusion}__{imputer}"


def study_path(dataset, fusion, imputer):
    return os.path.join(config.RESULTS_DIR, "studies", dataset, f"{fusion}_{imputer}.db")


def get_study(dataset, fusion, imputer, create=True):
    path = study_path(dataset, fusion, imputer)
    if not create and not os.path.exists(path):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{os.path.abspath(path)}",
                                         engine_kwargs={"connect_args": {"timeout": 60}})
    # each file holds one study: reuse its name if it exists (studies made by earlier versions
    # of the code carry a different name prefix)
    existing = optuna.study.get_all_study_names(storage)
    name = existing[0] if len(existing) == 1 else study_name(dataset, fusion, imputer)
    if not create:
        try:
            return optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            return None
    return optuna.create_study(
        study_name=name, storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))


def make_objective(runner, dataset, fusion, imputer, rates, seeds):
    def objective(trial):
        hp = dict(hidden=trial.suggest_categorical("hidden", HIDDEN_CHOICES),
                  lr=trial.suggest_float("lr", *LR_RANGE, log=True),
                  weight_decay=trial.suggest_float("weight_decay", *WD_RANGE, log=True))
        accs, detail = [], {}
        for step, rate in enumerate(rates):
            jobs = [dict(dataset=dataset, fusion=fusion, imputer=imputer, rate=rate, seed=s, **hp)
                    for s in seeds]
            for seed, res in zip(seeds, runner.map(jobs)):
                accs.append(res["val_acc"])
                detail[f"{rate:g}/{seed}"] = [res["val_acc"], res["best_epoch"]]
            trial.report(sum(accs) / len(accs), step=step)
            if step < len(rates) - 1 and trial.should_prune():
                trial.set_user_attr("val_runs", detail)
                raise optuna.TrialPruned()
        trial.set_user_attr("val_runs", detail)
        return sum(accs) / len(accs)
    return objective


def finished_trials(study):
    return len(study.get_trials(deepcopy=False, states=FINISHED)) if study else 0


def tune(runner, dataset, fusion, imputer, n_trials=N_TRIALS, rates=TUNE_RATES, seeds=TUNE_SEEDS):
    study = get_study(dataset, fusion, imputer)
    study.set_user_attr("rates", list(rates))
    study.set_user_attr("seeds", list(seeds))
    remaining = n_trials - finished_trials(study)
    if remaining > 0:
        study.optimize(make_objective(runner, dataset, fusion, imputer, rates, seeds), n_trials=remaining,
                       callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=FINISHED)])
    best = study.best_trial
    print(f"[{study.study_name}] best h={best.params['hidden']} lr={best.params['lr']:.2e} "
          f"wd={best.params['weight_decay']:.2e} val={best.value:.2f}% "
          f"({finished_trials(study)} trials)", flush=True)
    return best.params


def best_params(dataset, fusion, imputer):
    """Best hyperparameters of a finished study, or None."""
    study = get_study(dataset, fusion, imputer, create=False)
    if study is None:
        return None
    try:
        best = study.best_trial
    except ValueError:
        return None
    return dict(best.params, val_acc=best.value, trials=finished_trials(study))
