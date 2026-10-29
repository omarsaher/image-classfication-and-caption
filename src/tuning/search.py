"""Grid Search and Random Search over a few hyperparameters.

- Grid Search tries every combination of a small list of values per parameter.
- Random Search samples each parameter from a range (log-uniform for scales
  like learning rate), so with the same number of trials it tries more
  distinct values of each parameter.

Every trial is a short training run logged as its own experiment. Results
are appended to ``<output_dir>/<method>_results.json`` after each trial, so
an interrupted search keeps finished trials and resumes from the next one.
"""
import itertools
import json
import math
import random
import time
from pathlib import Path

from src.training.train_classifier import train_classifier
from src.utils.config import apply_overrides, load_config


def grid_trials(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def random_trials(space, n_trials, seed):
    rng = random.Random(seed)
    return [{key: _sample(spec, rng) for key, spec in space.items()} for _ in range(n_trials)]


def _sample(spec, rng):
    kind = spec["type"]
    if kind == "loguniform":
        return float(math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"]))))
    if kind == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if kind == "choice":
        return rng.choice(spec["values"])
    raise ValueError(f"Unknown search space type '{kind}'")


def _as_overrides(params):
    return [f"{key}={value}" for key, value in params.items()]


def run_search(tune_cfg, method, extra_overrides=None):
    """Run every trial of ``method`` ('grid' or 'random'); return the result rows."""
    if method == "grid":
        trials = grid_trials(tune_cfg["grid"])
    elif method == "random":
        rnd = tune_cfg["random"]
        trials = random_trials(rnd["space"], rnd["n_trials"], rnd["seed"])
    else:
        raise ValueError(f"Unknown search method '{method}', expected 'grid' or 'random'")

    output_dir = Path(tune_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{method}_results.json"
    results = _load_results(results_path)
    done = {r["trial"] for r in results}

    base_overrides = _as_overrides(tune_cfg.get("trial_overrides", {})) + list(extra_overrides or [])
    print(f"{method} search: {len(trials)} trials ({len(done)} already done)")

    for i, params in enumerate(trials):
        if i in done:
            continue
        name = f"tune_{method}_{i:02d}"
        cfg = load_config(tune_cfg["base_config"], base_overrides + _as_overrides(params))
        cfg = apply_overrides(cfg, [
            f"experiment.name={name}",
            f"experiment.output_dir={output_dir / method}",
            f"tracking.group=tune_{method}",
        ])
        print(f"\n[{method} {i + 1}/{len(trials)}] {params}")

        start = time.time()
        result = train_classifier(cfg, job_type=f"tune_{method}")
        results.append({
            "trial": i,
            "params": params,
            "best_val_acc": result.best_val_acc,
            "best_epoch": result.best_epoch,
            "time_sec": time.time() - start,
        })
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)

    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"\nBest {method} trial {best['trial']}: val acc {best['best_val_acc']:.4f} {best['params']}")
    return results


def _load_results(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_search(results):
    """Numbers used to compare the two methods."""
    ordered = sorted(results, key=lambda r: r["trial"])
    best = max(ordered, key=lambda r: r["best_val_acc"])
    elapsed = 0.0
    time_to_best = 0.0
    for row in ordered:
        elapsed += row["time_sec"]
        if row is best:
            time_to_best = elapsed
    return {
        "n_trials": len(ordered),
        "best_val_acc": best["best_val_acc"],
        "best_trial": best["trial"],
        "best_params": best["params"],
        "total_time_min": elapsed / 60,
        "time_to_best_min": time_to_best / 60,
        "mean_val_acc": sum(r["best_val_acc"] for r in ordered) / len(ordered),
    }


def compare_searches(results_dir):
    """Compare grid vs random results: summary table (JSON) and plot."""
    from src.visualization.plots import plot_search_comparison

    results_dir = Path(results_dir)
    results_by_method = {}
    for method in ("grid", "random"):
        path = results_dir / f"{method}_results.json"
        if path.exists():
            results_by_method[method] = _load_results(path)
    if not results_by_method:
        raise FileNotFoundError(f"No grid_results.json / random_results.json in {results_dir}")

    summary = {m: summarize_search(r) for m, r in results_by_method.items()}
    overall = max(summary.items(), key=lambda kv: kv[1]["best_val_acc"])
    summary["overall_best"] = {"method": overall[0], **overall[1]}

    with open(results_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    plot_search_comparison(results_by_method, results_dir / "comparison.png")

    for method in results_by_method:
        s = summary[method]
        print(
            f"{method:>6}: best val acc {s['best_val_acc']:.4f} | trials {s['n_trials']} | "
            f"total {s['total_time_min']:.1f} min | best found after {s['time_to_best_min']:.1f} min"
        )
    print(f"Overall best ({overall[0]}): {overall[1]['best_params']}")
    return summary


def best_params(results_dir):
    """Hyperparameters of the best trial across both searches (from comparison.json)."""
    path = Path(results_dir) / "comparison.json"
    if not path.exists():
        compare_searches(results_dir)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["overall_best"]["best_params"]
