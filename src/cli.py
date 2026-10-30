"""Command line entry point: python -m src.cli <command> --task <task> [options]

Commands (task: classification)
  train           train a model from a config
  tune            run Grid Search or Random Search
  compare-tuning  compare Grid vs Random results
  evaluate        metrics + confusion matrix of a checkpoint on val/test
  gradcam         Grad-CAM figures for correct and wrong predictions
  compare-runs    plot several training runs on the same axes
  predict         top-k species for one image

Any config value can be changed with --set key=value (dotted keys).
"""
import argparse
import json
from pathlib import Path

TASKS = ["classification"]


def cmd_train(args):
    from src.training.train_classifier import train_classifier
    from src.tuning.search import best_params
    from src.utils.config import load_config

    overrides = []
    if args.best_from:
        params = best_params(args.best_from)
        print(f"Using best tuned hyperparameters: {params}")
        overrides += [f"{k}={v}" for k, v in params.items()]
    cfg = load_config(args.config, overrides + args.set)
    train_classifier(cfg)


def cmd_tune(args):
    from src.tuning.search import run_search
    from src.utils.config import load_config

    tune_cfg = load_config(args.config)
    run_search(tune_cfg, args.method, extra_overrides=args.set)


def cmd_compare_tuning(args):
    from src.tuning.search import compare_searches

    compare_searches(args.results_dir)


def cmd_evaluate(args):
    from src.evaluation.evaluate import evaluate_checkpoint

    evaluate_checkpoint(args.checkpoint, args.split, args.set, args.output_dir)


def cmd_gradcam(args):
    from src.visualization.gradcam import run_gradcam

    run_gradcam(args.checkpoint, args.split, args.n_correct, args.n_wrong,
                args.set, args.output_dir)


def cmd_compare_runs(args):
    from src.visualization.plots import plot_run_comparison

    names = args.names or [Path(r).name for r in args.runs]
    if len(names) != len(args.runs):
        raise SystemExit("--names must have one name per run directory")
    histories = {}
    for name, run_dir in zip(names, args.runs):
        with open(Path(run_dir) / "history.json", "r", encoding="utf-8") as f:
            histories[name] = json.load(f)
    path = plot_run_comparison(histories, args.output)
    for name, history in histories.items():
        best = max(history, key=lambda r: r["val_acc"])
        print(f"{name}: best val acc {best['val_acc']:.4f} at epoch {best['epoch']} "
              f"(train acc {best['train_acc']:.4f})")
    print(f"Saved comparison to {path}")


def cmd_predict(args):
    import torch
    from PIL import Image

    from src.data.transforms import build_eval_transform
    from src.models.checkpoint import load_classifier
    from src.utils.seed import get_device

    device = get_device()
    model, checkpoint = load_classifier(args.checkpoint, device)
    stats = checkpoint["norm_stats"]
    transform = build_eval_transform(checkpoint["config"]["data"], stats["mean"], stats["std"])
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1)[0]
    top = torch.topk(probs, args.topk)
    print(f"Top-{args.topk} predictions for {args.image}:")
    for rank, (p, idx) in enumerate(zip(top.values, top.indices), start=1):
        print(f"  {rank}. {checkpoint['class_names'][idx]:<35} {p.item():.3f}")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--task", required=True, choices=TASKS)
        p.set_defaults(func=func)
        return p

    def add_set(p):
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                       help="override config values, e.g. train.lr=0.003")

    p = add("train", cmd_train, "train a model")
    p.add_argument("--config", required=True)
    p.add_argument("--best-from", metavar="TUNING_DIR",
                   help="apply the best hyperparameters found by tuning in this directory")
    add_set(p)

    p = add("tune", cmd_tune, "hyperparameter search")
    p.add_argument("--config", required=True, help="tuning config, e.g. configs/tune_scratch.yaml")
    p.add_argument("--method", required=True, choices=["grid", "random"])
    add_set(p)

    p = add("compare-tuning", cmd_compare_tuning, "compare grid vs random search")
    p.add_argument("--results-dir", default="outputs/tuning")

    for name, func, help_text in (
        ("evaluate", cmd_evaluate, "evaluate a checkpoint"),
        ("gradcam", cmd_gradcam, "Grad-CAM figures"),
    ):
        p = add(name, func, help_text)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--split", default="test", choices=["val", "test"])
        p.add_argument("--output-dir")
        add_set(p)
        if name == "gradcam":
            p.add_argument("--n-correct", type=int, default=8)
            p.add_argument("--n-wrong", type=int, default=8)

    p = add("compare-runs", cmd_compare_runs, "compare training runs")
    p.add_argument("--runs", nargs="+", required=True, help="run directories with history.json")
    p.add_argument("--names", nargs="+", help="legend names, one per run")
    p.add_argument("--output", default="outputs/classification/run_comparison.png")

    p = add("predict", cmd_predict, "classify one image")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--topk", type=int, default=5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
