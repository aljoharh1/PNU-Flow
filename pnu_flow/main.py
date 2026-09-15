"""
Citations (open-source):
- argparse (Python stdlib): https://docs.python.org/3/library/argparse.html
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pnu_flow.pipelines.inference_pipeline import query_route
from pnu_flow.pipelines.training_pipeline  import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    p   = argparse.ArgumentParser(description="PNU-Flow Phase 3 CLI")
    sub = p.add_subparsers(dest="mode", required=True)

    sub.add_parser("train", help="Run full simulation + LSTM training pipeline")

    demo = sub.add_parser("demo", help="Train then immediately infer one route")
    demo.add_argument("--from", dest="source",      default="main_entrance")
    demo.add_argument("--to",   dest="destination", default="lecture_hall_201")

    infer = sub.add_parser("infer", help="Run inference (requires prior train run)")
    infer.add_argument("--from", dest="source",      required=True)
    infer.add_argument("--to",   dest="destination", required=True)
    infer.add_argument("--time", dest="query_time",  default=None,
                       help="ISO datetime, e.g. 2026-03-30T10:30:00")
    return p


def _print(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        _print(run_training_pipeline())

    elif args.mode == "demo":
        train_out = run_training_pipeline()
        infer_out = query_route(source=args.source,
                                destination=args.destination)
        _print({"train": train_out, "infer": infer_out})

    elif args.mode == "infer":
        qt = datetime.fromisoformat(args.query_time) if args.query_time else None
        _print(query_route(source=args.source,
                           destination=args.destination,
                           query_time=qt))


if __name__ == "__main__":
    main()
