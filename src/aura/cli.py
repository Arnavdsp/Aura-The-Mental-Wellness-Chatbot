"""Command line entry point: ``aura serve | train | evaluate | dataset``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aura import __version__
from aura.config import get_settings
from aura.logging import configure_logging, get_logger

log = get_logger(__name__)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "aura.api.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def _dataset(args: argparse.Namespace) -> int:
    from aura.training.data import build_preference_dataset

    train_ds, eval_ds = build_preference_dataset(
        psychology_limit=args.limit, eval_fraction=args.eval_fraction
    )
    print(f"train: {len(train_ds)} pairs; eval: {len(eval_ds) if eval_ds else 0} pairs")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        train_ds.to_json(str(out / "train.jsonl"))
        if eval_ds:
            eval_ds.to_json(str(out / "eval.jsonl"))
        print(f"written to {out}")

    if args.show:
        for row in train_ds.select(range(min(args.show, len(train_ds)))):
            print(json.dumps(row, indent=2)[:1200], "\n")
    return 0


def _train(args: argparse.Namespace) -> int:
    from aura.training.train import TrainingConfig, run

    config = TrainingConfig(
        model_id=args.model,
        output_dir=Path(args.output),
        strategy=args.strategy,
        fallback_to_sft=args.fallback_to_sft,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        psychology_limit=args.limit,
        push_to_hub=args.push_to_hub,
    )
    destination = run(config)
    print(f"adapter saved to {destination}")
    print(f"serve it with: AURA_ADAPTER_PATH={destination} aura serve")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Score a checkpoint (or the running echo engine) on coaching behaviour."""
    import asyncio

    from aura.config import Settings
    from aura.engine.base import GenerationRequest
    from aura.engine.registry import build_engine
    from aura.safety import crisis_message, screen
    from aura.training.evaluate import evaluate

    settings = Settings(engine=args.engine, adapter_path=args.adapter)
    engine = build_engine(settings)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(engine.warmup())

    def generate(prompt: str) -> str:
        assessment = screen(prompt, region=settings.crisis_region)
        if assessment.should_short_circuit:
            return crisis_message(assessment)
        return loop.run_until_complete(
            engine.generate(
                GenerationRequest(system_prompt=_eval_system_prompt(), user_text=prompt)
            )
        )

    report = evaluate(generate)
    print(report.render())
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "mean_score": report.mean_score,
                    "rates": report.rates(),
                    "safety_passed": report.passed_safety,
                    "responses": [
                        {"prompt": s.prompt, "response": s.response, "score": s.total}
                        for s in report.scores
                    ],
                },
                indent=2,
            )
        )
    loop.run_until_complete(engine.shutdown())
    loop.close()
    return 0 if report.passed_safety else 1


def _eval_system_prompt() -> str:
    from aura.prompts import build_system_prompt
    from aura.schemas import AffectSignal

    return build_system_prompt(None, AffectSignal())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aura", description="Aura — a multimodal wellness coach built on Gemma 3n."
    )
    parser.add_argument("--version", action="version", version=f"aura {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the API and web UI")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--reload", action="store_true", help="reload on code changes")
    serve.set_defaults(func=_serve)

    dataset = sub.add_parser("dataset", help="build the preference dataset")
    dataset.add_argument("--limit", type=int, default=3000, help="psychology rows to use")
    dataset.add_argument("--eval-fraction", type=float, default=0.05)
    dataset.add_argument("--out", help="directory to write JSONL into")
    dataset.add_argument("--show", type=int, default=0, help="print N example pairs")
    dataset.set_defaults(func=_dataset)

    train = sub.add_parser("train", help="fine-tune the coach")
    train.add_argument("--model", default="unsloth/gemma-3n-E2B-it")
    train.add_argument("--output", default="artifacts/wellness-coach")
    train.add_argument("--strategy", choices=("dpo", "sft"), default="dpo")
    train.add_argument("--fallback-to-sft", action="store_true",
                       help="fall back to SFT if DPO fails (e.g. out of memory)")
    train.add_argument("--max-steps", type=int, default=200)
    train.add_argument("--learning-rate", type=float, default=5e-6)
    train.add_argument("--lora-rank", type=int, default=16)
    train.add_argument("--limit", type=int, default=3000)
    train.add_argument("--push-to-hub", help="HuggingFace repo id to push the adapter to")
    train.set_defaults(func=_train)

    evaluate_cmd = sub.add_parser("evaluate", help="score coaching behaviour")
    evaluate_cmd.add_argument("--engine", default="auto", choices=("auto", "gemma", "echo"))
    evaluate_cmd.add_argument("--adapter", help="path to a trained LoRA adapter")
    evaluate_cmd.add_argument("--json", help="write the full report to this path")
    evaluate_cmd.set_defaults(func=_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
