from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import core
from .core import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="email-finetuner", description=__doc__)
    parser.add_argument("--emails-dir", type=Path, help="Directory of raw email files")
    parser.add_argument("--output-dir", type=Path, help="Where artifacts are written")
    parser.add_argument("--model", dest="tuning_model", help="Base model to fine-tune")
    parser.add_argument("--suffix", help="Fine-tuned model name suffix")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Load emails, generate seed prompts, write JSONL")
    sub.add_parser("train", help="Upload the JSONL files and run the fine-tuning job")

    evaluate = sub.add_parser("evaluate", help="Compare tuned output to held-out emails")
    evaluate.add_argument("--model-id", help="Override the stored fine-tuned model id")

    generate = sub.add_parser("generate", help="Generate one email from a prompt")
    generate.add_argument("prompt")
    generate.add_argument("--model-id")

    sub.add_parser("run", help="prepare + train + evaluate")
    sub.add_parser("state", help="Show the stored run state")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    try:
        settings = Settings.from_env(
            emails_dir=args.emails_dir,
            output_dir=args.output_dir,
            tuning_model=args.tuning_model,
            suffix=args.suffix,
        )

        if args.command == "prepare":
            train, validation = core.prepare(settings)
            print(f"{len(train)} train / {len(validation)} validation examples")
        elif args.command == "train":
            print(f"Fine-tuned model: {core.train(settings)}")
        elif args.command == "evaluate":
            print(core.format_report(core.evaluate(settings, args.model_id)))
        elif args.command == "generate":
            print(core.generate(settings, args.prompt, args.model_id))
        elif args.command == "run":
            print(core.format_report(core.run(settings)))
        elif args.command == "state":
            print(json.dumps(core.load_state(settings.state_file), indent=2))
        return 0

    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
