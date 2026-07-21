from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import openai as backend

logger = logging.getLogger(__name__)

SEED_PROMPT_SYSTEM = """\
You are an expert prompt engineer and content creator.
Analyze the email and draft a prompt up to 10 words that would result in that email's body.
Write just the content of the prompt and nothing else."""

PERSONA_SYSTEM = """\
You write newsletter emails about analytics and AI.
You are an expert at writing Engaging and Conversational emails.
The paragraphs are 1 sentence only and written with a humorous, provoking / light hearted way.
You start the emails with a thought-provoking hook."""

_TERMINAL = {"succeeded", "failed", "cancelled"}


@dataclass(frozen=True)
class Email:
    title: str
    body: str
    source: str

    @classmethod
    def from_raw(cls, raw: str, source: str) -> "Email":
        lines = raw.strip().splitlines()
        if not lines:
            raise ValueError(f"Email '{source}' is empty")
        title = lines[0].removeprefix("Title:").strip()
        body = "\n".join(lines[1:]).strip()
        return cls(title=title, body=body, source=source)

    @property
    def content(self) -> str:
        return f"Title: {self.title}, Body: {self.body}"


@dataclass(frozen=True)
class Settings:
    api_key: str
    emails_dir: Path = Path("emails_data")
    output_dir: Path = Path("artifacts")

    base_model: str = "gpt-4o"
    tuning_model: str = "gpt-4o-2024-08-06"
    suffix: str = "email-writer"

    train_ratio: float = 0.8
    seed: int | None = 42
    max_workers: int = 8
    temperature: float = 1.0

    poll_interval_seconds: float = 30.0
    poll_timeout_seconds: float = 3600.0

    seed_prompt_system: str = SEED_PROMPT_SYSTEM
    persona_system: str = PERSONA_SYSTEM

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("An OpenAI API key is required")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must be strictly between 0 and 1")

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        base = cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            emails_dir=Path(os.getenv("EMAILS_DIR", "emails_data")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "artifacts")),
            base_model=os.getenv("BASE_MODEL", "gpt-4o"),
            tuning_model=os.getenv("TUNING_MODEL", "gpt-4o-2024-08-06"),
            suffix=os.getenv("FINETUNE_SUFFIX", "email-writer"),
        )
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(base, **clean) if clean else base

    @property
    def train_file(self) -> Path:
        return self.output_dir / "train_data.jsonl"

    @property
    def validation_file(self) -> Path:
        return self.output_dir / "validation_data.jsonl"

    @property
    def state_file(self) -> Path:
        return self.output_dir / "run_state.json"


def load_emails(directory, *, pattern: str = "*", encoding: str = "utf-8") -> list[Email]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Email directory not found: {directory}")

    emails: list[Email] = []
    for path in sorted(p for p in directory.glob(pattern) if p.is_file()):
        try:
            emails.append(Email.from_raw(path.read_text(encoding=encoding), path.name))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)

    if not emails:
        raise ValueError(f"No readable emails found in {directory}")
    logger.info("Loaded %d emails from %s", len(emails), directory)
    return emails


def seed_prompts(
    client, emails, *, model: str, system_prompt: str, max_workers: int = 8
) -> list[tuple[Email, str]]:
    if not emails:
        return []

    def one(email: Email) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": email.content},
        ]
        try:
            return backend.chat(client, messages, model=model, temperature=0.7)
        except Exception as exc:
            logger.warning("Prompt generation failed for %s: %s", email.source, exc)
            return ""

    with ThreadPoolExecutor(max_workers=min(max_workers, len(emails))) as pool:
        prompts = list(pool.map(one, emails))

    paired = [(email, prompt) for email, prompt in zip(emails, prompts) if prompt]
    logger.info("Generated %d/%d seed prompts", len(paired), len(emails))
    return paired


def to_example(email: Email, prompt: str, persona_system: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": persona_system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": email.content},
        ]
    }


def build_dataset(
    prompted, *, persona_system: str, train_ratio: float = 0.8, seed: int | None = 42
) -> tuple[list[dict], list[dict]]:
    if len(prompted) < 2:
        raise ValueError("Need at least 2 examples to build a train/validation split")

    examples = [to_example(email, prompt, persona_system) for email, prompt in prompted]
    random.Random(seed).shuffle(examples)

    cutoff = max(1, int(len(examples) * train_ratio))
    cutoff = min(cutoff, len(examples) - 1)
    train, validation = examples[:cutoff], examples[cutoff:]
    logger.info("Split dataset: %d train / %d validation", len(train), len(validation))
    return train, validation


def write_jsonl(examples, path, *, encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = list(examples)
    with path.open("w", encoding=encoding) as out:
        for example in examples:
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples to %s", len(examples), path)
    return path


def read_jsonl(path, *, encoding: str = "utf-8") -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"JSONL file not found: {path}")

    examples: list[dict] = []
    with path.open(encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_no}") from exc
    return examples


def run_finetune(
    client,
    train_path,
    validation_path,
    *,
    model: str,
    suffix: str,
    poll_interval_seconds: float = 30.0,
    poll_timeout_seconds: float = 3600.0,
    sleep=time.sleep,
) -> str:
    train_id = backend.upload_file(client, train_path)
    validation_id = backend.upload_file(client, validation_path)
    job = backend.create_job(
        client,
        training_file_id=train_id,
        validation_file_id=validation_id,
        model=model,
        suffix=suffix,
    )
    logger.info("Fine-tuning job %s created (status=%s)", job["id"], job["status"])

    deadline = time.monotonic() + poll_timeout_seconds
    while True:
        job = backend.retrieve_job(client, job["id"])
        logger.info("Job %s status=%s", job["id"], job["status"])

        if job["status"] in _TERMINAL:
            if job["status"] != "succeeded" or not job.get("fine_tuned_model"):
                detail = f": {job['error']}" if job.get("error") else ""
                raise RuntimeError(f"Job {job['id']} ended with status '{job['status']}'{detail}")
            logger.info("Fine-tuned model ready: %s", job["fine_tuned_model"])
            return job["fine_tuned_model"]

        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out after {poll_timeout_seconds}s waiting on job {job['id']}")
        sleep(poll_interval_seconds)


def load_state(path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(path, **updates: Any) -> dict:
    path = Path(path)
    state = load_state(path)
    state.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def format_report(records) -> str:
    divider = "-" * 70
    blocks = [
        f"[{i}] PROMPT: {r['prompt']}\n"
        f"{divider}\nREFERENCE:\n{r['expected']}\n"
        f"{divider}\nGENERATED:\n{r['generated']}\n"
        for i, r in enumerate(records, start=1)
    ]
    return f"\n{divider}\n".join(blocks)


def _content(example: dict, role: str) -> str:
    return next(m["content"] for m in example["messages"] if m["role"] == role)


def prepare(settings: Settings, *, client=None, emails=None) -> tuple[list[dict], list[dict]]:
    emails = emails if emails is not None else load_emails(settings.emails_dir)
    client = client or backend.make_client(settings.api_key)

    prompted = seed_prompts(
        client,
        emails,
        model=settings.base_model,
        system_prompt=settings.seed_prompt_system,
        max_workers=settings.max_workers,
    )
    train, validation = build_dataset(
        prompted,
        persona_system=settings.persona_system,
        train_ratio=settings.train_ratio,
        seed=settings.seed,
    )
    write_jsonl(train, settings.train_file)
    write_jsonl(validation, settings.validation_file)
    save_state(
        settings.state_file,
        train_examples=len(train),
        validation_examples=len(validation),
    )
    return train, validation


def train(settings: Settings, *, client=None) -> str:
    client = client or backend.make_client(settings.api_key)
    model_id = run_finetune(
        client,
        settings.train_file,
        settings.validation_file,
        model=settings.tuning_model,
        suffix=settings.suffix,
        poll_interval_seconds=settings.poll_interval_seconds,
        poll_timeout_seconds=settings.poll_timeout_seconds,
    )
    save_state(settings.state_file, model_id=model_id)
    return model_id


def evaluate(settings: Settings, model_id: str | None = None, *, client=None) -> list[dict]:
    model_id = model_id or load_state(settings.state_file).get("model_id")
    if not model_id:
        raise RuntimeError("No fine-tuned model available — run `train` first")
    client = client or backend.make_client(settings.api_key)

    records = []
    for example in read_jsonl(settings.validation_file):
        prompt = _content(example, "user")
        records.append(
            {
                "prompt": prompt,
                "expected": _content(example, "assistant"),
                "generated": generate(settings, prompt, model_id, client=client),
            }
        )
    logger.info("Evaluated %d validation examples", len(records))
    return records


def generate(
    settings: Settings, user_prompt: str, model_id: str | None = None, *, client=None
) -> str:
    model_id = model_id or load_state(settings.state_file).get("model_id")
    if not model_id:
        raise RuntimeError("No fine-tuned model available — run `train` first")
    client = client or backend.make_client(settings.api_key)

    messages = [
        {"role": "system", "content": settings.persona_system},
        {"role": "user", "content": user_prompt},
    ]
    return backend.chat(client, messages, model=model_id, temperature=settings.temperature)


def run(settings: Settings, *, client=None) -> list[dict]:
    client = client or backend.make_client(settings.api_key)
    prepare(settings, client=client)
    model_id = train(settings, client=client)
    return evaluate(settings, model_id, client=client)
