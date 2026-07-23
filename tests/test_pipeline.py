from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from email_finetuner import (
    Email,
    Settings,
    build_dataset,
    evaluate,
    prepare,
    read_jsonl,
    run,
    to_example,
    write_jsonl,
)
from email_finetuner import core


class FakeClient:
    def __init__(self) -> None:
        self.uploads: list[str] = []
        self._poll = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._chat_create)
        )
        self.files = SimpleNamespace(create=self._file_create)
        self.fine_tuning = SimpleNamespace(
            jobs=SimpleNamespace(create=self._job_create, retrieve=self._job_retrieve)
        )

    def _chat_create(self, *, model, messages, temperature=1.0):
        user = next(m["content"] for m in messages if m["role"] == "user")
        message = SimpleNamespace(content=f"reply to: {user[:40]}")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def _file_create(self, *, file, purpose):
        self.uploads.append(getattr(file, "name", "file"))
        return SimpleNamespace(id=f"file-{len(self.uploads)}")

    def _job_create(self, *, training_file, validation_file, model, suffix):
        return SimpleNamespace(id="ft-1", status="queued", fine_tuned_model=None, error=None)

    def _job_retrieve(self, job_id):
        self._poll += 1
        if self._poll < 2:
            return SimpleNamespace(id=job_id, status="running", fine_tuned_model=None, error=None)
        return SimpleNamespace(
            id=job_id, status="succeeded", fine_tuned_model="ft:model-1", error=None
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        emails_dir=tmp_path / "emails",
        output_dir=tmp_path / "artifacts",
        poll_interval_seconds=0.0,
    )


@pytest.fixture
def emails() -> list[Email]:
    return [
        Email.from_raw(f"Title: Subject {i}\nBody line for email {i}.", source=f"{i}.txt")
        for i in range(10)
    ]


def _write_emails(directory: Path, emails: list[Email]) -> None:
    directory.mkdir(parents=True)
    for i, email in enumerate(emails):
        (directory / f"{i}.txt").write_text(f"Title: {email.title}\n{email.body}")


def test_email_parsing_splits_title_and_body():
    email = Email.from_raw("Title: Hello\n\nFirst line.\nSecond line.", source="a.txt")
    assert email.title == "Hello"
    assert email.body == "First line.\nSecond line."
    assert email.content.startswith("Title: Hello, Body:")


def test_split_is_deterministic_for_a_fixed_seed(emails):
    prompted = [(e, f"prompt {i}") for i, e in enumerate(emails)]
    first, _ = build_dataset(prompted, persona_system="p", seed=7)
    second, _ = build_dataset(prompted, persona_system="p", seed=7)
    assert [core._content(e, "user") for e in first] == [core._content(e, "user") for e in second]


def test_split_always_keeps_a_validation_example(emails):
    prompted = [(e, "p") for e in emails[:2]]
    _, validation = build_dataset(prompted, persona_system="p", train_ratio=0.99)
    assert len(validation) == 1


def test_build_rejects_a_corpus_too_small_to_split(emails):
    with pytest.raises(ValueError):
        build_dataset([(emails[0], "p")], persona_system="p")


def test_jsonl_roundtrip(tmp_path, emails):
    example = to_example(emails[0], "write something", "persona")
    path = write_jsonl([example], tmp_path / "out.jsonl")
    assert read_jsonl(path) == [example]


def test_full_pipeline_runs_against_fakes(settings, emails):
    _write_emails(settings.emails_dir, emails)
    client = FakeClient()

    records = run(settings, client=client)

    assert settings.train_file.exists() and settings.validation_file.exists()
    assert len(client.uploads) == 2
    assert core.load_state(settings.state_file)["model_id"] == "ft:model-1"
    assert records and all(r["generated"].startswith("reply to:") for r in records)


def test_evaluate_without_a_trained_model_fails(settings, emails):
    _write_emails(settings.emails_dir, emails)
    client = FakeClient()
    prepare(settings, client=client)
    with pytest.raises(RuntimeError):
        evaluate(settings, client=client)
