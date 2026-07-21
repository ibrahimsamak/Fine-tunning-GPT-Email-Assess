from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def make_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def chat(client, messages, *, model: str, temperature: float = 1.0) -> str:
    response = client.chat.completions.create(
        model=model, messages=list(messages), temperature=temperature
    )
    return (response.choices[0].message.content or "").strip()


def upload_file(client, path) -> str:
    with Path(path).open("rb") as handle:
        response = client.files.create(file=handle, purpose="fine-tune")
    logger.info("Uploaded %s as file %s", Path(path).name, response.id)
    return response.id


def create_job(
    client, *, training_file_id: str, validation_file_id: str, model: str, suffix: str
) -> dict:
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        suffix=suffix,
    )
    return _job(response)


def retrieve_job(client, job_id: str) -> dict:
    return _job(client.fine_tuning.jobs.retrieve(job_id))


def _job(response) -> dict:
    error = getattr(response, "error", None)
    message = getattr(error, "message", None) if error else None
    return {
        "id": response.id,
        "status": response.status,
        "fine_tuned_model": getattr(response, "fine_tuned_model", None),
        "error": message,
    }
