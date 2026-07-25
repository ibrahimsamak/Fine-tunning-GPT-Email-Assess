# email-finetuner

Fine-tunes a GPT model so it writes emails in my newsletter style, then checks the output against emails it hasn't seen.

## How it works

1. Load the emails from `emails_data/` (one `.txt` per email, first line is the title).
2. For each email, ask `gpt-4o` to guess a short prompt (max 10 words) that would produce it. That gives me prompt -> email pairs without writing them by hand.
3. Shuffle and split into train / validation (80/20, fixed seed) and write them as JSONL in the chat format.
4. Upload both files, start the fine-tuning job on `gpt-4o-2024-08-06` and poll until it's done.
5. For each validation example, generate a new email from the same prompt and print it next to the real one.

The model id gets saved in `artifacts/run_state.json`, so I can run `train` and come back later for `evaluate` without copying ids around.

## Usage

```bash
pip install openai pytest
export OPENAI_API_KEY=sk-...
export PYTHONPATH=src

python -m email_finetuner prepare --emails-dir emails_data
python -m email_finetuner train
python -m email_finetuner evaluate
python -m email_finetuner generate "Write an email about prompt caching"
```

`run` does prepare + train + evaluate in one go, and `state` prints what's saved so far.

Other options: `--output-dir`, `--model`, `--suffix`, `-v` for debug logs. Most settings can also come from env vars (`EMAILS_DIR`, `OUTPUT_DIR`, `BASE_MODEL`, `TUNING_MODEL`, `FINETUNE_SUFFIX`).

## Files

- `core.py` - loading emails, building the dataset, fine-tuning, evaluation
- `openai.py` - all the OpenAI calls are here (chat, upload, create/retrieve job)
- `cli.py` - command line

## Tests

```bash
PYTHONPATH=src pytest
```

The tests use a fake client instead of the real OpenAI one, so they don't cost anything and don't need a key.
