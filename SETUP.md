# Local Setup

Run the examples from *The Hundred-Page Language Models Book* on your own
machine. The notebooks were authored for Google Colab; this is the local path.

## Requirements

- **Python 3.10–3.12** (3.13/3.14 are too new for the pinned ML stack).
- [`uv`](https://github.com/astral-sh/uv) (already installed via Homebrew here).
- A GPU helps but is **not** required — the small examples run on CPU.

## Quick start

```bash
make install      # create .venv (Python 3.12) + install core deps + register Jupyter kernel
make notebook     # launch JupyterLab
```

In JupyterLab, open any `*.ipynb` and select the **theLMbook (.venv)** kernel.

Run a standalone script:

```bash
.venv/bin/python quadratic_loss.py
.venv/bin/python embedding_vs_linear.py
.venv/bin/python GRPO.py            # heavy: downloads a model, GPU recommended
```

## Side apps (optional)

`spotify_gemini_playlist.py` and `openclaw.py` need extra deps and API keys:

```bash
make install-apps           # adds spotipy, dotenv, google-genai, playwright + chromium
cp .env.example .env        # then fill in your keys
```

- **spotify_gemini_playlist.py** reads `SPOTIPY_*` + `OPENROUTER_API_KEY` from `.env`.
- **openclaw.py** currently *hardcodes* `GOOGLE_API_KEY` at the top of the file —
  paste your key there (or edit it to read the env var).

## Notes

- **flash-attn** is excluded from `requirements.txt`: it is CUDA-only and won't
  build on macOS/CPU. A couple of GRPO notebooks `!pip install flash-attn` — skip
  that cell, or set `attn_implementation="eager"` when loading the model.
- **wandb**: GRPO notebooks log to Weights & Biases. Run `wandb offline` (or
  `wandb disabled`) in the venv to skip the login prompt.
- The notebooks contain Colab-only cells (`from google.colab import drive`,
  `!pip install ...`). Skip those locally — deps are already in the venv.
- Without `uv`, substitute: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## What's covered

| Example | Type | Key deps |
|---|---|---|
| `quadratic_loss.py` | script | matplotlib, numpy |
| `embedding_vs_linear.py` | script | torch |
| `GRPO.py`, GRPO notebooks | script/nb | torch, transformers, datasets, trl |
| `byte_pair_encoding`, `count_language_model` | nb | stdlib |
| `news_RNN_language_model`, `news_decoder_language_model` | nb | torch, datasets |
| `emotion_*`, `instruct_GPT2`, `sampling_method` | nb | torch, transformers, peft, sklearn |
| `document_classifier_with_LLMs_as_labelers` | nb | openai, FlagEmbedding |
| `spotify_gemini_playlist.py` | app | spotipy, dotenv, requests |
| `openclaw.py` | app | google-genai, playwright |
