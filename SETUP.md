# Local setup

Use Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), and
Make on macOS or Linux. On Windows, these commands work in WSL.

```sh
make install
make notebook
```

Select the **theLMbook (.venv)** kernel in JupyterLab. Start with the chapter 1
[logistic regression and feedforward network notebook](linear_and_fnn_models.ipynb).
It runs on CPU with synthetic data. Two other CPU examples are available as scripts.

```sh
.venv/bin/python embedding_vs_linear.py
.venv/bin/python quadratic_loss.py
```

Some notebooks rely on Colab, Google Drive, or external datasets. Skip their
package installation cells when the local environment already has the required
packages. Replace Colab-specific paths with local paths to the corresponding
data and model files.

## GRPO

A CUDA GPU with bfloat16 support is recommended for the 0.5B notebook.
The DataParallel walkthrough requires at least two GPUs and was originally
run on eight A100s. The multi-GPU notebook setup cell pins Transformers 4.49.0 for reproduction.
The local environment and the 0.5B notebook use Transformers 4.51.3 to
support both the Transformers Trainer and TRL 0.17 in one process. Use the
multi-GPU setup cell in a separate environment when reproducing that run.
Both GRPO notebooks use PyTorch SDPA for attention.

Set `WANDB_MODE=offline` before launching JupyterLab to record runs locally,
or set `WANDB_API_KEY` in the environment to log online.

## Optional apps

```sh
make install-apps
cp .env.example .env
```

Fill in the variables for the app you want to run. For Spotify, register the
exact loopback redirect URI from `.env` in the Spotify developer dashboard.
The browser example uses `GOOGLE_API_KEY` and installs Playwright Chromium.
