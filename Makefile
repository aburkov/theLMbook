# Local setup for The Hundred-Page Language Models Book examples.
# Uses uv to create a Python 3.12 venv (system Python may be too new for torch).

PY_VERSION ?= 3.12
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip

.PHONY: help install install-apps kernel notebook clean

help:
	@echo "make install       - create .venv (Python $(PY_VERSION)) and install core deps"
	@echo "make install-apps  - also install side-app deps (spotify/openclaw) + chromium"
	@echo "make kernel        - register the venv as a Jupyter kernel"
	@echo "make notebook      - launch JupyterLab"
	@echo "make clean         - remove the venv and caches"

$(PYTHON):
	uv venv --seed --python $(PY_VERSION) $(VENV)

install: $(PYTHON)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@$(MAKE) kernel

install-apps: $(PYTHON)
	$(PIP) install -r requirements-apps.txt
	$(PYTHON) -m playwright install chromium

kernel:
	$(PYTHON) -m ipykernel install --user --name thelmbook --display-name "theLMbook (.venv)"

notebook:
	$(PYTHON) -m jupyterlab

clean:
	rm -rf $(VENV) .ipynb_checkpoints __pycache__ .pytest_cache .mypy_cache
