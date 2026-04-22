PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PY := $(VENV_DIR)/bin/python

MODEL_PATH ?= deploy/robot_brain_int8_stateful.pt
MANIFEST_PATH ?= deploy/deployment_manifest.json
SMOKE_STEPS ?= 32
SMOKE_SEED ?= 2026
SMOKE_MAX_REPLAY_ERROR ?= 1e-6
SMOKE_BOOTSTRAP ?= 1

.PHONY: help env install install-lock test smoke clean

help:
	@echo "Available targets:"
	@echo "  env         Create .venv and install locked deps + editable package"
	@echo "  install     Install package in current environment"
	@echo "  install-lock Install locked dependencies in current environment"
	@echo "  test        Run unit/regression/export/integration test discovery"
	@echo "  smoke       Run deployment smoke test (auto-bootstrap by default)"
	@echo "  clean       Remove Python caches"

env:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.lock
	$(VENV_PY) -m pip install -e .

install:
	$(PYTHON) -m pip install -e .

install-lock:
	$(PYTHON) -m pip install -r requirements.lock

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

smoke:
	@if [ "$(SMOKE_BOOTSTRAP)" = "1" ] && { [ ! -f "$(MODEL_PATH)" ] || [ ! -f "$(MANIFEST_PATH)" ]; }; then \
		echo "Smoke artifacts missing, bootstrapping deterministic bundle..."; \
		$(PYTHON) scripts/bootstrap_smoke_bundle.py --output-dir $$(dirname "$(MODEL_PATH)"); \
	fi
	@test -f "$(MODEL_PATH)" || (echo "ERROR: MODEL_PATH not found: $(MODEL_PATH)" && exit 2)
	@test -f "$(MANIFEST_PATH)" || (echo "ERROR: MANIFEST_PATH not found: $(MANIFEST_PATH)" && exit 2)
	$(PYTHON) scripts/deployment_smoke_test.py \
		--model-path $(MODEL_PATH) \
		--manifest-path $(MANIFEST_PATH) \
		--steps $(SMOKE_STEPS) \
		--seed $(SMOKE_SEED) \
		--max-replay-error $(SMOKE_MAX_REPLAY_ERROR)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
