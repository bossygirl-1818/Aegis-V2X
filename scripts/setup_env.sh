#!/usr/bin/env bash
# Aegis-V2X — one-shot Phase 1 environment setup
set -e

echo "Creating virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing Phase 1 dependencies (base + dev tooling)..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Verifying environment..."
python scripts/verify_env.py

echo ""
echo "Phase 1 environment ready. Activate with: source .venv/bin/activate"
echo "Install phase-specific stacks later with, e.g.:"
echo "  pip install -r requirements/simulation.txt   # Phase 2"
echo "  pip install -r requirements/backend.txt      # Phase 3"
echo "  pip install -r requirements/ai.txt            # Phase 4-5"
echo "  pip install -r requirements/deployment.txt    # Phase 6"
