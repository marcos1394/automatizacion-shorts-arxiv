#!/bin/bash
# Activar venv y ejecutar pipeline
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PATH="/Users/marcossandovalruiz/Documents/arxiv-social-automation/venv312/bin:$PATH"
cd "/Users/marcossandovalruiz/Documents/arxiv-social-automation"
source "/Users/marcossandovalruiz/Documents/arxiv-social-automation/venv312/bin/activate"
python main.py --shorts --category ai >> "/Users/marcossandovalruiz/Documents/arxiv-social-automation/logs/run_$(date +%Y%m%d_%H%M).log" 2>&1
