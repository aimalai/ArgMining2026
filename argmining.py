# -*- coding: utf-8 -*-
"""
# ArgMining 2026: Reasoning Reconstruction in UN Resolutions
Core Research: Semantic Entropy Pruning for Context Optimization

Shared Task Consraint Constraint: Open-Source Models ≤ 8B Parameters
"""

# Project Environment & Directory Configuration
# Defines the root file path and modular folder structure used to organize raw data, source code, experiments, and submission outputs.
import os
import sys

# relative path so it works on any machine/cluster
PROJECT_ROOT = './ArgMining_2026_Project'

# The modular structure
folders = [
    'data/raw',          # For the UN Resolution JSONs
    'src/pruners',       # For your Semantic Entropy logic
    'src/evaluators',    # For tracking latency/cost for the grant
    'experiments',       # For your ML4NLP2 plots and logs
    'submissions'        # For final ArgMining zipped file
]

# Comment out after initial execution
for folder in folders:
    os.makedirs(os.path.join(PROJECT_ROOT, folder), exist_ok=True)

print(f"✅ Success! Your research structure is ready at: {os.path.abspath(PROJECT_ROOT)}")

# CHANGE 3: The "Emergency Brake"
# This stops the script here so it doesn't try to run Colab code below.
sys.exit("Stopping here: Folder structure created. Next step: Data Download.")
