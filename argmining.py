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
# for folder in folders:
#    os.makedirs(os.path.join(PROJECT_ROOT, folder), exist_ok=True)

#print(f"✅ Success! Your research structure is ready at: {os.path.abspath(PROJECT_ROOT)}")


# Data Acquisition and Environment Setup
# Authenticates with Hugging Face using a secure token and downloads the ArgMining 2026 dataset (UN Resolutions) into the project's raw data directory.

import os
from huggingface_hub import login
# from google.colab import userdata  # Removed for Cluster compatibility

# Retrieve the token from environment variables (Cluster equivalent to Colab Secrets)
hf_token = os.getenv('HF_TOKEN')
login(token=hf_token)

# !pip install -q huggingface_hub  # Removed for Cluster compatibility

from huggingface_hub import snapshot_download
#import os

# Define the dataset path inside your project
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/raw')

# Download the ArgMining 2026 dataset
# This pulls the UN resolutions (English/French) and the test set
repo_id = "ZurichNLP/ArgMining-2026-UZH-Shared-Task"

print("Downloading dataset from Hugging Face...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=RAW_DATA_PATH,
    local_dir_use_symlinks=False
)

print(f"✅ Data successfully downloaded to: {RAW_DATA_PATH}")
# List the files to make sure everything is there
print("Files in data/raw:", os.listdir(RAW_DATA_PATH))

# The "Emergency Brake"
sys.exit("Stopping here: Folder structure created and Data downloaded. Next step: Processing.")