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
#     os.makedirs(os.path.join(PROJECT_ROOT, folder), exist_ok=True)

# print(f"✅ Success! Your research structure is ready at: {os.path.abspath(PROJECT_ROOT)}")


# Data Acquisition and Environment Setup
# Authenticates with Hugging Face using a secure token and downloads the ArgMining 2026 dataset (UN Resolutions) into the project's raw data directory.

from huggingface_hub import login
from huggingface_hub import snapshot_download

# Retrieve the token from environment variables (Cluster equivalent to Colab Secrets)
hf_token = os.getenv('HF_TOKEN')
# login(token=hf_token)

# Define the dataset path inside your project
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/raw')

# Download the ArgMining 2026 dataset
# Commented out as data is already downloaded
"""
repo_id = "ZurichNLP/ArgMining-2026-UZH-Shared-Task"
print("Downloading dataset from Hugging Face...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=RAW_DATA_PATH,
    local_dir_use_symlinks=False
)
"""

# Training Data Validation and Schema Inspection
# Verifies dataset integrity by loading a sample JSON file to confirm presence, resolve schema structure (list vs. dict), and preview key metadata and English content.

import json

# Identify training directory and all associated paragraph files
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, 'data/raw/train-data')
all_files = [f for f in os.listdir(TRAIN_DATA_DIR) if f.endswith('.json')]

if not all_files:
    print("❌ No JSON files found. Ensure data download was successful.")
else:
    # Select first available sample for structural validation
    sample_path = os.path.join(TRAIN_DATA_DIR, all_files[0])
    with open(sample_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # UN-RES dataset often packages paragraphs in single-item lists;
    # this logic ensures compatibility across different parsing versions.
    para_data = data[0] if isinstance(data, list) and len(data) > 0 else data

    print(f"✅ Success! File: {all_files[0]}")
    print(f"📌 Paragraph Type: {para_data.get('type', 'Unknown')}")
    print(f"📏 Structure Level: {para_data.get('level', 'N/A')}")

    # Isolate English translation for Semantic Entropy analysis 
    english_text = para_data.get('text_en', 'N/A')
    print("\n--- English Content (text_en) ---")
    print(f"{english_text[:300]}...")


# Quantized Model Instantiation
# Loads the Llama-3-8B-Instruct model with 4-bit Normal Float (NF4) quantization to minimize memory footprint while maintaining inference fidelity on the GPU.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# 2. 4-bit config is what makes this "Innovation Grant" material!
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 3. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # This sends it to the GPU automatically
    token=hf_token
)

print("🚀 8B Engine Ready to calculate Entropy on GPU!")


# Semantic Entropy Metric Definition
# Calculates the average token-level entropy (H) to quantify model uncertainty, serving as the primary metric for identifying and pruning high-noise text segments.

import torch.nn.functional as F
import numpy as np

def calculate_semantic_entropy(text, model, tokenizer):
    """
    Core Research Metric: Calculates average token-level entropy.
    Higher Score = Higher model uncertainty/noise.
    """
    # 1. Prepare input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # 2. Get model predictions (logits)
        outputs = model(**inputs)
        logits = outputs.logits # Shape: [1, seq_len, vocab_size]

        # 3. Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # 4. Calculate Entropy: H = -sum(p * log(p))
        # We add a tiny 1e-10 to avoid log(0) errors
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # 5. Average across all tokens in the paragraph
        mean_entropy = torch.mean(entropy).item()

    return mean_entropy

# --- TEST ---
# Use the 'english_text' from our previous data inspector cell
if 'english_text' in locals():
    score = calculate_semantic_entropy(english_text, model, tokenizer)
    print(f"📊 Success! Semantic Entropy for first paragraph: {score:.4f}")
else:
    print("❌ Run your 'Data Inspector' cell first to define english_text!")


# Pilot Batch Processing and Entropy Benchmarking
# Executes the semantic entropy calculation across a sample dataset to establish baseline metrics, saving the aggregated results to a CSV file for preliminary analysis.

import pandas as pd
from tqdm import tqdm # Provides a progress bar

# 1. Define how many files to process for this first "pilot" run
BATCH_SIZE = 50
results = []

print(f"🚀 Starting Batch Process for {BATCH_SIZE} paragraphs...")

for filename in tqdm(all_files[:BATCH_SIZE]):
    file_path = os.path.join(TRAIN_DATA_DIR, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle the list structure we discovered earlier
    para_data = data[0] if isinstance(data, list) and len(data) > 0 else data
    text_en = para_data.get('text_en', '')
    para_type = para_data.get('type', 'unknown')

    if text_en:
        # Calculate entropy using our 8B Engine
        entropy_val = calculate_semantic_entropy(text_en, model, tokenizer)

        results.append({
            'filename': filename,
            'type': para_type,
            'entropy': entropy_val,
            'text_length': len(text_en)
        })

# 2. Convert to DataFrame and save to your Project Folder
df_results = pd.DataFrame(results)
output_path = os.path.join(PROJECT_ROOT, 'data/entropy_pilot_results.csv')
df_results.to_csv(output_path, index=False)

print(f"\n✅ Batch Complete! Results saved to: {output_path}")


# Exploratory Data Analysis and Visualization
# Generates a histogram of semantic entropy distributions by paragraph type to visualize model uncertainty and calculates descriptive statistics for the grant proposal.

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for a professional research paper look
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Create the histogram
ax = sns.histplot(data=df_results, x='entropy', hue='type', kde=True, bins=15, palette='viridis')

# Add titles and labels for your grant proposal
plt.title('Distribution of Semantic Entropy in UN Paragraphs', fontsize=15)
plt.xlabel('Semantic Entropy Score (Uncertainty)', fontsize=12)
plt.ylabel('Frequency (Number of Paragraphs)', fontsize=12)

# Save the plot to your project folder for your paper
plot_path = os.path.join(PROJECT_ROOT, 'experiments/entropy_distribution.png')
plt.savefig(plot_path)

# plt.show() # Disabled for Cluster use

print(f"✅ Visualization complete! Chart saved to: {plot_path}")

# Quick Insight for your paper:
mean_by_type = df_results.groupby('type')['entropy'].mean()
print("\n📊 Average Entropy by Category:")
print(mean_by_type)


# (Zero-Shot) Paragraph Classification Logic
# Implements a strict prompt engineering strategy to categorize UN paragraphs (operative, preambular, header) by analyzing syntax markers like numbering and leading verbs.
def subtask1_classifier_v2(text, model, tokenizer):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a UN Document Auditor. Your ONLY goal is to classify the text.

RULES:
1. 'operative': MUST be a formal decision. Look for paragraphs starting with a number (e.g., "1.", "2.") OR starting directly with: Decides, Requests, Stresses, Reaffirms, Calls, Urges, Adopts, Notes, Authorizes.
2. 'preambular': MUST start with an "-ing" word (Welcoming, Recalling, etc.) or "Having examined".
3. 'header': Use for everything else: Dates, Symbols (A/RES...), Image tags, Titles, or text that doesn't start with a Number or an -ing word.

CRITICAL: Do not assume words exist if they are not in the text. Look at the VERY FIRST WORD.

Return JSON ONLY.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Paragraph: "{text}"

Target JSON:
{{
  "type": "header, preambular, or operative",
  "think": "The first word is [WORD], which triggers the [RULE] rule."
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_new_tokens=100, temperature=0.01)

    return tokenizer.decode(output_tokens[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

# (Zero-Shot) Classification Inference
# Executes the v2 classifier on a sample paragraph to validate the model's ability to distinguish between header, preambular, and operative text using the new rule-based logic.
# (Note: Assumes 'english_text' was defined during your earlier session run)
if 'english_text' in locals():
    prediction_result = subtask1_classifier_v2(english_text, model, tokenizer)
    print("🤖 Subtask 1 (v2) Analysis Result:")
    print(prediction_result)

# Argumentative Relation Extraction Logic
# Analyzes the semantic link between a current paragraph and its predecessor to determine if the text "Supports" the reasoning or is "Independent," a key requirement for reconstructing the argumentative graph.
def predict_argumentative_relation(text, prev_text, model, tokenizer):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Analyze the relationship between two UN paragraphs.
PREVIOUS: {prev_text[:300]}
CURRENT: {text[:300]}

Rules:
1. 'supports': The current paragraph expands on, justifies, or provides detail for the previous one.
2. 'independent': The current paragraph starts a new topic, a new list item, or is unrelated.

Return JSON only: {{"relation": "supports/independent"}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_new_tokens=50, temperature=0.01)
    return tokenizer.decode(output_tokens[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

# End-to-End Resolution Processing Pipeline
# Iterates through document paragraphs to execute v2 classification and relation extraction, aggregating structural metadata and argumentative links into the final submission-ready JSON format.
def process_full_resolution_v2(file_path, model, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Adapt to structure found in cluster validation cell
    raw_paras = data if isinstance(data, list) else [data]
    
    processed_paras = []
    prev_text = "None (Start of Document)"
    
    for i, p in enumerate(tqdm(raw_paras[:20], desc="Processing Resolution")): # Limit to 20 for test
        text = p.get('text_en', '')
        if not text: continue
        
        # 1. Classify
        class_raw = subtask1_classifier_v2(text, model, tokenizer)
        
        # 2. Relation
        rel_raw = predict_argumentative_relation(text, prev_text, model, tokenizer)
        
        processed_paras.append({
            "index": i,
            "classification": class_raw,
            "relation": rel_raw
        })
        prev_text = text
        
    return processed_paras

# UNIVERSAL RESULTS CHECKER
# Final validation step to ensure the AI output folder contains valid JSON files and that the classification logic hasn't introduced empty or corrupt entries.
def universal_checker(directory):
    print(f"🔍 Checking results in: {directory}")
    if not os.path.exists(directory):
        print("❌ Directory not found.")
        return
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    print(f"📈 Found {len(files)} processed resolutions.")
    # Add your specific integrity checks here (Check 1, Check 2 as per Colab)

# TEST THE PIPELINE
if all_files:
    test_file = os.path.join(TRAIN_DATA_DIR, all_files[0])
    results = process_full_resolution_v2(test_file, model, tokenizer)
    print(f"✅ Full processing complete for {all_files[0]}")

# Multi-Task Argument Mining Inference Engine
# Orchestrates joint classification, thematic tagging, and relation extraction within a single structured prompt to produce competition-compliant JSON outputs.

import json
import torch

def competition_final_processor(text, prev_text, prev_idx, model, tokenizer):
    # Contextual check for the first paragraph
    context_info = f"PREVIOUS PARA [Index {prev_idx}]: \"{prev_text}\"" if prev_idx is not None else "This is the start of the document."

    # We feed the 'Dimensions' from your CSV directly into the prompt to guide tagging
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a UNESCO Legal-Political Analyst. Your task is to perform Argument Mining on an Education Conference document.

OUTPUT REQUIREMENTS:
1. TYPE: Classify as 'preambular' or 'operative'.
2. TAGS: Assign 1-3 labels from the Education Dimensions (e.g., Teachers, Curriculum, Education level, Policy theme).
3. RELATIONS: Link to Index {prev_idx if prev_idx is not None else 'N/A'}.
    Types: "supporting", "contradictive", "complemental", "modifying".
4. THINK: Provide a brief reasoning string for your choices.

Return JSON ONLY.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{context_info}
CURRENT PARA: "{text}"

Target JSON:
{{
  "type": "preambular/operative",
  "tags": ["Tag1", "Tag2"],
  "matched_paras": {{"{prev_idx if prev_idx is not None else 'null'}": "relation_type"}},
  "think": "Your step-by-step reasoning here."
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # Keep temperature low for high structural accuracy
        output_tokens = model.generate(**inputs, max_new_tokens=250, temperature=0.01)

    return tokenizer.decode(output_tokens[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)


# Production Inference Loop and Leaderboard Serialization
# Iterates through the UNESCO test dataset to perform high-resolution argument mining, updating resolution metadata and saving final competition-ready JSON files.

import os
import json
import time

# --- CONFIGURATION ---
# Adapted for Science Cluster Project Structure
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw/test-data/")
FINAL_SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submissions/leaderboard_submission/")
os.makedirs(FINAL_SUBMISSION_DIR, exist_ok=True)

test_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.json')]

print(f"🚀 PRO RUN (Take 2): Processing {len(test_files)} UNESCO Test Files...")

for f_idx, file_name in enumerate(test_files):
    # --- So that re-running completed test files doesn't happen ---
    if os.path.exists(os.path.join(FINAL_SUBMISSION_DIR, file_name)):
        continue 
    # ---------------------------------
    file_path = os.path.join(TEST_DATA_DIR, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)

    # TARGETING THE CORRECT KEY: 'paragraphs'
    paras = data.get('body', {}).get('paragraphs', [])

    if not paras:
        print(f"❌ ERROR: No paragraphs found in {file_name}. Still looking for wrong key?")
        continue

    print(f"📈 [{f_idx+1}/{len(test_files)}] Analyzing {len(paras)} paras in: {file_name}")

    op_indices = []
    pre_indices = []
    prev_text = None
    prev_idx = None

    for i, p in enumerate(paras):
        # UNESCO Test Set uses 'para_en'
        current_text = p.get('para_en', "")

        # Call our specialized UNESCO engine (Ensure this function is defined!)
        raw_response = competition_final_processor(current_text, prev_text, prev_idx, model, tokenizer)

        try:
            clean_json = raw_response.strip().replace('```json', '').replace('```', '')
            res = json.loads(clean_json)

            p['type'] = res.get('type', 'header')
            p['tags'] = res.get('tags', [])
            p['matched_paras'] = res.get('matched_paras', {})
            p['think'] = res.get('think', "")

            if p['type'] == 'operative': op_indices.append(i)
            if p['type'] == 'preambular': pre_indices.append(i)

        except Exception as e:
            p['type'] = 'header'

        prev_text = current_text
        prev_idx = i

    # Sync the METADATA structure for UZH requirements
    data['METADATA']['structure']['preambular_para'] = pre_indices
    data['METADATA']['structure']['operative_para'] = op_indices
    data['METADATA']['structure']['think'] = "Llama-3.1-8B: UNESCO Education Specialized Run."

    # Save to the new clean folder
    with open(os.path.join(FINAL_SUBMISSION_DIR, file_name), 'w') as out_f:
        json.dump(data, out_f, indent=2)

print(f"🏁 DONE! Real processing complete.")

print("🚀 Pipeline migration to Science Cluster successful.")