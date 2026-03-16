# ArgMining 2026: Reasoning Reconstruction in UN Resolutions

### **Team Name:** Ockham

### **Shared Task:** UZH ArgMining Workshop 2026 (ACL 2026)

This repository contains the official system implementation for the UZH 2026 Shared Task on reconstructing argumentative reasoning in United Nations and UNESCO resolutions.

Our approach focuses on **Semantic Entropy Pruning**, an optimization strategy designed to maintain high-fidelity reasoning on edge-tier open-weight hardware while adhering to the shared task's constraint of models ≤ 8B parameters.

## 🚀 System Architecture: Window-of-3 Fan Reconstruction

To meet the shared task's requirements for predicting multi-link argumentative relations, our system implements a **Sliding Context Window (n=3)**. Unlike standard pairwise chains, this architecture allows the model to identify "Fan" structures where a single operative paragraph supports or modifies multiple preceding preambular clauses.

### Key Features:

- **Quantized Inference:** 4-bit NF4 quantization using `bitsandbytes` to satisfy computational sustainability goals.
- **Schema Enforcement:** Strict JSON output validation ensuring integer-based indices and valid relation types: `supporting`, `contradictive`, `complemental`, and `modifying`.
- **Semantic Entropy Pruning:** Contextual noise reduction by limiting input history to logically proximate neighbors to maximize "Thinking" mode coherence.

## 📦 Installation & Environment

The system is designed to run on a High-Performance Computing (HPC) Science Cluster with CUDA-enabled GPUs (minimum 12GB VRAM recommended).

```bash
# Clone the repository
git clone https://github.com/aimalai/ArgMining2026
cd ArgMining2026

# Install required libraries
pip install torch transformers bitsandbytes accelerate tqdm pandas scipy

```

## ⚡ Execution Pipeline

The processing is split into two stages: Inference and Sanitization.

### 1. Production Inference

Run the main processing script to generate predictions for the 89 UNESCO test resolutions. The script utilizes Llama-3.1-8B-Instruct with a sliding window buffer of 3 previous paragraphs.

```bash
python argmining.py

```

### 2. Post-Processing & Sanitization

After production run, run the experiments and sanitization suite to align the JSON schema with official UZH requirements (e.g., key remapping from `paragraphs` to `paras` and field re-ordering).

```bash
python post_process.py

```

## 📊 Evaluation Schema

Our outputs conform strictly to the UZH Shared Task Argmining 2026 JSON specification:


```json
{
  "TEXT_ID": "ICPE-25-1962_RES1-FR_res_54",
  "RECOMMENDATION": 54,
  "TITLE": "LA PLANIFICATION DE L'ÉDUCATION",
  "METADATA": {
    "structure": {
      "doc_title": "ICPE-25-1962_RES1-FR",
      "nb_paras": 58,
      "preambular_para": [], 
      "operative_para": [],   
      "think": ""
    }
  },
  "body": {
    "paras": [
      {
        "para_number": 1,
        "para": "La Conférence internationale de l'instruction publique, Convoquée à...",
        "type": null,
        "tags": [],
        "matched_paras": {},
        "think": "",
        "para_en": "The International Conference on Education, convened in..."
      }
    ]
  }
}
```

METADATA.structure:

-preambular_paras: list of paragraph indices (int) classified as preambular
-operative_paras: list of paragraph indices (int) classified as operative
-think: string describing the reasoning process (e.g., LLM thinking output)

paras:

-type: "preambular" or "operative"
-tags: list of tag labels (strings), one than more tags from different dimensions and categories are possible.
-matched_paras: dictionary of paragraph indices (int) linked by content or reference as keys, and relation types ("contradictive", "supporting", "complemental", "modifying") as values
-think: string describing the reasoning process (e.g., LLM thinking output)

## ⚖️ License

This code is provided under the MIT License. Training data follows the restricted UN-RES license and is not redistributed in this repository.
