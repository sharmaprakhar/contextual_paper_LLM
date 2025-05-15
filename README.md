# contextual_paper_LLM

This repository implements contextual retrieval for academic papers, with a focus on 4G/5G/LTE network security. It sets up:
- A domain-adapted LLM acting as a 5G security expert
- A vector database index for efficient semantic retrieval
- A pipeline for processing and extracting insights from relevant academic papers

---

## Environment

- Python 3.11
- CUDA-enabled GPU (required)
- Tested on Linux environments

Note: CPU-only environments are currently not supported.

---

## Installation

Create a new Conda environment and install dependencies:

```bash
conda create -n contextual_paper_retrieval python=3.11
conda activate contextual_paper_retrieval
pip install -r requirements.txt
```

## Configuration

This project uses [Hydra](https://hydra.cc/) to manage configuration parameters.

Configuration files are located in the `conf/` directory:

- `conf/index_config.yaml` — settings for indexing academic papers
- `conf/inference_config.yaml` — parameters for inference and retrieval

## Create contextual index using an extracted paper json

```bash
python run_index.py
```


## Run inference for the default query (extracting names of novel 4G/5G/LTE/ORAN attacks - assumes a contextual index was created from the target paper)

```bash
python run_inference.py
```