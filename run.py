import os
import json
import torch
import warnings

from omegaconf import OmegaConf
from hydra import initialize, compose

from prepare_db import PaperSemanticVectorIndex
from load_model import ModelLoader
from utils import load_index, invoke_llm
from reranker import LocalLLMReranker
from prompts import inference_prompt_template as prompt_template
from prompts import retrieval_query as query
from pathlib import Path

warnings.filterwarnings("ignore")

def setup_model_and_tokenizer(base_model_id):
    model_artifacts = ModelLoader(base_model_id=base_model_id)
    return model_artifacts.llm, model_artifacts.tokenizer


def setup_retriever(index, top_k):
    return index.as_retriever(similarity_top_k=top_k, retriever_mode="mmr")


def filter_nodes(nodes, min_score=7):
    return [node for node in nodes if node.score and node.score >= min_score]


def extract_target_sections(nodes):
    return {node.metadata['title'] for node in nodes}


def process_paper(json_path, target_sections, tokenizer, model):
    with open(json_path, 'r') as fh:
        data = json.load(fh)

    content = data['body']
    outputs = []

    model.eval()
    with torch.no_grad():
        for doc in content:
            title = doc['title']
            if title in target_sections:
                prompt = prompt_template + ''.join(doc['content'])
                output = invoke_llm(prompt, tokenizer, model)
                outputs.append({"title": title, "response": output})

    return outputs

def paper_inference(model, tokenizer, paper_name, cfg):
    print("Step 3: Loading vector index...\n")
    index = load_index(paper_name)
    retriever = setup_retriever(index, top_k=cfg.retriever.top_k)

    print("Step 4: Retrieving relevant nodes...\n")
    initial_nodes = retriever.retrieve(query)

    print("Initial retrieved nodes:")
    for node in initial_nodes:
        print(node.metadata['title'], f"-- {node.score:.4f}")

    reranker = LocalLLMReranker(tokenizer=tokenizer, model=model, top_n=cfg.reranker.top_n)
    reranked_nodes = reranker.postprocess_nodes(initial_nodes)
    filtered_nodes = filter_nodes(reranked_nodes, min_score=cfg.reranker.min_score)

    print("\nFinal nodes selected for inference:")
    for node in filtered_nodes:
        print(node.metadata['title'], f"-- {node.score:.4f}")

    target_sections = extract_target_sections(filtered_nodes)

    print("\nStep 5: Running inference on each paper...\n")
    os.makedirs("outputs", exist_ok=True)

    json_path = os.path.join(cfg.paper_dir, paper_name + ".json")
    print(f"\n--- Processing paper: {paper_name} ---")

    outputs = process_paper(json_path, target_sections, tokenizer, model)

    output_path = os.path.join("outputs", f"{paper_name}_results.json")
    with open(output_path, "w") as fout:
        json.dump(outputs, fout, indent=2)

    print(f"Results saved to: {output_path}")

def main():
    with initialize(config_path="conf"):
        cfg = compose(config_name="config")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

    if cfg.run_indexing:
        print("Step 1: Indexing all papers...\n")
        db = PaperSemanticVectorIndex(cfg)
        db.run()
    else:
        print("Skipping indexing as run_indexing is set to False.\n\n")

    # # Inference 
    if not cfg.run_inference:
        print("run_inference set to False. Skipping inference")
        return
    
    print("Step 2: Loading model and tokenizer...\n")
    model, tokenizer = setup_model_and_tokenizer(cfg.base_model)
    
    if cfg.process_all_inference:
        paper_files = Path(cfg.paper_dir).glob("*.json")
        for path in paper_files:
            paper_name = path.stem  # filename without `.json`
            paper_inference(model, tokenizer, paper_name, cfg)
    else:
        paper_name = Path(cfg.paper).stem
        paper_inference(model, tokenizer, paper_name, cfg)

    del model
    del tokenizer


if __name__ == "__main__":
    main()
