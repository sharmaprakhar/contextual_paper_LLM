import warnings
warnings.filterwarnings('ignore')

import os
import torch
import json
import hydra
from omegaconf import DictConfig

from load_model import ModelLoader
from utils import load_index, invoke_llm
from reranker import LocalLLMReranker
from prompts import inference_prompt_template as prompt_template
from prompts import retrieval_query as query

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
    print('processing paper')
    with open(json_path, 'r') as fh:
        data = json.load(fh)

    content = data['body']
    outputs = []

    model.eval()
    with torch.no_grad():
        for doc in content:
            title = doc['title']
            print('\n current title:', title)
            if title in target_sections:
                prompt = prompt_template + ''.join(doc['content'])
                output = invoke_llm(prompt, tokenizer, model)
                outputs.append([title, output])
                print(output)
                print('---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----')
    return outputs

@hydra.main(config_path="conf", config_name="inference_config", version_base=None)
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(cfg.base_model_id)

    # Load index and retriever
    index = load_index(cfg.colname)
    retriever = setup_retriever(index, top_k=cfg.retriever.top_k)

    # Initial retrieval
    res_init = retriever.retrieve(query)
    print("Initial retrieved nodes:")
    for node in res_init:
        print(node.metadata['title'], f"-- {node.score:.4f}")

    # Rerank
    reranker = LocalLLMReranker(tokenizer=tokenizer, model=model, top_n=cfg.reranker.top_n)
    res_reranked = reranker.postprocess_nodes(res_init)
    filtered_res = filter_nodes(res_reranked, min_score=cfg.reranker.min_score)

    print('Final nodes containing most target information:', len(filtered_res))
    for node in filtered_res:
        print(node.metadata['title'], f"-- {node.score:.4f}")
    
    # this is where additional filtering strategies could be added
    target_sections = extract_target_sections(filtered_res)
    
    # Process paper
    json_path = os.path.join(cfg.paper_dir, cfg.paper_file)
    
    print('target sections:', target_sections)
    print('json_path:', json_path)
    outputs = process_paper(json_path, target_sections, tokenizer, model)
    
    print('outputs:', outputs)

    del model
    del tokenizer

if __name__ == "__main__":
    main()