### # Re-ranker inherited from BaseNodePostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from pydantic import Field
from prompts import reranking_prompt
from utils import invoke_llm, extract_score
# from llama_index.retrievers.postprocess_retriever import RetrieverWithPostProcessors

class LocalLLMReranker(BaseNodePostprocessor):
    model: any = Field(...)
    tokenizer: any = Field(...)
    top_n: int = Field(default=5)
    # no __init_ constructor because BaseNodePostprocessor inherits is a pydantic class

    def _postprocess_nodes(self, nodes, query):
        ranked_nodes = []
        for node in nodes:
            if node.metadata['title'] == '':
                continue
            # print('\ncur node: ', node.metadata['title'])
            prompt = reranking_prompt.format(query, node.text)
            model_output = invoke_llm(prompt, self.tokenizer, self.model)
            print('scoring model output:', model_output)
            try:
                score = extract_score(model_output)
                print('found score for node: {}, score: {}\n'.format(node.metadata['title'], score))
            except:
                score = 0.0
            ranked_nodes.append(NodeWithScore(node=node.node, score=score))
        # return sorted(ranked_nodes, key=lambda x: x.score, reverse=True)[:self.top_n]
        return ranked_nodes