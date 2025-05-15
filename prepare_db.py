import os
import chromadb
import hydra
from omegaconf import DictConfig

import prompts
from paper_loader import PaperLoader
from load_model import ModelLoader
from load_embedder import EmbeddingLoader
from paper_processor import PaperProcessor


# LlamaIndex components
from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings

class PaperSemanticVectorIndex:
    def __init__(self, config: DictConfig) -> None:
        db_dir = config.db_dir
        
        # model
        print('\nLoading Embedding Model...\n')
        embedder = EmbeddingLoader(config.embed_model)
        self.embed_model = embedder.embed_model
        
        print('\nLoading Inference LLM and Tokenizer...\n')
        self.model = ModelLoader(base_model_id=config.base_model)

        # data
        print('\nLoading Paper...\n')
        paper_path = os.path.join(config.paper_dir, config.paper)
        self.paper = PaperLoader(paper_path, self.model)

        # vector store
        print('\nLoading Vector Store...\n')
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_client.get_or_create_collection(config.collection_name))
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # process paper
        print('\nProcessing paper...\n')
        self.paper_processor = PaperProcessor(self.embed_model, self.model, self.paper)
        documents = self.paper_processor.get_semantic_documents()
        
        # create index
        print('\nCreating Index...\n')
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )
        
        # optional
        # self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=4)
        # self.query_engine = RetrieverQueryEngine(retriever=self.retriever)

        del self.model

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    return PaperSemanticVectorIndex(cfg)


if __name__=="__main__":
    index_creator = main()
    print('created index and did nothign with it!!')
