import os
import chromadb
import hydra
from omegaconf import DictConfig

from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from paper_loader import PaperLoader
from load_model import ModelLoader
from load_embedder import EmbeddingLoader
from paper_processor import PaperProcessor


class PaperSemanticVectorIndex:
    def __init__(self, config: DictConfig):
        self.config = config
        self.db_dir = config.db_dir

        print('\n[INDEXING 1/4] Loading Embedding Model...\n')
        self.embed_model = EmbeddingLoader(config.embed_model).embed_model

        print('\n[INDEXING 2/4] Loading Inference LLM and Tokenizer...\n') # optim point
        self.model = ModelLoader(base_model_id=config.base_model)

    def _process_paper(self, paper_name: str):
        print(f'\n[INDEXING 3/4] Processing Paper: {paper_name}\n')

        collection_name = os.path.splitext(paper_name)[0]  # remove '.json'
        chroma_client = chromadb.PersistentClient(path=self.db_dir)
        existing = set([col.name for col in chroma_client.list_collections()])
        
        if collection_name in existing:
            print(f'paper {paper_name} already exists in the chroma collection, skipping!')
            return
        
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        processor = PaperProcessor(self.embed_model, self.model, self.config.paper_dir, paper_name)
        documents = processor.get_semantic_documents()

        print('\n[INDEXING 4/4] Creating Index...\n')
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model
        )

    def run(self):
        if self.config.process_all_index:
            print('\n process_all_index set to true: will index all papers\n')
            paper_files = [f for f in os.listdir(self.config.paper_dir) if f.endswith('.json')]
            for paper_name in paper_files:
                self._process_paper(paper_name)
        else:
            print('\n process_all_index set to False: will index only: ', self.config.paper)
            self._process_paper(self.config.paper)

        # Optional cleanup
        del self.model


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    indexer = PaperSemanticVectorIndex(cfg)
    indexer.run()


if __name__ == "__main__":
    main()
