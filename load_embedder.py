from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class EmbeddingLoader:
    def __init__(self, embed_model = "BAAI/bge-large-en"):
        self.embed_model  = HuggingFaceEmbedding(model_name = embed_model)
        