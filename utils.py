import warnings
warnings.filterwarnings('ignore')

import re 
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



def invoke_llm(prompt, tokenizer, model):
    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_tokens = model.generate(
                                        **model_input,
                                        max_new_tokens=128,
                                        repetition_penalty=1.15,
                                        pad_token_id=tokenizer.eos_token_id
                                    )[0]
    # Decode only the newly generated tokens, excluding the input prompt
    output = tokenizer.decode(generated_tokens[model_input['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return output

def extract_score(text):
    """
    Extracts the numerical relevance score enclosed in <score>...</score> tags.

    Args:
        text (str): Output string from the LLM.

    Returns:
        int or None: The extracted score as an integer if found, otherwise None.
    """
    # match = re.search(r"core>\s*(\d)\s*</score>", text)
    match = re.search(r"ore>\s*(-?\d+(?:\.\d+)?)\s*</score>", text)
    if match:
        return int(match.group(1))
    return None

def load_index(colname):
    db = chromadb.PersistentClient(path="./contextual_chroma/")
    print("Collections present in chroma :", db.list_collections())
    colname = "lte_inspector"
    chroma_collection = db.get_collection(colname)
    print("Number of documents in {} collection: {}".format(colname, len(chroma_collection.get()['ids'])))
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model  = HuggingFaceEmbedding(model_name = "BAAI/bge-large-en")

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
    return index