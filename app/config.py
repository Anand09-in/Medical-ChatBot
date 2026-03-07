import os

DATA_PATH = "data/"
VECTOR_DB_PATH = "vectorstore/db_faiss"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#LLM_MODEL = "microsoft/Phi-3.5-mini-instruct" 
LLM_MODEL= "microsoft/phi-2"#"Qwen/Qwen3-4B" #"Mistral-7B-Instruct" 
#LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEVICE = "cuda" if os.getenv("USE_CUDA") == "1" else "cpu"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 3