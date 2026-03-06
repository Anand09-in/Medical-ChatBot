import logging

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import *
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def load_retriever():

    logging.info("Loading embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE}
    )

    logging.info("Loading FAISS index...")

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(
        search_kwargs={"k": TOP_K}
    )
    logging.info("Retriever loaded successfully.")
    logging.info(f"Retriever details: {retriever}")
    return retriever