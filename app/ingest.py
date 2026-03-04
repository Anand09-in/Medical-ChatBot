import gc
import logging
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def create_vector_db():
    logging.info("Loading PDFs...")

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # Normalize text (prevents micro-fragmentation)
    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", " ").strip()

    logging.info("Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    texts = splitter.split_documents(documents)

    # Free original documents early
    del documents
    gc.collect()

    logging.info(f"Total chunks created: {len(texts)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={
            "batch_size": 32,
            "normalize_embeddings": True
        }
    )

    # ---- IMPORTANT: Preserve metadata ----
    logging.info("Generating embeddings in batches...")

    contents = [doc.page_content for doc in texts]
    metadatas = [doc.metadata for doc in texts]

    # Free texts early (reduce memory peak)
    del texts
    gc.collect()

    vectors = embeddings.embed_documents(contents)

    logging.info("Creating FAISS index...")

    db = FAISS.from_embeddings(
        list(zip(contents, vectors)),
        embeddings,
        metadatas=metadatas
    )

    db.save_local(VECTOR_DB_PATH)

    # Cleanup
    del contents
    del vectors
    gc.collect()

    logging.info("Vector DB created successfully!")
