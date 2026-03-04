import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .config import *

logging.basicConfig(level=logging.INFO)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class RAGPipeline:
    def __init__(self):

        logging.info("Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE}
        )

        logging.info("Loading FAISS index...")
        self.db = FAISS.load_local(
            VECTOR_DB_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        logging.info("Loading LLM...")

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            # load_in_4bit=True,
            # low_cpu_mem_usage=True
        )

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            temperature=0.3,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id
        )

        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        prompt_template = """
You are a helpful medical assistant.

Use the context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, question: str):
        logging.info(f"Received question: {question}")
        
        result = self.qa({"query": question})
        logging.info(f"Raw LLM output: {result}")
        answer = result["result"]
        sources = [
            doc.metadata.get("source", "")
            for doc in result["source_documents"]
        ]
        logging.info(f"Sources for '{question}': {sources}")
        return answer, sources