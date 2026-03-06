import logging
import re
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.rag.llm import load_llm
from app.rag.retriever import load_retriever
from app.memory.redis_memory import get_redis_memory
from app.rag.prompts import *


logging.basicConfig(level=logging.INFO)


class RAGPipeline:

    def __init__(self):

        logging.info("Initializing RAG pipeline...")

        self.llm = load_llm()
        self.retriever = load_retriever()

    def build_chain(self, session_id: str):

        memory = get_redis_memory(session_id)

        # ----------------------------------------
        # Prompt to rewrite follow-up questions
        # ----------------------------------------

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given the chat history and a follow-up question, "
                    "rewrite the question so it is a standalone question."
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            contextualize_q_prompt
        )

        # ----------------------------------------
        # QA prompt
        # ----------------------------------------

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt,
            document_prompt=DOC_PROMPT
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        return rag_chain, memory

    def query(self, question: str, session_id: str):

        logging.info(f"Processing question for session {session_id}: {question}")
        rag_chain, memory = self.build_chain(session_id)

        chat_history = memory.chat_memory.messages
        logging.info(f"Current chat history for session {session_id}: {chat_history}")
        result = rag_chain.invoke(
            {
                "input": question,
                "chat_history": chat_history
            }
        )

        answer = result["answer"].strip()

        # remove leading punctuation or stray tokens
        answer = re.sub(r"^[\?\.\!\n\s]+", "", answer)

        # remove unwanted role tokens if they appear
        for token in ["AI:", "Assistant:", "Computer:", "Human:", "System:"]:
            answer = answer.replace(token, "")

        answer = answer.strip()

        sources = set(
            doc.metadata.get("source", "").split("\\")[-1]
            for doc in result["context"]
        )

        logging.info(f"Answer: {answer}")
        # save messages to memory
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)

        return answer, sources