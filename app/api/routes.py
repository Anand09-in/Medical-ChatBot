from fastapi import APIRouter
from pydantic import BaseModel
import logging

from app.rag.rag_pipeline import RAGPipeline


router = APIRouter()

rag = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    session_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):

    logging.info(f"Received query: {request.question}")

    answer, sources = rag.query(
        question=request.question,
        session_id=request.session_id
    )

    return {
        "answer": answer,
        "sources": sources
    }