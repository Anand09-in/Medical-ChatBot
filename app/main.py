from fastapi import FastAPI, HTTPException
from .schemas import QueryRequest, QueryResponse
from .rag_pipeline import RAGPipeline

app = FastAPI(title="Medical RAG API")

rag = RAGPipeline()

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
def query_bot(request: QueryRequest):
    try:
        answer, sources = rag.query(request.question)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))