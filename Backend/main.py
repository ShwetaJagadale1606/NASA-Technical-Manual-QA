from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ask_question

app = FastAPI(title="Technical Manual QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Technical Manual QA API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask(request: QuestionRequest):
    result = ask_question(request.question)
    return result