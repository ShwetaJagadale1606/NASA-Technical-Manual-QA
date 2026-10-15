
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "storage" / "chroma"
COLLECTION_NAME = "nasa_handbook"


# 1. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Load ChromaDB
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_PATH),
    embedding_function=embeddings,
)


# 3. Build BM25 index ONCE when the application starts
all_data = vectorstore.get(
    include=["documents", "metadatas"]
)

corpus = []

for text, metadata in zip(
    all_data["documents"],
    all_data["metadatas"],
):
    if text:
        corpus.append(
            Document(
                page_content=text,
                metadata=metadata or {},
            )
        )

bm25 = None

if corpus:
    bm25 = BM25Retriever.from_documents(corpus)
    bm25.k = 8

print(f"BM25 index ready: {len(corpus)} documents")


# 4. Configure LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# 5. Define prompt
prompt = ChatPromptTemplate.from_template("""
You are a technical manual QA assistant.

Answer using only the supplied handbook context.

Rules:
- Answer the exact question asked.
- Explain relevant references, captions, and technical terms.
- Use only information supported by the context.
- Do not invent facts or unsupported details.
- If information is incomplete, clearly state what is known.
- If the information is unavailable, say:
  "I could not find this information in the handbook."
- Keep the answer concise and include relevant PDF page numbers.

Context:
{context}

Question:
{question}

Answer:
""")


# 6. In-memory answer cache
answer_cache = {}


# 7. Answer the user's question
def ask_question(question):

    # A. Normalize the question for cache lookup
    cache_key = " ".join(question.lower().split())

    # B. Return saved answer if available
    if cache_key in answer_cache:
        print("CACHE HIT - Returning saved answer")
        return answer_cache[cache_key]

    print("CACHE MISS - Running RAG")

    # C. Semantic search
    vector_docs = vectorstore.similarity_search(
        question,
        k=8,
    )

    # D. Keyword search
    if bm25 is not None:
        keyword_docs = bm25.invoke(question)
    else:
        keyword_docs = []

    # E. Combine results and remove duplicates
    combined_docs = []
    seen = set()

    for doc in vector_docs + keyword_docs:
        key = (
            doc.page_content,
            doc.metadata.get("page"),
        )

        if key not in seen:
            seen.add(key)
            combined_docs.append(doc)

    # Limit the amount of context sent to the LLM
    combined_docs = combined_docs[:12]

    # F. Build context
    context = "\n\n".join(
        f"PDF page {doc.metadata.get('page', 'Unknown')}:\n"
        f"{doc.page_content}"
        for doc in combined_docs
    )

    # G. Generate answer
    response = (prompt | llm).invoke({
        "context": context,
        "question": question,
    })

    # H. Collect unique sources
    sources = []
    seen_sources = set()

    for doc in combined_docs:
        page = doc.metadata.get("page", "Unknown")
        source_name = doc.metadata.get(
            "source", "NASA Handbook"
        )

        key = (source_name, page)

        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "page": page,
                "source": source_name,
            })

    # I. Save successful result in cache
    result = {
        "answer": response.content,
        "sources": sources,
    }

    answer_cache[cache_key] = result

    return result


# 8. Test from terminal
if __name__ == "__main__":
    while True:
        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if question.strip().lower() == "exit":
            break

        result = ask_question(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for source in result["sources"]:
            print(source)
