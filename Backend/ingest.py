
import os
import base64
import shutil
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from groq import Groq

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "Data" / "nasa_handbook.pdf"
DB_PATH = BASE_DIR / "storage" / "chroma"
COLLECTION_NAME = "nasa_handbook"

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Add it to Backend/.env"
    )

client = Groq(api_key=GROQ_API_KEY)

# Speed settings
VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

MAX_VISUAL_PAGES = 8
IMAGE_SCALE = 0.7
MAX_VISION_TOKENS = 350


# --------------------------------------------------
# 2. EXTRACT TEXT FROM PDF
# --------------------------------------------------

def load_pdf():
    documents = []

    with pymupdf.open(str(PDF_PATH)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": PDF_PATH.name,
                            "page": page_number,
                            "content_type": "text",
                        },
                    )
                )

    print(f"Extracted text from {len(documents)} pages.")
    return documents


# --------------------------------------------------
# 3. SPLIT TEXT INTO CHUNKS
# --------------------------------------------------

def create_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks.")
    return chunks


# --------------------------------------------------
# 4. EXTRACT TABLES
# --------------------------------------------------

def extract_tables():
    table_documents = []

    with pymupdf.open(str(PDF_PATH)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            try:
                tables = page.find_tables().tables
            except Exception as error:
                print(
                    f"Table extraction failed on page "
                    f"{page_number}: {error}"
                )
                continue

            for table_number, table in enumerate(tables, start=1):
                rows = table.extract()

                if not rows:
                    continue

                table_lines = []

                for row in rows:
                    cells = [
                        str(cell or "").replace("\n", " ").strip()
                        for cell in row
                    ]
                    table_lines.append(" | ".join(cells))

                table_text = "\n".join(table_lines)

                if not table_text.strip():
                    continue

                table_documents.append(
                    Document(
                        page_content=(
                            f"Table {table_number} from PDF page "
                            f"{page_number}:\n{table_text}"
                        ),
                        metadata={
                            "source": PDF_PATH.name,
                            "page": page_number,
                            "content_type": "table",
                            "table_number": table_number,
                        },
                    )
                )

    print(f"Extracted {len(table_documents)} tables.")
    return table_documents


# --------------------------------------------------
# 5. ANALYZE DIAGRAMS WITH A VISION MODEL
# --------------------------------------------------

def analyze_visual_page(page, page_number):
    """Render and describe a page likely to contain a diagram."""

    images = page.get_images(full=True)
    drawings = page.get_drawings()

    # Skip pages unlikely to contain meaningful diagrams
    if not images and len(drawings) < 20:
        return None

    print(f"Analyzing visual content on page {page_number}...")

    # Lower resolution reduces image size and processing time
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(IMAGE_SCALE, IMAGE_SCALE),
        alpha=False,
    )

    image_bytes = pixmap.tobytes("jpeg")
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_base64}"

    prompt = """
You are analyzing an engineering technical manual.

Describe the meaningful visual information on this page:
- Diagrams, block diagrams, and flowcharts
- Component names and readable labels
- Arrows, connections, and direction of flow
- Relationships between components
- Figure captions and important process steps

Be concise and accurate. Do not invent unreadable labels
or relationships. If no meaningful diagram is present,
say so briefly.
"""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_completion_tokens=MAX_VISION_TOKENS,
        )

        description = (
            response.choices[0].message.content or ""
        ).strip()

        if not description:
            return None

        return Document(
            page_content=(
                f"Diagram information from PDF page "
                f"{page_number}:\n{description}"
            ),
            metadata={
                "source": PDF_PATH.name,
                "page": page_number,
                "content_type": "visual_description",
            },
        )

    except Exception as error:
        print(
            f"Vision analysis failed on page "
            f"{page_number}: {error}"
        )
        return None


# --------------------------------------------------
# 6. PROCESS A LIMITED NUMBER OF VISUAL PAGES
# --------------------------------------------------

def extract_visual_content():
    visual_documents = []

    with pymupdf.open(str(PDF_PATH)) as pdf:
        total_pages = len(pdf)

        for page_number, page in enumerate(pdf, start=1):
            if len(visual_documents) >= MAX_VISUAL_PAGES:
                print(
                    f"Reached visual-analysis limit "
                    f"of {MAX_VISUAL_PAGES} pages."
                )
                break

            print(f"Checking page {page_number}/{total_pages}")

            visual_document = analyze_visual_page(
                page,
                page_number,
            )

            if visual_document is not None:
                visual_documents.append(visual_document)

    print(
        f"Created {len(visual_documents)} visual descriptions."
    )
    return visual_documents


# --------------------------------------------------
# 7. CREATE CHROMADB VECTOR DATABASE
# --------------------------------------------------

def create_vector_database(documents):
    if not documents:
        raise ValueError("No documents available for indexing.")

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Rebuild database from the latest ingestion
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Creating embeddings and indexing documents...")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH),
    )

    return vectorstore


# --------------------------------------------------
# 8. MAIN INGESTION PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Place the PDF at Backend/Data/nasa_handbook.pdf"
        )

    print("\nSTEP 1: Extracting PDF text...")
    text_documents = load_pdf()

    print("\nSTEP 2: Creating text chunks...")
    text_chunks = create_text_chunks(text_documents)

    print("\nSTEP 3: Extracting tables...")
    table_documents = extract_tables()

    print("\nSTEP 4: Analyzing selected diagrams...")
    visual_documents = extract_visual_content()

    # Combine text, tables, and visual descriptions
    all_documents = (
        text_chunks
        + table_documents
        + visual_documents
    )

    print("\nSTEP 5: Creating embeddings and indexing...")
    create_vector_database(all_documents)

    print("\n" + "=" * 50)
    print("MULTIMODAL INGESTION COMPLETED")
    print("=" * 50)
    print(f"Text chunks: {len(text_chunks)}")
    print(f"Tables: {len(table_documents)}")
    print(f"Visual descriptions: {len(visual_documents)}")
    print(f"Total indexed documents: {len(all_documents)}")
    print(f"Database location: {DB_PATH}")
