
from rag import vectorstore

results = vectorstore.similarity_search(
    "Adapted from INCOSE-TP-2003-002-04, 2015",
    k=10,
)

for doc in results:
    print("\nPDF page:", doc.metadata.get("page"))
    print("Content type:", doc.metadata.get("content_type"))
    print("Text:", doc.page_content[:1000])
