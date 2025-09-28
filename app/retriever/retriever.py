from typing import List
from haystack.dataclasses.document import Document
from app.retriever.qdrant_retriever import QdrantRetriever

# ----- Instances globales
qdrant_retriever = QdrantRetriever()
# -----

def retrieve(query: str) -> List[Document]:
    """Retrieves relevant documents for a given query using the QdrantRetriever."""
    return qdrant_retriever.retrieve(query)