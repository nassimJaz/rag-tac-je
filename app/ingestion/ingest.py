from app.ingestion.qdrant_ingestion import QdrantIngestion

# ----- Instance globale
qdrant_ingestion = QdrantIngestion()
# -----


def ingest():
    """Ingests documents (PDF/CSV/JSON) and builds a vector store with Qdrant."""
    qdrant_ingestion.index_docs()
    print("Vector store built successfully")

if __name__ == "__main__":
    ingest()