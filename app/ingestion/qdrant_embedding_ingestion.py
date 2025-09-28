from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.dataclasses.document import Document
from app.load_secrets import LoadSecrets

class QdrantEmbeddingIngestion:
    _instance = None
    load_secrets = LoadSecrets()

    def __new__(cls):
        # Implements a Singleton pattern to ensure only one instance of QdrantEmbeddingIngestion exists.
        # This prevents redundant initialization and warm-up of the embedding model.
        if cls._instance is None:
            cls._instance = super(QdrantEmbeddingIngestion, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized :
            self.model_name = self.load_secrets.get_model()
            self.device = self.load_secrets.get_device()
            self.embedding_ingestion = None
            self._initialized = True

    def get_model_name(self):
        return self.model_name
    
    def get_device(self):
        return self.device
    
    def get_embedding_ingestion(self):
        """
        Initializes and returns a SentenceTransformersDocumentEmbedder instance.
        The embedder is warmed up on first access to optimize performance.
        """
        if self.embedding_ingestion is None:
            self.embedding_ingestion = SentenceTransformersDocumentEmbedder(
                model=self.get_model_name(),
                device=self.get_device(),
                normalize_embeddings=True,
            )
            self.embedding_ingestion.warm_up()
        return self.embedding_ingestion
    
    def run_embedding(self, documents: list[Document]) -> list:
        """Runs the document embedding process for a list of Haystack Documents."""
        embedder = self.get_embedding_ingestion()
        result = embedder.run(documents=documents)
        return result.get("documents", [])