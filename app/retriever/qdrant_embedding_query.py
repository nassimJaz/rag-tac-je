from haystack.components.embedders import SentenceTransformersTextEmbedder
from app.load_secrets import LoadSecrets
import threading

class QdrantEmbeddingQuery:
    _instance = None
    _instance_lock = threading.Lock()  # verrou pour le singleton

    def __new__(cls):
        # Singleton : s'assure qu'une seule instance de la classe existe
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:  # double-check
                    cls._instance = super(QdrantEmbeddingQuery, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.load_secrets = LoadSecrets()
            self.model_name = self.load_secrets.get_model()
            self.device = self.load_secrets.get_device()
            self.embedding_query = None
            self._embedding_lock = threading.Lock()  # verrou pour warm_up
            self._initialized = True

    def get_model_name(self):
        return self.model_name

    def get_device(self):
        return self.device

    def get_embedding_query(self):
        """
        Retourne un SentenceTransformersTextEmbedder (singleton).
        Chargé + warm_up seulement une fois, en mode thread-safe.
        """
        if self.embedding_query is None:
            with self._embedding_lock:
                if self.embedding_query is None:  # double-check
                    self.embedding_query = SentenceTransformersTextEmbedder(
                        model=self.get_model_name(),
                        device=self.get_device(),
                        normalize_embeddings=True,
                    )
                    self.embedding_query.warm_up()
        return self.embedding_query

    def run_query_embedding(self, query: str):
        """
        Exécute l’embedding d’une requête utilisateur en toute sécurité.
        """
        try:
            embedder = self.get_embedding_query()
            result = embedder.run(text=query)
            return result.get("embedding", [])
        except Exception as e:
            raise RuntimeError(f"[ERROR] : Unable to embed query : {e}")