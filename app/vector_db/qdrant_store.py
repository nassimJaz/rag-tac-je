import os
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from app.load_secrets import LoadSecrets
from haystack.utils import Secret

class QdrantStore :
    _instance = None
    load_secrets = LoadSecrets()

    def __new__(cls):
        # Implements a Singleton pattern to ensure only one instance of QdrantStore exists.
        # This prevents redundant connections to the Qdrant vector database.
        if cls._instance is None:
            cls._instance = super(QdrantStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) :
        if not self._initialized:
            self.document_store = None
            self._initialized = True
    
    def get_embedding_dim(self):
        return self.load_secrets.get_embed_dim()


    def get_connexion(self) -> QdrantDocumentStore:
        """
        Establishes and returns a connection to the Qdrant document store.
        It uses lazy initialization to create the connection only when first accessed.
        The connection can be configured for local persistent storage.
        """
        if self.document_store is None :
            try :
                host = os.getenv("QDRANT_HOST", "localhost")
                port = int(os.getenv("QDRANT_PORT", 6333))
                qdrant_url = f"http://{host}:{port}"

                self.document_store = QdrantDocumentStore(
                    url=qdrant_url,
                    api_key=Secret.from_token(self.load_secrets.get_qdrant_key()),
                    embedding_dim= self.get_embedding_dim(),
                    index= "Documents",
                    recreate_index= False
                )
            except Exception as e :
                raise RuntimeError(f"[ERROR] : Unable to connect to Qdrant : {e}")
        return self.document_store
