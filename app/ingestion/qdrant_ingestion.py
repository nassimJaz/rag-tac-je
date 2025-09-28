from app.vector_db.qdrant_store import QdrantStore
from haystack.dataclasses.document import Document
from haystack.components.writers import DocumentWriter
from app.ingestion.qdrant_embedding_ingestion import QdrantEmbeddingIngestion
import logging
from qdrant_client.http import exceptions as qdrant_exceptions

from app.ingestion.document_splitter import DocumentSplitting

logger = logging.getLogger(__name__)

class QdrantIngestion:
    _instance = None

    def __new__(cls):
        # Implements a Singleton pattern to ensure only one instance of QdrantIngestion exists.
        # This prevents redundant initialization of Qdrant connections and ingestion components.
        if cls._instance is None:
            cls._instance = super(QdrantIngestion, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized :
            self.qdrant_store = QdrantStore()
            self.qdrant_embedding_ingestion = QdrantEmbeddingIngestion()
            self.document_splitting = DocumentSplitting()
            self.writer = None
            self._initialized = True  

    def get_qdrant_store(self):
        return self.qdrant_store.get_connexion()
      
    def get_qdrant_ingestion_run(self, docs : list[Document]):
        return self.qdrant_embedding_ingestion.run_embedding(documents=docs)
    
    def get_writer(self):
        if self.writer is None:
            self.writer = DocumentWriter(document_store=self.get_qdrant_store())
        return self.writer
    
    def get_document_splitting(self):
        return self.document_splitting
    
    def get_docs(self):
        return self.get_document_splitting().split_docs()
    
    def index_docs(self) :
        """
        Orchestrates the document ingestion process:
        1. Splits raw documents into smaller chunks.
        2. Generates embeddings for these document chunks.
        3. Writes the embedded documents to the Qdrant vector store.
        """
        try:
            docs_with_embeddings = self.get_qdrant_ingestion_run(docs=self.get_docs())
            self.get_writer().run(documents=docs_with_embeddings)
        except qdrant_exceptions.UnexpectedResponse as e:
            logger.exception("An error occurred during document indexing.")

