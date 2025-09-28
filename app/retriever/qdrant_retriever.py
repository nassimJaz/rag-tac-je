from typing import List
from app.load_secrets import LoadSecrets
from app.vector_db.qdrant_store import QdrantStore
from app.retriever.qdrant_embedding_query import QdrantEmbeddingQuery
from haystack import Document
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
import logging
from qdrant_client.http import exceptions as qdrant_exceptions

from app.retriever.pipeline_hyde_retriever import PipelineHydeRetriever
from app.ingestion.document_process import DocumentProcess

logger = logging.getLogger(__name__)

class QdrantRetriever:
    _instance = None
    
    def __new__(cls):
        # Implements a Singleton pattern to ensure only one instance of QdrantRetriever exists.
        # This prevents redundant initialization of Qdrant connections and retrieval components.
        if cls._instance is None:
            cls._instance = super(QdrantRetriever, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.load_secrets = LoadSecrets()
            self.qdrant_store = QdrantStore()
            self.qdrant_embedding_query = QdrantEmbeddingQuery()
            self.document_process = DocumentProcess()
            self.hyde = PipelineHydeRetriever()
            self.topk = self.load_secrets.get_topk()
            self.method = self.load_secrets.get_rag_method().lower()
            self.provider = self.load_secrets.get_provider().lower()
            self.retriever = None
            self._initialized = True
    
    def get_topk(self):
        return self.topk
    
    def get_method(self):
        return self.method
    
    def get_provider(self):
        return self.provider
    
    def get_document_store(self):
        return self.qdrant_store.get_connexion()
    
    def get_query_embed(self, query: str):
        return self.qdrant_embedding_query.run_query_embedding(query=query)
    
    def get_deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        return self.document_process.deduplicate_docs(docs)
    
    def rerank_docs(self, query:str, docs:List[Document]) -> List[Document]:
        return self.document_process.rerank_documents(query=query, documents=docs)
    
    def get_hyde_pipeline(self):
        return self.hyde

    def get_retriever(self):
        """
        Builds and returns a QdrantEmbeddingRetriever instance.
        This retriever is responsible for fetching relevant documents from the Qdrant store
        based on a given query embedding.
        """
        if self.retriever is None:
            self.retriever = QdrantEmbeddingRetriever(
            document_store=self.get_document_store(),
            top_k=self.get_topk()
            )
        return self.retriever

    def search_query(self, query:str) -> List[Document] :
        """Performs a cosine similarity search in Qdrant using the embedded query."""
        try :
            retriever = self.get_retriever()
            results = retriever.run(query_embedding=self.get_query_embed(query))
            return results["documents"]
        except (qdrant_exceptions.UnexpectedResponse, KeyError) as e:
            logger.exception("Unable to search query in Qdrant.")
            raise RuntimeError("Unable to search query in Qdrant.") from e
    
    def search_sim(self, query:str) -> List[Document]:
        """
        Performs a similarity search in Qdrant and then reranks the retrieved documents.
        This method combines basic retrieval with an optional reranking step to improve relevance.
        """
        docs_sim = self.search_query(query)
        docs = self.rerank_docs(query=query, docs=docs_sim)
        return docs
    
    def search_hyde_and_sim(self, query:str) -> List[Document]:
        """
        Implements the HyDE (Hypothetical Document Embedding) retrieval strategy.
        It first generates a hypothetical answer to the query, embeds it, and uses it for retrieval.
        Then, it performs a standard similarity search with the original query.
        The results from both searches are deduplicated and then reranked to provide a comprehensive set of relevant documents.
        If HyDE generation fails, it gracefully falls back to a similarity search with the original query.
        """
        try:
            hypo_answ = self.get_hyde_pipeline().generate_hypothetical_document(query=query)
            docs_hypo = self.search_query(hypo_answ)
        except RuntimeError as e:
            logger.warning("HyDE step failed, falling back to similarity search only. Reason: %s", e)
            docs_hypo = []
        docs_orig = self.search_query(query)
        #self.log_docs(docs_hypo, docs_orig)
        if not docs_orig and not docs_hypo:
            logger.info("No documents were retrieved.")
        documents=self.get_deduplicate_docs(docs_hypo + docs_orig)
        docs = self.rerank_docs(query=query, docs=documents)
        return docs
    
    def log_docs(self, docs_hypo:List[Document], docs_orig:List[Document]):
        """Logs the number of documents found by HyDE and similarity search, along with their sources."""
        logger.info("Number of documents found with HyDE: %d", len(docs_hypo))
        for d in docs_hypo:
            logger.info("  Source: %s", d.meta["source"])
        logger.info("Number of documents found with Similarity: %d", len(docs_orig))
        for d in docs_orig:
            logger.info("  Source: %s", d.meta["source"])

    def retrieve(self, query:str) -> List[Document]:
        """
        Main retrieval method that dispatches to different retrieval strategies
        (HyDE or similarity) based on the configured RAG method.
        """
        method = self.get_method()
        if method == "hyde":
            return self.search_hyde_and_sim(query)
        elif method == "similarity":
            return self.search_sim(query)
        else:
            logger.error("Unknown RAG method: %s", method)
            raise ValueError(f"Unknown RAG method: {method}")