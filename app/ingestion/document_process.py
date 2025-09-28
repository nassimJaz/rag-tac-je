from haystack.dataclasses.document import Document
from sentence_transformers import CrossEncoder
from typing import List
from app.load_secrets import LoadSecrets
from haystack.utils import ComponentDevice

class DocumentProcess:
    # Class to the small process of documents (deduplicate, reranking etc)
    _instance = None
    
    def __new__(cls):
        # Implements a Singleton pattern to ensure only one instance of DocumentProcess exists.
        # This prevents redundant initialization of the cross-encoder model.
        if cls._instance is None:
            cls._instance = super(DocumentProcess, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.cross_encoder = None
        self._initialized = True
        self.load_secrets = LoadSecrets()
    
    def get_cross_encorder(self)->CrossEncoder:
        """
        Initializes and returns the CrossEncoder model for document reranking.
        The device (CPU/GPU) for the model is selected based on configuration.
        """
        if self.cross_encoder is None:
            match self.load_secrets.get_provider:
                case "ollama":
                    device="cpu"
                case _:
                    device=self.load_secrets.get_device()._single_device.type.value
            self.cross_encoder = CrossEncoder(
                model_name_or_path = self.load_secrets.get_cross_encoder(),
                device=device
            )
        return self.cross_encoder
    
    def get_reranker_topk(self):
        return self.load_secrets.get_reranker_topk()
    
    def get_reranker_enable(self) -> bool:
        return self.load_secrets.get_reranker_enable()

    @staticmethod
    def deduplicate_docs(docs:List[Document]) -> List[Document]:
        """
        Deduplicates a list of Haystack Document objects based on their source, page, and a snippet of their content.
        This ensures that only unique documents are processed further.
        """
        seen=set()
        cleaned: List[Document] = []
        for d in docs:
            src = d.meta.get("source")
            page = d.meta.get("page")
            key = (src, page, d.content[:120])  # Use a tuple for hashing, content snippet for uniqueness
            if key not in seen:
                seen.add(key)
                cleaned.append(d)
        return cleaned
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Reranks a list of Haystack Document objects based on their relevance to a given query.
        It uses a CrossEncoder model to assign a score to each document and then sorts them.
        If reranking is disabled or no documents are provided, it returns the original list.
        """
        if not documents : return []
        if not self.get_reranker_enable(): return documents
        try :
            # Create pairs of [query, document_content] for the model
            pairs = [[query, doc.content] for doc in documents]
            # Calculate the score [0,1] for each documents based on the query
            scores = self.get_cross_encorder().predict(pairs, show_progress_bar=False)
            for i in range(len(documents)):
                # Add the scores to the document metadata   
                documents[i].meta['rerank_score'] = scores[i]
            # Sort the documents by score in descending order
            docs = sorted(documents, key=lambda doc: doc.meta['rerank_score'], reverse=True)
            docs = docs[:self.get_reranker_topk()]
            return docs
        except Exception as e:
            print(f"[ERROR] Rerank documents failed: {e}")
            return documents
