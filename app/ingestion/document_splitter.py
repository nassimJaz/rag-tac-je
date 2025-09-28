from typing import List
from haystack.dataclasses.document import Document
from haystack.components.preprocessors import DocumentSplitter
import logging

from app.load_secrets import LoadSecrets
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.document_process import DocumentProcess
from langdetect import detect

logger = logging.getLogger(__name__)

class DocumentSplitting:
    _splitter_cache = {}
    _docs_split:List[Document]

    def __init__(self):
        """
        Initializes the DocumentSplitting class, setting up parameters for document chunking
        and initializing dependencies like `LoadSecrets`, `DocumentLoader`, and `DocumentProcess`.
        """
        self.load_secrets = LoadSecrets()
        self.document_loader = DocumentLoader()
        self.document_process = DocumentProcess()
        self.chunk_size = self.load_secrets.get_chunk_size()
        self.chunk_overlap = self.load_secrets.get_chunk_overlap()
        self._docs_split = []

    def get_chunk_size(self):
        return self.chunk_size
    
    def get_chunk_overlap(self):
        return self.chunk_overlap
    
    def get_document_loader(self):
        return self.document_loader
    
    def set_docs(self, docs:List[Document]):
        if docs: self._docs_split.extend(docs)
    
    def get_docs_split(self):
        return self._docs_split
    
    def get_splitter(self, lang:str) -> DocumentSplitter:
        """
        Retrieves a DocumentSplitter instance for a given language, utilizing a cache
        to avoid redundant object creation. This ensures efficient splitting based on
        configured chunk size and overlap.
        """
        if lang not in self._splitter_cache:
            self._splitter_cache[lang] = DocumentSplitter(
                split_length=self.get_chunk_size(),
                split_overlap=self.get_chunk_overlap(),
                split_by="word",
                language= lang,
                skip_empty_documents=True,
            )
        return self._splitter_cache[lang]

    def split_docs(self) -> List[Document]:
        """
        Loads all documents, detects their language, and then splits them into smaller chunks
        using language-specific document splitters. This process prepares documents for embedding.
        """
        docs = self.get_document_loader().get_list_docs()
        #clean_docs:List[Document] = self.document_process.deduplicate_docs(docs)
        for doc in docs:
            try :
                # Detect language to apply appropriate splitting rules.
                lang = detect(doc.content) if doc.content else "fr" 
                docs_split = self.get_splitter(lang).run(documents=[doc])
                self.set_docs(docs_split.get("documents"))
            except Exception as e:
                logger.warning("Splitting document failed for %s", doc.id, exc_info=e)
        return self.get_docs_split()