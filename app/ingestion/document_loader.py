import os
import glob
import csv
import json
from typing import List
from pypdf import PdfReader
from haystack.dataclasses.document import Document
import logging

from app.load_secrets import LoadSecrets
from app.security.file_validator import FileValidator
from app.security.resource_limits import JSONSecurityValidator

logger = logging.getLogger(__name__)

class DocumentLoader:
    _list_docs: List[Document]

    def __init__(self):
        """
        Initializes the DocumentLoader, setting up file directories, delimiters,
        and security validators for various document types (PDF, CSV, JSON).
        """
        load_secret = LoadSecrets()
        self.file_dir = load_secret.get_file_dir()
        self.csv_delimiter = load_secret.get_csv_delimiter()
        self._list_docs = []  # Stores the loaded Haystack Document objects
        self.validator = FileValidator(
            allowed_extensions=[ ".pdf", ".csv", ".json"],
            max_size_mb=100,
            base_dir=self.file_dir
        )
        self.json_validator = JSONSecurityValidator()

    def get_file_dir(self) -> str:
        return self.file_dir
    
    def get_csv_delimiter(self) -> str:
        return self.csv_delimiter
    
    def set_doc(self, document : Document):
        self._list_docs.append(document)
    
    def load_pdfs_from_dir(self):
        """
        Loads PDF documents from the configured file directory.
        Each page of a valid PDF is treated as a separate Haystack Document.
        Includes file validation and error handling for corrupted or invalid PDFs.
        """
        pdf_paths = sorted(glob.glob(os.path.join(self.get_file_dir(), "**", "*.pdf"), recursive=True))
        for path in pdf_paths:
            is_valid, message = self.validator.validate(path)
            if not is_valid:
                logger.warning("Skipping invalid file: %s", message)
                continue
            try :
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():  # skip empty pages
                        doc = Document(
                            content=text,
                        meta={"source": os.path.basename(path), "page": i + 1}
                        )
                        if not self.clear_doc(doc=doc): self.set_doc(document=doc)
            except Exception as e:
                logger.warning("PDF failed for %s", path, exc_info=e)
                continue

    def load_csvs_from_dir(self):
        """
        Loads CSV documents from the configured file directory.
        Each CSV file is read, cleaned (empty rows/cells removed), and its content
        is stored as a single Haystack Document. Includes file validation.
        """
        csv_paths = sorted(glob.glob(os.path.join(self.get_file_dir(), "**", "*.csv"), recursive=True))
        for path in csv_paths:
            is_valid, message = self.validator.validate(path)
            if not is_valid:
                logger.warning("Skipping invalid file: %s", message)
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    delimiter = self.get_csv_delimiter()
                    reader = csv.reader(f, delimiter=delimiter, skipinitialspace=True)
                    rows = list(reader)
                    if not rows :
                        continue
                    cleaned_rows = []
                    for row in rows:
                        stripped_row = [cell.strip() for cell in row if cell.strip()]
                        if stripped_row:  # Only add non-empty rows
                            cleaned_rows.append(stripped_row)
                    # Join rows into CSV string
                    content = "\n".join([delimiter.join(row) for row in cleaned_rows])
                    doc = Document(content=content, meta={"source": path, "type": "csv"})
                    if not self.clear_doc(doc=doc): self.set_doc(document=doc)
            except Exception as e:
                logger.warning("CSV failed for %s", path, exc_info=e)
                continue

    def load_jsons_from_dir(self):
        """
        Loads JSON documents from the configured file directory.
        Each JSON file is parsed. If it's a list, each item becomes a Document.
        If it's a single object, it becomes a Document. Includes file validation.
        """
        json_paths = sorted(glob.glob(os.path.join(self.get_file_dir(), "**", "*.json"), recursive=True))
        for path in json_paths:
            is_valid, message = self.validator.validate(path)
            if not is_valid:
                logger.warning("Skipping invalid file: %s", message)
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # If the JSON is a list of objects, create a document for each object.
                # Otherwise, treat the entire JSON content as a single document.
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        content = json.dumps(item, ensure_ascii=False, indent=2)
                        doc = Document(content=content, meta={"source": path, "record_index": idx, "type": "json"})
                        if not self.clear_doc(doc=doc): self.set_doc(document=doc)
                else:
                    # This block seems to duplicate the list handling if data is iterable but not a list.
                    # It might be intended to handle single JSON objects as well.
                    for idx, item in enumerate(data):
                        content = json.dumps(data, ensure_ascii=False, indent=2)
                        doc = Document(content=content, meta={"source": path, "record_index": idx, "type": "json"})
                        if not self.clear_doc(doc=doc): self.set_doc(document=doc)
            except Exception as e:
                logger.warning("JSON load failed for %s", path, exc_info=e)
                continue
    
    def load_text_files_from_dir(self):
        """
        Loads text (.txt) and markdown (.md) documents from the configured file directory.
        Each file is read and its content is stored as a single Haystack Document.
        Includes file validation.
        """
        all_paths = []
        for ext in ["*.txt", "*.md"]:
            all_paths.extend(sorted(glob.glob(os.path.join(self.get_file_dir(), "**", ext), recursive=True)))

        for path in all_paths:
            is_valid, message = self.validator.validate(path)
            if not is_valid:
                logger.warning("Skipping invalid file: %s", message)
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():  # skip empty files
                        file_extension = os.path.splitext(path)[1].lower()
                        doc_type = "markdown" if file_extension == ".md" else "text"
                        doc = Document(
                            content=content,
                            meta={"source": os.path.basename(path), "type": doc_type}
                        )
                        if not self.clear_doc(doc=doc): self.set_doc(document=doc)
            except Exception as e:
                logger.warning("Text/MD file failed for %s", path, exc_info=e)
                continue

    def clear_doc(self, doc):
        """Helper to check if a document is valid before adding it to the list."""
        if not isinstance(doc, Document):
            return True
        return False
    
    def load_all(self):
        self.load_pdfs_from_dir()
        self.load_csvs_from_dir()
        self.load_jsons_from_dir()
        self.load_text_files_from_dir()
        logger.info("%d documents loaded", len(self._list_docs))
    
    def get_list_docs(self) -> List[Document]:
        """Load all documents from dir and return the list of documents"""
        self.load_all()
        return self._list_docs
