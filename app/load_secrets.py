import os
from dotenv import load_dotenv
from haystack.utils import ComponentDevice
import torch
import logging

logger = logging.getLogger(__name__)

class LoadSecrets:
    _instance = None

    def __new__(cls):
        # Implements a Singleton pattern to ensure only one instance of LoadSecrets exists.
        # This prevents redundant loading of environment variables.
        if cls._instance is None:
            cls._instance = super(LoadSecrets, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            load_dotenv()  # Loads environment variables from a .env file
            self.model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/distiluse-base-multilingual-cased-v2")
            
            # Determines whether to use GPU or CPU based on the USE_GPU environment variable
            # and CUDA availability. This optimizes performance for embedding models.
            use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
            if use_gpu and torch.cuda.is_available():
                logger.info("Using GPU")
                self.device = ComponentDevice.from_str("cuda")
            else:
                logger.info("Using CPU")
                self.device = ComponentDevice.from_str("cpu")

            self.embed_dim = int(os.getenv("EMBED_DIM", "512"))
            self.topk = int(os.getenv("TOP_K", 10))
            self.rag_method = os.getenv("RAG_METHOD", "similarity")
            self.provider = os.getenv("PROVIDER", "ollama")
            self.file_dir = os.getenv("FILE_DIR", "./docs")
            self.csv_delimiter = os.getenv("CSV_DELIMITER", ",")
            self.chunk_size = int(os.getenv("CHUNK_SIZE", 1024))
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 300))
            
            self.portkey_api_key = None
            raw_key = os.getenv("PORTKEY_API_KEY")
            if raw_key is not None : self.portkey_api_key = self._validate_portkey_api_key(raw_key)
            self.slug_portkey = os.getenv("SLUG_PORTKEY", "rag_llm")

            self.generation_model = os.getenv("GENERATION_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
            self.hyde_model = os.getenv("HYDE_MODEL", "mistralai/devstral-small-2505:free")
            self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
            self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.temperature_answer = self._validate_temperature(os.getenv("TEMPERATURE"))
            self.cross_encorder = os.getenv("CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker_topk = int(os.getenv("RERANKER_TOP_K", 3))
            self.reranker_enable = os.getenv("RERANKER_ENABLE", "true").lower() == "true"
            self.prompts_dir = os.getenv("PROMPTS_DIR", "./app/prompts")

            self.qdrant_key = os.getenv("QDRANT_API_KEY")
            self._initialized = True

    def _validate_temperature(self, temp_str: str | None) -> float:
        """
        Validates the temperature value loaded from environment variables.
        Ensures the temperature is within a valid range [0.0, 2.0] and handles invalid inputs
        by returning a default value and logging a warning.
        """
        default_temp = 0.0
        if temp_str is None:
            return default_temp
        try:
            temp = float(temp_str)
            if 0.0 <= temp <= 2.0:
                return temp
            else:
                logger.warning(f"Temperature {temp} is out of range [0.0, 2.0]. Using default value {default_temp}")
                return default_temp
        except ValueError:
            logger.warning(f"Invalid temperature value '{temp_str}'. Using default value {default_temp}")
            return default_temp

    def get_model(self):
        return self.model
    
    def get_device(self):
        return self.device
    
    def get_embed_dim(self):
        return self.embed_dim
    
    def get_topk(self):
        return self.topk
    
    def get_rag_method(self):
        return self.rag_method
    
    def get_provider(self):
        return self.provider

    def get_file_dir(self):
        os.makedirs(self.file_dir, exist_ok=True)
        return self.file_dir
    
    def get_csv_delimiter(self):
        return self.csv_delimiter
    
    def get_chunk_size(self):
        return self.chunk_size
    
    def get_chunk_overlap(self):
        return self.chunk_overlap

    def _validate_portkey_api_key(self, api_key: str | None) -> str | None:
        """
        Validates the Portkey API key loaded from environment variables.
        Checks for minimum length to ensure a basic level of validity.
        """
        if not api_key:
            return None

        # Minimum length check
        if len(api_key) < 16:
            raise ValueError("Invalid PORTKEY_API_KEY: Key is too short (minimum 16 characters).")

        return api_key
    
    def get_portkey_key(self):
        return self.portkey_api_key
    
    def get_portkey_slug(self):
        return self.slug_portkey
    
    def set_portkey_slug(self, slug: str):
        self.slug_portkey = slug
    
    def get_generation_model(self):
        return self.generation_model
    
    def get_hyde_model(self):
        return self.hyde_model
    
    def get_ollama_model(self):
        return self.ollama_model
    
    def get_ollama_host(self):
        return self.ollama_host

    def get_temperature(self):
        return self.temperature_answer
    
    def get_cross_encoder(self):
        return self.cross_encorder
    
    def get_reranker_topk(self):
        return self.reranker_topk
    
    def get_reranker_enable(self):
        return self.reranker_enable
    
    def get_prompts_dir(self):
        return self.prompts_dir
    
    def get_qdrant_key(self):
        return self.qdrant_key
