from haystack_integrations.components.generators.ollama import OllamaGenerator
from app.load_secrets import LoadSecrets
from portkey_ai import Portkey

class LlmLib:
    _instance = None

    def __new__(cls):
        # Implements a Singleton pattern to ensure only one instance of LlmLib exists.
        # This prevents redundant initialization of LLM clients and secret loading.
        if cls._instance is None:
            cls._instance = super(LlmLib, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.load_secrets = LoadSecrets()
            self.portkey_client = None
            self.llm_generation_ollama = None
            self.llm_hyde_ollama = None
            self._initialized = True
    
    def get_portkey_key(self):
        key = self.load_secrets.get_portkey_key()
        if not key:
            raise RuntimeError("[ERROR] : The portkey api key is not set.")
        return key
    
    def get_generation_model(self):
        return self.load_secrets.get_generation_model()
    
    def get_hyde_model(self):
        return self.load_secrets.get_hyde_model()
    
    def get_ollama_host(self):
        return self.load_secrets.get_ollama_host()
    
    def get_temperature(self):
        return self.load_secrets.get_temperature()
    
    def build_generation_llm_ollama(self):
        """Builds and returns an OllamaGenerator instance for final answer generation. Handles lazy initialization."""  
        if self.llm_generation_ollama is None:
            try:
                self.llm_generation_ollama = OllamaGenerator(  
                    model=self.get_generation_model(),
                    url=self.get_ollama_host(),
                    generation_kwargs={"temperature": self.get_temperature()}
                    )
            except Exception as e:
                raise RuntimeError(f"[ERROR] : Ollama went wrong : {e}")
        return self.llm_generation_ollama
    
    def build_hyde_llm_ollama(self):
        """Builds and returns an OllamaGenerator instance specifically for HyDE (Hypothetical Document Embedding) generation. Handles lazy initialization."""  
        if self.llm_hyde_ollama is None:
            try:
                self.llm_hyde_ollama = OllamaGenerator(  
                    model=self.get_hyde_model(),
                    url=self.get_ollama_host(),
                    generation_kwargs={"temperature": self.get_temperature()}
                    )
            except Exception as e:
                raise RuntimeError(f"[ERROR] : Ollama went wrong : {e}")
        return self.llm_hyde_ollama

    def build_portkey_client(self):
        """Builds and returns a Portkey client instance. Handles lazy initialization."""
        if self.portkey_client is None:
            self.portkey_client = Portkey(api_key=self.get_portkey_key())
        return self.portkey_client
                
