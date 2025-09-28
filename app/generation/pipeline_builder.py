import os
from app.generation.llm_lib import LlmLib
from app.load_secrets import LoadSecrets
import jinja2
from jinja2 import Environment, FileSystemLoader

class PipelineBuilder:
    _instance = None

    def __new__(cls):
        # Implements a Singleton pattern to ensure only one instance of PipelineBuilder exists.
        # This prevents redundant initialization of prompt templates and LLM libraries.
        if cls._instance is None:
            cls._instance = super(PipelineBuilder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.prompt_template = None
            self.load_secrets = LoadSecrets()
            self.llm_lib = LlmLib()
            self.prompts_dir = self.load_secrets.get_prompts_dir()
            self._initialized = True
        
    def get_provider(self):
        return (self.load_secrets.get_provider() or "").lower()
    
    def get_llm_lib(self):
        return self.llm_lib

    def get_prompt_template(self) -> jinja2.Template:
        """
        Builds and returns a Jinja2 prompt template for generating responses.
        This template defines the structure and instructions for the LLM,
        including how to incorporate available contexts and format the output.
        It's designed to guide the LLM in identifying relevant tables and variables
        from the provided documents to answer data science questions.
        """
        if self.prompt_template is None:
            env = Environment(loader=FileSystemLoader(self.prompts_dir))
            self.prompt_template = env.get_template("generation.j2")
        return self.prompt_template

    def get_llm_generation(self):
        """
        Retrieves the appropriate LLM client for generation based on the configured provider.
        It delegates to `LlmLib` to build the specific LLM instance (Portkey or Ollama).
        """
        provider = self.get_provider()
        match provider:
            case "portkey":
                return self.get_llm_lib().build_portkey_client()
            case "ollama":
                return self.get_llm_lib().build_generation_llm_ollama()
            case _:
                raise RuntimeError(f"[Error] : Unknow provider for LLM generation : {provider}")
