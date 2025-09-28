import os
from haystack.components.builders import PromptBuilder
from haystack import Pipeline
from app.generation.llm_lib import LlmLib
from app.load_secrets import LoadSecrets
import jinja2
from jinja2 import Environment, FileSystemLoader
import re

class PipelineHydeRetriever:
    _instance = None

    def __new__(cls):
        # Implements a Singleton pattern to ensure only one instance of PipelineHydeRetriever exists.
        # This prevents redundant initialization of LLM clients and prompt builders for HyDE.
        if cls._instance is None:
            cls._instance = super(PipelineHydeRetriever, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.llm_lib = LlmLib()
            self.prompt_builder = None
            self.hyde_pipeline = None
            self._initialized = True
        self.load_secrets = LoadSecrets()
        self.prompts_dir = self.load_secrets.get_prompts_dir()

    def get_llm_lib(self):
        return self.llm_lib
    
    def get_provider(self):
        return (self.load_secrets.get_provider() or "").lower()
    
    def get_llm_model(self):
        return self.load_secrets.get_hyde_model()
    
    def get_build_prompt(self) -> jinja2.Template:
        """
        Builds and returns a Jinja2 prompt template specifically for HyDE (Hypothetical Document Embedding).
        This template instructs the LLM to generate a hypothetical document (in JSON format)
        based on the user's query, which is then used for retrieval.
        """
        if self.prompt_builder is None:
            env = Environment(loader=FileSystemLoader(self.prompts_dir))
            self.prompt_template = env.get_template("hyde.j2")
        return self.prompt_template
    
    def get_llm_hyde(self):
        provider = self.get_provider()
        match provider:
            case "portkey":
                return self.get_llm_lib().build_portkey_client()
            case "ollama":
                return self.get_llm_lib().build_hyde_llm_ollama()
            case _:
                raise ValueError(f"[Error] : Unknow provider for LLM generation : {provider}")
            
    def generate_hypothetical_document(self, query:str):
        """
        Generates a hypothetical document based on the input query using the configured LLM.
        This hypothetical document is then used to improve retrieval results (HyDE).
        The method handles different LLM providers (Portkey, Ollama) and their respective API calls.
        """
        prompt_template = self.get_build_prompt()
        prompt = prompt_template.render(query=query)

        llm = self.get_llm_hyde()
        provider = self.get_provider()

        answer = ""
        match provider:
            case "portkey":
                try:
                    response = llm.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=self.load_secrets.get_temperature(),
                    model=f"@rag_llm/{self.get_llm_model()}"
                    )
                    answer = response.choices[0].message.content
                    #print(f"\n---------HyDE Hypo doc :--------\n{answer}\n----------------------\n\n")
                except Exception as e:
                        # Network error / API error (invalid key, model unavailable) / Choices is None
                        raise RuntimeError(f"[ERROR] : Portkey went wrong : {e}")
            case "ollama":
                try :
                    response = llm.run(prompt)
                    answer = response["replies"][0]
                    #print(f"\n---------HyDE Hypo doc :--------\n{answer}\n----------------------\n\n")
                except Exception as e:
                    raise RuntimeError(f"[ERROR] : Ollama went wrong : {e}")
            case _:
                raise RuntimeError(f"Unknown provider: {provider}")
        
        # Extract JSON content if wrapped in markdown code block
        json_match = re.search(r"```json\s*(.*?)\s*```", answer, re.DOTALL)
        if json_match:
            answer = json_match.group(1)

        return answer
