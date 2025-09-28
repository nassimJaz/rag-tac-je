from typing import List, Dict, Any
from haystack.dataclasses.document import Document
from app.generation.pipeline_builder import PipelineBuilder
from app.load_secrets import LoadSecrets
import logging
import httpx
import ollama
import re

logger = logging.getLogger(__name__)

class GenerateResponse:
    def __init__(self, documents: List[Document], query: str):
        """
        Initializes the GenerateResponse class with retrieved documents and the user's query.
        This class is responsible for generating a response using an LLM and extracting relevant sources.
        """
        self._documents = documents
        self._query = query
        self.pipeline_builder = PipelineBuilder()
        self.load_secrets = LoadSecrets()

    def get_documents(self) -> List[Document]:
        return self._documents

    def get_query(self) -> str:
        return self._query

    def get_llm_model(self):
        return self.load_secrets.get_generation_model()

    def generate(self):
        prompt_template = self.pipeline_builder.get_prompt_template()
        prompt = prompt_template.render(documents=self.get_documents(), query=self.get_query())

        llm = self.pipeline_builder.get_llm_generation()
        provider = self.pipeline_builder.get_provider()

        llm_response = ""
        # Depending on the configured LLM provider (Portkey or Ollama),
        # the appropriate API call is made to generate the response.
        # Error handling is included for each provider.
        match provider:
            case "portkey":
                try :
                    slug=self.load_secrets.get_portkey_slug()
                    response = llm.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=self.load_secrets.get_temperature(),
                        model=f"@{slug}/{self.get_llm_model()}"
                    )
                    llm_response = response.choices[0].message.content
                except (httpx.RequestError, KeyError, IndexError) as e:
                    # Network error / API error (invalid key, model unavailable) / Choices is None
                    logger.exception("Portkey API call failed.")
                    raise RuntimeError("Portkey API call failed.") from e
            case "ollama":
                try :
                    result = llm.run(prompt)
                    llm_response = result["replies"][0]
                except (ollama.ResponseError, KeyError, IndexError) as e:
                    logger.exception("Ollama generation failed.")
                    raise RuntimeError("Ollama generation failed.") from e
            case _:
                logger.error("Unknown provider: %s", provider)
                raise RuntimeError(f"Unknown provider: {provider}")
            
        return llm_response