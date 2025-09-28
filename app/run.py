from app.generation.generate_response import GenerateResponse
from app.retriever.qdrant_retriever import QdrantRetriever

class RunRag():

    def __init__(self):
        self.retriever = QdrantRetriever()

    def get_retriever(self):
        return self.retriever
    
    def run(self, query:str):
        docs_retrieved = self.get_retriever().retrieve(query)
        result_gen = GenerateResponse(docs_retrieved, query).generate()

        return result_gen