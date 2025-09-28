import os
import logging
from app.ingestion.ingest import ingest
from app.retriever.qdrant_retriever import QdrantRetriever
from app.generation.generate_response import GenerateResponse

from cli_ans import *
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def build_vector_store(force_rebuild: bool):
    """Load FAISS index from disk if present, otherwise build it (ingest).
    Uses VECTOR_STORE_DIR env var; defaults to ./vector_store if unset.
    """
    vs_dir = os.getenv("VECTOR_STORE_DIR")
    if not vs_dir:
        vs_dir = "vector_store"
        os.environ["VECTOR_STORE_DIR"] = vs_dir  # keep ingest() consistent

    if (force_rebuild) and os.path.isdir(vs_dir) :
        ingest()


def run_rag(query: str, force_rebuild: bool = False):
    try :
        # 1) build vector store or not
        build_vector_store(force_rebuild=force_rebuild)

        # 2) Retrieve with similarity or HyDE based on mode
        logger.info("Retrieving documents...")
        retrieved = QdrantRetriever().retrieve(query=query)
        logger.info("Generating answer...")
        result_gen = GenerateResponse(retrieved, query).generate()
        
        try :
            used_sources = extracts_sources(result_gen, retrieved)
            return {"answer": result_gen, "sources": used_sources}
        except Exception :
            return {"answer" : result_gen, "sources": None}
    except Exception as e:
        logger.exception("An error occurred during RAG execution.")
        #import traceback
        #traceback.print_exc()
    


if __name__ == "__main__":
    logger.info("Type 'exit' to quit the console.")
    query =""
    while(query != "exit"):
        query = input("Question : ")
        if query == "exit":
            break
        result = run_rag(query, force_rebuild=False)

        if result:
            print("\n", result["answer"])
    
            if result["sources"]:
                print("\nSources utilisées:")
                for s in result["sources"]:
                    print("- ", s)

        print("\n\n")
    logger.info("Thank you for testing the RAG :)")
