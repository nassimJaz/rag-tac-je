import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import httpx
from qdrant_client.http import exceptions as qdrant_exceptions

from app.ingestion.document_loader import DocumentLoader
from app.generation.generate_response import GenerateResponse
from app.generation.pipeline_builder import PipelineBuilder
from app.retriever.qdrant_retriever import QdrantRetriever
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

class TestLogging(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_docs"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('app.ingestion.document_loader.logger')
    def test_logging_on_invalid_file(self, mock_logger):
        # Create a fake invalid file
        invalid_file_path = os.path.join(self.test_dir, "invalid.pdf")
        with open(invalid_file_path, "w") as f:
            f.write("invalid content")

        # Instantiate DocumentLoader and point it to the test directory
        with patch.object(DocumentLoader, 'get_file_dir', return_value=self.test_dir):
            loader = DocumentLoader()
            loader.load_pdfs_from_dir()

        # Assert that logger.warning was called
        self.assertTrue(mock_logger.warning.called)

    @patch('app.generation.generate_response.logger')
    @patch('app.generation.generate_response.PipelineBuilder')
    def test_logging_on_api_error(self, MockPipelineBuilder, mock_logger):
        # Mock the LLM to raise an exception
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = httpx.RequestError("API Error", request=MagicMock())

        # Mock PipelineBuilder instance and its methods
        mock_pipeline_builder_instance = MockPipelineBuilder.return_value
        mock_pipeline_builder_instance.get_llm_generation.return_value = mock_llm
        mock_pipeline_builder_instance.get_provider.return_value = "portkey"
        mock_pipeline_builder_instance.get_prompt_template.return_value.render.return_value = "dummy_prompt"

        # Instantiate GenerateResponse
        gen = GenerateResponse(documents=[], query="test_query")

        # Now, patch the instance's pipeline_builder
        gen.pipeline_builder = mock_pipeline_builder_instance

        # Call generate and assert
        with self.assertRaises(RuntimeError):
            gen.generate()

        # Assert that logger.exception was called
        self.assertTrue(mock_logger.exception.called)

    @patch('app.retriever.qdrant_retriever.logger')
    @patch('app.retriever.qdrant_retriever.QdrantEmbeddingRetriever')
    def test_qdrant_connection_error(self, MockQdrantEmbeddingRetriever, mock_logger):
        # Mock QdrantEmbeddingRetriever.run to raise QdrantException
        headers = httpx.Headers()
        exception = qdrant_exceptions.UnexpectedResponse(404, "Not Found", b"", headers)
        
        mock_retriever_instance = MockQdrantEmbeddingRetriever.return_value
        mock_retriever_instance.run.side_effect = exception

        # Instantiate QdrantRetriever and call search_query
        retriever = QdrantRetriever()
        retriever.retriever = mock_retriever_instance

        with self.assertRaises(RuntimeError):
            retriever.search_query("test query")

        # Assert that logger.exception was called
        self.assertTrue(mock_logger.exception.called)

if __name__ == '__main__':
    unittest.main()