import pytest
import json
from unittest.mock import patch, MagicMock
from app.generation.generate_response import GenerateResponse
from haystack.dataclasses.document import Document

class TestGenerateResponse:

    @patch('app.generation.generate_response.PipelineBuilder')
    def test_source_extraction_with_valid_json(self, MockPipelineBuilder):
        llm_response = '''```json
{
    "table": {
        "name": "My Table",
        "objective": "My Objective"
    },
    "variables": [
        {
            "name": "Var1",
            "description": "Desc1",
            "source": [1]
        },
        {
            "name": "Var2",
            "description": "Desc2",
            "source": [3]
        }
    ],
    "sources": [1, 3]
}
```'''
        documents = [
            Document(content="doc1", meta={"source": "source1"}),
            Document(content="doc2", meta={"source": "source2"}),
            Document(content="doc3", meta={"source": "source3"}),
        ]

        mock_llm = MagicMock()
        mock_llm.chat.completions.create.return_value.choices[0].message.content = llm_response

        mock_pipeline_builder_instance = MockPipelineBuilder.return_value
        mock_pipeline_builder_instance.get_llm_generation.return_value = mock_llm
        mock_pipeline_builder_instance.get_provider.return_value = "portkey"
        mock_pipeline_builder_instance.get_prompt_template.return_value.render.return_value = "dummy_prompt"

        gen = GenerateResponse(documents=documents, query="test query")
        result = gen.generate()

        expected_answer = '''>> Table pertinente : "My Table"
    - Objectif de la table : My Objective
Variables pertinentes :
    1. "Var1" - Desc1 [1]
    2. "Var2" - Desc2 [3]\n'
        assert result["answer"] == expected_answer
        assert len(result["sources"]) == 2
        assert "[1] source1" in result["sources"][0]
        assert "[3] source3" in result["sources"][1]'''

    @patch('app.generation.generate_response.PipelineBuilder')
    def test_source_extraction_with_malformed_json(self, MockPipelineBuilder):
        llm_response = '''```json
{
    "answer": "This is the answer.",
    "sources": [1, 3
}
```'''
        documents = [Document(content="doc1")]

        mock_llm = MagicMock()
        mock_llm.chat.completions.create.return_value.choices[0].message.content = llm_response

        mock_pipeline_builder_instance = MockPipelineBuilder.return_value
        mock_pipeline_builder_instance.get_llm_generation.return_value = mock_llm
        mock_pipeline_builder_instance.get_provider.return_value = "portkey"
        mock_pipeline_builder_instance.get_prompt_template.return_value.render.return_value = "dummy_prompt"

        gen = GenerateResponse(documents=documents, query="test query")
        result = gen.generate()

        assert result["answer"] == llm_response
        assert len(result["sources"]) == 0

    @patch('app.generation.generate_response.PipelineBuilder')
    def test_source_extraction_with_no_json(self, MockPipelineBuilder):
        llm_response = "This is the answer without JSON."
        documents = [Document(content="doc1")]

        mock_llm = MagicMock()
        mock_llm.chat.completions.create.return_value.choices[0].message.content = llm_response

        mock_pipeline_builder_instance = MockPipelineBuilder.return_value
        mock_pipeline_builder_instance.get_llm_generation.return_value = mock_llm
        mock_pipeline_builder_instance.get_provider.return_value = "portkey"
        mock_pipeline_builder_instance.get_prompt_template.return_value.render.return_value = "dummy_prompt"

        gen = GenerateResponse(documents=documents, query="test query")
        result = gen.generate()

        assert result["answer"] == llm_response
        assert len(result["sources"]) == 0