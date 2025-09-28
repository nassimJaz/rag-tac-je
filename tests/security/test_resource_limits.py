import pytest
import os
import shutil
import json
from app.security.resource_limits import JSONSecurityValidator

class TestJSONSecurityValidator:

    def setup_method(self):
        self.test_dir = "test_docs"
        os.makedirs(self.test_dir, exist_ok=True)
        self.validator = JSONSecurityValidator(
            max_size_mb=1,
            max_depth=5,
            max_keys=10
        )

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_json_validator_allows_valid_json(self):
        valid_json = {"a": {"b": {"c": "d"}}}
        file_path = os.path.join(self.test_dir, "valid.json")
        with open(file_path, "w") as f:
            json.dump(valid_json, f)

        is_valid, message, data = self.validator.validate(file_path)
        assert is_valid is True
        assert message == "Validation successful"
        assert data == valid_json

    def test_json_validator_rejects_deep_nesting(self):
        deep_json = {"a": {"b": {"c": {"d": {"e": {"f": "g"}}}}}}
        file_path = os.path.join(self.test_dir, "deep.json")
        with open(file_path, "w") as f:
            json.dump(deep_json, f)

        is_valid, message, data = self.validator.validate(file_path)
        assert is_valid is False
        assert "JSON depth exceeds limit" in message

    def test_json_validator_rejects_large_file(self):
        large_file_path = os.path.join(self.test_dir, "large.json")
        with open(large_file_path, "wb") as f:
            f.write(os.urandom(2 * 1024 * 1024))  # 2MB

        is_valid, message, data = self.validator.validate(large_file_path)
        assert is_valid is False
        assert "File size exceeds" in message

    def test_json_validator_rejects_malformed_json(self):
        malformed_file_path = os.path.join(self.test_dir, "malformed.json")
        with open(malformed_file_path, "w") as f:
            f.write("not a json")

        is_valid, message, data = self.validator.validate(malformed_file_path)
        assert is_valid is False
        assert "Invalid JSON structure" in message

    def test_json_validator_rejects_too_many_keys(self):
        too_many_keys_json = {f"key{i}": i for i in range(15)}
        file_path = os.path.join(self.test_dir, "too_many_keys.json")
        with open(file_path, "w") as f:
            json.dump(too_many_keys_json, f)

        is_valid, message, data = self.validator.validate(file_path)
        assert is_valid is False
        assert "JSON key count exceeds limit" in message