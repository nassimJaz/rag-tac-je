import pytest
import os
import shutil
from app.security.file_validator import FileValidator

class TestFileValidator:

    def setup_method(self):
        self.test_dir = "test_docs"
        os.makedirs(self.test_dir, exist_ok=True)
        self.validator = FileValidator(
            allowed_extensions=[".txt", ".pdf"],
            max_size_mb=1,
            base_dir=self.test_dir
        )

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_file_validator_allows_valid_file(self):
        # Create a valid file
        valid_file_path = os.path.join(self.test_dir, "valid.txt")
        with open(valid_file_path, "w") as f:
            f.write("some content")

        is_valid, message = self.validator.validate(valid_file_path)
        assert is_valid is True
        assert message == "File is valid"

    def test_file_validator_rejects_path_traversal(self):
        # Attempt path traversal
        malicious_path = os.path.join(self.test_dir, "..", "..", "etc", "passwd")
        is_valid, message = self.validator.validate(malicious_path)
        assert is_valid is False
        assert "Path traversal attempt detected" in message

    def test_file_validator_rejects_bad_extension(self):
        # Create a file with a disallowed extension
        invalid_file_path = os.path.join(self.test_dir, "invalid.exe")
        with open(invalid_file_path, "w") as f:
            f.write("some content")

        is_valid, message = self.validator.validate(invalid_file_path)
        assert is_valid is False
        assert "File extension not allowed" in message

    def test_file_validator_rejects_file_too_large(self):
        # Create a file that is too large
        large_file_path = os.path.join(self.test_dir, "large.txt")
        with open(large_file_path, "wb") as f:
            f.write(os.urandom(2 * 1024 * 1024))  # 2MB

        is_valid, message = self.validator.validate(large_file_path)
        assert is_valid is False
        assert "File size exceeds the limit" in message

    def test_file_validator_rejects_non_existent_file(self):
        non_existent_file_path = os.path.join(self.test_dir, "non_existent.txt")
        is_valid, message = self.validator.validate(non_existent_file_path)
        assert is_valid is False
        assert "File size exceeds the limit" in message # This is because getsize raises OSError

    def test_file_validator_rejects_file_outside_base_dir(self):
        # Create a file outside the base directory
        outside_file_path = os.path.abspath("outside.txt")
        with open(outside_file_path, "w") as f:
            f.write("some content")

        is_valid, message = self.validator.validate(outside_file_path)
        assert is_valid is False
        assert "Path traversal attempt detected" in message

        os.remove(outside_file_path)