import os
import re
from typing import List

class FileValidator:
    def __init__(self, allowed_extensions: List[str], max_size_mb: int, base_dir: str):
        """
        Initializes the FileValidator with a list of allowed file extensions,
        a maximum allowed file size, and a base directory for path safety checks.
        """
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions]
        self.max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.base_dir = os.path.abspath(base_dir)
        # Regex to ensure filenames contain only alphanumeric characters, underscores, hyphens, and periods.
        self.filename_pattern = re.compile(r"^[a-zA-Z0-9_.-]+$")

    def is_path_safe(self, file_path: str) -> bool:
        """
        Checks if the given file path is safe, meaning it resides within the configured
        base directory and does not contain any path traversal sequences (e.g., '../').
        """
        abs_path = os.path.abspath(file_path)
        return os.path.commonpath([abs_path, self.base_dir]) == self.base_dir

    def is_extension_allowed(self, filename: str) -> bool:
        """Checks if the file's extension is present in the list of allowed extensions."""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.allowed_extensions

    def is_size_allowed(self, file_path: str) -> bool:
        """Checks if the file's size is within the maximum allowed limit."""
        try:
            return os.path.getsize(file_path) <= self.max_size
        except OSError:
            return False

    def is_filename_valid(self, filename: str) -> bool:
        """Checks if the filename contains only allowed characters to prevent injection or other issues."""
        return self.filename_pattern.match(os.path.basename(filename)) is not None

    def validate(self, file_path: str) -> (bool, str): # type: ignore
        """
        Performs a comprehensive set of validation checks on a given file path,
        including path safety, filename validity, allowed extension, and file size.
        Returns a tuple indicating success (boolean) and a message (string).
        """
        filename = os.path.basename(file_path)
        if not self.is_path_safe(file_path):
            return False, f"Path traversal attempt detected for {file_path}"
        if not self.is_filename_valid(filename):
            return False, f"Invalid filename: {filename}"
        if not self.is_extension_allowed(filename):
            return False, f"File extension not allowed for {filename}"
        if not self.is_size_allowed(file_path):
            return False, f"File size exceeds the limit for {file_path}"
        return True, "File is valid"
