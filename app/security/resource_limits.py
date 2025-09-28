import os
import json
from typing import Any, Dict, List

# Define limits
MAX_JSON_FILE_SIZE_MB = 50
MAX_JSON_DEPTH = 64
MAX_JSON_KEYS = 1000

class JSONSecurityValidator:
    def __init__(self, max_size_mb=MAX_JSON_FILE_SIZE_MB, max_depth=MAX_JSON_DEPTH, max_keys=MAX_JSON_KEYS):
        """
        Initializes the JSONSecurityValidator with configurable limits for JSON file size,
        nesting depth, and total key count. These limits help prevent denial-of-service attacks
        or excessive resource consumption from malformed or overly complex JSON files.
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_depth = max_depth
        self.max_keys = max_keys

    def check_file_size(self, file_path: str) -> bool:
        """Checks if the file size is within the allowed limit."""
        if os.path.getsize(file_path) > self.max_size_bytes:
            return False
        return True

    def _get_object_depth(self, obj: Any, level: int = 1) -> int:
        """
        Recursively calculates the maximum nesting depth of a Python object (dict or list).
        This is used to enforce `MAX_JSON_DEPTH` and prevent excessively nested JSON structures.
        """
        if not isinstance(obj, (dict, list)) or not obj:
            return level
        
        if isinstance(obj, dict):
            values = obj.values()
        else: # it's a list
            values = obj

        return max(self._get_object_depth(item, level + 1) for item in values) if values else level + 1


    def validate_json_depth(self, data: Any) -> bool:
        """Validates that the nesting depth of the JSON data does not exceed the configured limit."""
        if self._get_object_depth(data) > self.max_depth:
            return False
        return True

    def _count_keys(self, data: Any) -> int:
        """
        Recursively counts the total number of keys within a JSON object (dictionary).
        This helps enforce `MAX_JSON_KEYS` to prevent overly complex JSON structures.
        """
        if not isinstance(data, dict):
            return 0
        
        count = len(data.keys())
        for key, value in data.items():
            if isinstance(value, dict):
                count += self._count_keys(value)
            elif isinstance(value, list):
                for item in value:
                    count += self._count_keys(item)
        return count

    def validate_key_count(self, data: Any) -> bool:
        """Validates that the total number of keys in the JSON data does not exceed the configured limit."""
        if self._count_keys(data) > self.max_keys:
            return False
        return True

    def validate(self, file_path: str) -> (bool, str, Any): # type: ignore
        """
        Performs a comprehensive set of validation checks on a JSON file.
        This includes file size, JSON structure validity, nesting depth, and total key count.
        Returns a tuple indicating success (boolean), a message (string), and the loaded JSON data (if successful).
        """
        if not self.check_file_size(file_path):
            return False, f"File size exceeds {self.max_size_bytes / (1024*1024)}MB limit", None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON structure: {e}", None
        except Exception as e:
            return False, f"Failed to read file: {e}", None

        if not self.validate_json_depth(data):
            return False, f"JSON depth exceeds limit of {self.max_depth}", None

        if not self.validate_key_count(data):
            return False, f"JSON key count exceeds limit of {self.max_keys}", None

        return True, "Validation successful", data
