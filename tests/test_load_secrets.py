import os
import unittest
from unittest.mock import patch
from app.load_secrets import LoadSecrets

class TestLoadSecrets(unittest.TestCase):

    def tearDown(self):
        # Reset singleton instance for isolation between tests
        LoadSecrets._instance = None

    @patch.dict(os.environ, {"TEMPERATURE": "1.5"})
    def test_valid_temperature(self):
        secrets = LoadSecrets()
        self.assertEqual(secrets.get_temperature(), 1.5)

    @patch.dict(os.environ, {"TEMPERATURE": "-1.0"})
    def test_temperature_below_range(self):
        secrets = LoadSecrets()
        self.assertEqual(secrets.get_temperature(), 0.0)

    @patch.dict(os.environ, {"TEMPERATURE": "3.0"})
    def test_temperature_above_range(self):
        secrets = LoadSecrets()
        self.assertEqual(secrets.get_temperature(), 0.0)

    @patch.dict(os.environ, {"TEMPERATURE": "invalid"})
    def test_invalid_temperature(self):
        secrets = LoadSecrets()
        self.assertEqual(secrets.get_temperature(), 0.0)

    @patch.dict(os.environ, {}, clear=True)
    def test_default_temperature(self):
        secrets = LoadSecrets()
        self.assertEqual(secrets.get_temperature(), 0.0)

if __name__ == '__main__':
    unittest.main()
