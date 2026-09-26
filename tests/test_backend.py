import tempfile
import unittest
from pathlib import Path

from backend.app import validate_model_artifacts


class ValidateModelArtifactsTests(unittest.TestCase):
    def test_missing_artifacts_raise_helpful_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_dir = Path(tmpdir)
            with self.assertRaises(FileNotFoundError) as context:
                validate_model_artifacts(missing_dir)

            message = str(context.exception)
            self.assertIn("model.pkl", message)
            self.assertIn("train_model.py", message)


if __name__ == "__main__":
    unittest.main()
