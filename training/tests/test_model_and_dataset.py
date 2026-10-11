import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.model import DrowsinessCNN
from training.dataset_loader import EyeDataset


class TestDrowsinessCNN(unittest.TestCase):
    def test_output_shape(self):
        model = DrowsinessCNN()
        model.eval()

        with torch.no_grad():
            output = model(torch.randn(4, 1, 24, 24))

        self.assertEqual(tuple(output.shape), (4, 2))

    def test_outputs_are_finite(self):
        model = DrowsinessCNN()
        model.eval()

        with torch.no_grad():
            output = model(torch.randn(2, 1, 24, 24))

        self.assertTrue(torch.isfinite(output).all().item())


class TestEyeDataset(unittest.TestCase):
    def test_loads_open_and_closed_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for class_name in ("open", "closed"):
                folder = root / class_name
                folder.mkdir()
                Image.new("RGB", (40, 30), color="white").save(
                    folder / "sample.png"
                )

            dataset = EyeDataset(root)

            self.assertEqual(len(dataset), 2)
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (1, 24, 24))
            self.assertEqual(image.dtype, torch.float32)
            self.assertGreaterEqual(label, 0)
            self.assertLessEqual(label, 1)

    def test_rejects_empty_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "open").mkdir()
            (root / "closed").mkdir()

            with self.assertRaises(ValueError):
                EyeDataset(root)


if __name__ == "__main__":
    unittest.main()
