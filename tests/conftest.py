"""
conftest.py - Mock heavy ML modules before any test imports them.

This prevents torch, transformers, and gradio from being loaded during tests,
making the test suite fast (~seconds instead of minutes).
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Create mock modules for heavy dependencies BEFORE they are imported anywhere.
# This ensures that when the app modules do `import torch`, `import gradio`, etc.
# they receive lightweight mocks instead of loading gigabytes of ML frameworks.
# ---------------------------------------------------------------------------

# Mock torch
mock_torch = MagicMock()
mock_torch.device.return_value = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.tensor = MagicMock()
mock_torch.zeros = MagicMock()
mock_torch.ones = MagicMock()
mock_torch.nn.functional.normalize = lambda x, **kw: x
mock_torch.argsort = MagicMock(return_value=[0])

# Mock torchvision
mock_torchvision = MagicMock()

# Mock transformers
mock_transformers = MagicMock()

# Mock gradio
mock_gradio = MagicMock()
mock_gradio.Blocks.return_value.__enter__ = MagicMock()
mock_gradio.Blocks.return_value.__exit__ = MagicMock(return_value=False)

# Mock dotenv
mock_dotenv = MagicMock()
mock_dotenv.load_dotenv = MagicMock()

# Register mocks in sys.modules
sys.modules.setdefault("torch", mock_torch)
sys.modules.setdefault("torch.nn", mock_torch.nn)
sys.modules.setdefault("torch.nn.functional", mock_torch.nn.functional)
sys.modules.setdefault("torch.hub", mock_torch.hub)
sys.modules.setdefault("torchvision", mock_torchvision)
sys.modules.setdefault("transformers", mock_transformers)
sys.modules.setdefault("gradio", mock_gradio)
sys.modules.setdefault("dotenv", mock_dotenv)
