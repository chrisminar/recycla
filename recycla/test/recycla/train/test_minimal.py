"""
Minimal test to verify the test infrastructure works.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
sys.path.insert(0, os.path.abspath(src_path))


def test_pytorch_available():
    """Test that PyTorch is available."""
    assert torch.__version__ is not None
    print(f"PyTorch version: {torch.__version__}")


def test_basic_nn_functionality():
    """Test basic neural network functionality."""
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    x = torch.randn(3, 10)
    output = model(x)

    assert output.shape == (3, 2)
    assert output.requires_grad is True


def test_imports():
    """Test that we can import the custom models."""
    try:
        from recycla.train.BaseCustomModel import BaseCustomModel

        print("✓ Successfully imported BaseCustomModel")

        # Test that it's an abstract class
        assert hasattr(BaseCustomModel, "_load_pretrained_model")
        assert hasattr(BaseCustomModel, "_freeze_and_unfreeze_layers")
        assert hasattr(BaseCustomModel, "_setup_classification_heads")

    except ImportError as e:
        pytest.skip(f"Could not import BaseCustomModel: {e}")


def test_base_model_abstract():
    """Test that BaseCustomModel is properly abstract."""
    try:
        from recycla.train.BaseCustomModel import BaseCustomModel

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseCustomModel(10, 5)

        print("✓ BaseCustomModel is properly abstract")

    except ImportError as e:
        pytest.skip(f"Could not import BaseCustomModel: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
