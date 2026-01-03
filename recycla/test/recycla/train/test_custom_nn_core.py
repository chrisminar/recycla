"""
Core functionality tests for custom neural network modules.
These tests focus on the essential functionality without complex model instantiation.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
sys.path.insert(0, os.path.abspath(src_path))

try:
    from recycla.train.BaseCustomModel import BaseCustomModel
    from recycla.train.CustomConvNeXt import CustomConvNeXt
    from recycla.train.CustomEfficientNetV2 import CustomEfficientNetV2
    from recycla.train.CustomMobileNetV2 import CustomMobileNetV2
    from recycla.train.CustomRegNet import CustomRegNet
    from recycla.train.CustomResNet import CustomResNet
    from recycla.train.CustomViT import CustomViT
except ImportError as e:
    pytest.skip(f"Could not import modules: {e}", allow_module_level=True)


class MockModelWithClassifiers(nn.Module):
    """A mock model that properly implements the expected structure."""

    def __init__(self, feature_dim=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, feature_dim)
        )
        self.primary_classifier = None
        self.secondary_classifier = None

    def forward(self, x):
        """Forward pass through feature extractor."""
        return self.features(x)


class MockModel(BaseCustomModel):
    """A concrete implementation of BaseCustomModel for testing."""

    def __init__(
        self,
        num_primary_classes,
        num_secondary_classes,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
    ):
        super().__init__(
            num_primary_classes,
            num_secondary_classes,
            freeze_backbone,
            unfreeze_last_n_blocks,
        )
        self._load_pretrained_model()
        self._freeze_and_unfreeze_layers()
        self._setup_classification_heads()

    def _load_pretrained_model(self):
        """Create a simple mock model."""
        self.model = MockModelWithClassifiers(feature_dim=20)

    def _freeze_and_unfreeze_layers(self):
        """Implement simple freezing logic."""
        if self.freeze_backbone:
            # Freeze the first layer of the feature extractor
            for param in self.model.features[0].parameters():
                param.requires_grad = False

    def _setup_classification_heads(self):
        """Setup classification heads."""
        self.model.primary_classifier = nn.Linear(20, self.num_primary_classes)
        self.model.secondary_classifier = nn.Linear(20, self.num_secondary_classes)

    def forward(self, x):
        """Custom forward method that handles tensor shapes correctly."""
        # The BaseCustomModel's forward pass is used for dual classification
        return super().forward(x)


class TestBaseCustomModel:
    """Test the BaseCustomModel abstract class functionality."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseCustomModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseCustomModel(10, 5)

    def test_abstract_methods_exist(self):
        """Test that abstract methods are defined."""
        abstract_methods = [
            "_load_pretrained_model",
            "_freeze_and_unfreeze_layers",
            "_setup_classification_heads",
        ]

        for method_name in abstract_methods:
            assert hasattr(BaseCustomModel, method_name)

    def test_concrete_methods_exist(self):
        """Test that concrete methods are defined."""
        concrete_methods = ["get_trainable_params_info", "log_model_info", "forward"]

        for method_name in concrete_methods:
            assert hasattr(BaseCustomModel, method_name)
            assert callable(getattr(BaseCustomModel, method_name))

    def test_mock_model_creation(self):
        """Test creating a mock model that inherits from BaseCustomModel."""
        with patch("recycla.log"):
            model = MockModel(10, 5)

            assert model.num_primary_classes == 10
            assert model.num_secondary_classes == 5
            assert model.freeze_backbone is True
            assert model.unfreeze_last_n_blocks == 2
            assert hasattr(model, "model")

    def test_parameter_info_calculation(self):
        """Test parameter information calculation."""
        with patch("recycla.log"):
            model = MockModel(3, 2)
            param_info = model.get_trainable_params_info()

            assert isinstance(param_info, dict)
            assert "total_params" in param_info
            assert "trainable_params" in param_info
            assert "frozen_params" in param_info
            assert "trainable_percentage" in param_info
            assert param_info["total_params"] > 0
            assert param_info["trainable_percentage"] >= 0

    def test_forward_pass(self):
        """Test forward pass functionality."""
        with patch("recycla.log"):
            model = MockModel(4, 3)

            # Test forward pass with correct input size (100 to match the first Linear layer)
            x = torch.randn(2, 100)  # Match the input size of the model
            primary_output, secondary_output = model(x)

            assert primary_output.shape == (2, 4)
            assert secondary_output.shape == (2, 3)
            assert isinstance(primary_output, torch.Tensor)
            assert isinstance(secondary_output, torch.Tensor)

    def test_parameter_freezing(self):
        """Test parameter freezing functionality."""
        with patch("recycla.log"):
            # Test with freezing
            model_frozen = MockModel(5, 3, freeze_backbone=True)
            param_info_frozen = model_frozen.get_trainable_params_info()

            # Test without freezing
            model_unfrozen = MockModel(5, 3, freeze_backbone=False)
            param_info_unfrozen = model_unfrozen.get_trainable_params_info()

            # Frozen model should have fewer trainable parameters
            assert (
                param_info_frozen["trainable_params"]
                < param_info_unfrozen["trainable_params"]
            )
            assert (
                param_info_frozen["frozen_params"]
                > param_info_unfrozen["frozen_params"]
            )

    def test_log_model_info(self):
        """Test model info logging."""
        with patch("recycla.train.BaseCustomModel.log") as mock_log:
            model = MockModel(2, 1)
            model.log_model_info("Test Model")

            # Check that log.info was called (the actual calls happen in BaseCustomModel)
            assert mock_log.info.call_count >= 1
            call_args = [call[0][0] for call in mock_log.info.call_args_list]

            # Should log the model name and parameter info
            assert any("Test Model" in arg for arg in call_args)


class TestModelClassInheritance:
    """Test that all model classes properly inherit from BaseCustomModel."""

    def test_all_models_inherit_from_base(self):
        """Test inheritance hierarchy."""
        model_classes = [
            CustomMobileNetV2,
            CustomEfficientNetV2,
            CustomConvNeXt,
            CustomRegNet,
            CustomViT,
            CustomResNet,
        ]

        for model_class in model_classes:
            assert issubclass(model_class, BaseCustomModel)
            assert issubclass(model_class, nn.Module)

    def test_all_models_have_required_methods(self):
        """Test that all models implement required methods."""
        required_methods = [
            "_load_pretrained_model",
            "_freeze_and_unfreeze_layers",
            "_setup_classification_heads",
            "get_trainable_params_info",
            "log_model_info",
            "forward",
        ]

        model_classes = [
            CustomMobileNetV2,
            CustomEfficientNetV2,
            CustomConvNeXt,
            CustomRegNet,
            CustomViT,
            CustomResNet,
        ]

        for model_class in model_classes:
            for method_name in required_methods:
                assert hasattr(
                    model_class, method_name
                ), f"{model_class.__name__} missing method {method_name}"


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_invalid_class_numbers(self):
        """Test handling of invalid class numbers."""
        with patch("recycla.log"):
            # Test zero classes (should work)
            model = MockModel(0, 1)
            assert model.num_primary_classes == 0

            model = MockModel(1, 0)
            assert model.num_secondary_classes == 0

    def test_boolean_parameters(self):
        """Test boolean parameter handling."""
        with patch("recycla.log"):
            # Test freeze_backbone parameter
            model_true = MockModel(2, 2, freeze_backbone=True)
            assert model_true.freeze_backbone is True

            model_false = MockModel(2, 2, freeze_backbone=False)
            assert model_false.freeze_backbone is False

    def test_unfreeze_blocks_parameter(self):
        """Test unfreeze_last_n_blocks parameter."""
        with patch("recycla.log"):
            model = MockModel(2, 2, unfreeze_last_n_blocks=5)
            assert model.unfreeze_last_n_blocks == 5

            model = MockModel(2, 2, unfreeze_last_n_blocks=0)
            assert model.unfreeze_last_n_blocks == 0


class TestModelFunctionality:
    """Test core model functionality."""

    def test_model_training_mode(self):
        """Test switching between training and evaluation modes."""
        with patch("recycla.log"):
            model = MockModel(3, 2)

            # Test training mode
            model.train()
            assert model.training is True

            # Test evaluation mode
            model.eval()
            assert model.training is False

    def test_model_device_compatibility(self):
        """Test device compatibility."""
        with patch("recycla.log"):
            model = MockModel(2, 1)

            # Test CPU
            model.to("cpu")
            assert next(model.parameters()).device.type == "cpu"

            # Test CUDA if available
            if torch.cuda.is_available():
                model.to("cuda")
                assert next(model.parameters()).device.type == "cuda"

    def test_gradient_computation(self):
        """Test gradient computation."""
        with patch("recycla.log"):
            model = MockModel(2, 1)
            model.train()

            x = torch.randn(1, 100, requires_grad=True)  # Match model input size (100)
            primary_output, secondary_output = model(x)

            # Test that outputs require gradients when in training mode
            assert primary_output.requires_grad is True
            assert secondary_output.requires_grad is True

            # Test backward pass
            loss = primary_output.sum() + secondary_output.sum()
            loss.backward()

            # Check that some parameters have gradients
            has_gradients = any(
                p.grad is not None for p in model.parameters() if p.requires_grad
            )
            assert has_gradients


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
