"""
Simplified tests for custom neural network modules.
These tests focus on the core functionality without loading actual pretrained weights.
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


@pytest.fixture
def mock_mobilenet_model():
    """Create a mock MobileNet model."""
    mock_model = MagicMock()
    # Create a realistic features structure
    mock_model.features = nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1),
        nn.BatchNorm2d(32),
        nn.ReLU6(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    # Create a realistic classifier structure
    mock_model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1000))

    # Mock the named_modules method
    def mock_named_modules():
        yield "features.0", mock_model.features[0]
        yield "features.1", mock_model.features[1]
        yield "features.2", mock_model.features[2]
        yield "features.3", mock_model.features[3]
        yield "classifier.0", mock_model.classifier[0]
        yield "classifier.1", mock_model.classifier[1]

    mock_model.named_modules = mock_named_modules
    return mock_model


@pytest.fixture
def mock_efficientnet_model():
    """Create a mock EfficientNet model."""
    mock_model = MagicMock()
    mock_model.features = nn.Sequential(
        nn.Conv2d(3, 24, 3, 2, 1), nn.BatchNorm2d(24), nn.AdaptiveAvgPool2d((1, 1))
    )
    mock_model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1000))
    return mock_model


class TestBaseCustomModelFunctionality:
    """Test the base model abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseCustomModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCustomModel(10, 5)

    def test_base_model_methods_exist(self):
        """Test that the base model has all required methods."""

        # Create a concrete implementation for testing
        class ConcreteModel(BaseCustomModel):
            def _load_pretrained_model(self):
                self.model = nn.Sequential(nn.Linear(10, 5))
                self.model.primary_classifier = nn.Linear(5, self.num_primary_classes)
                self.model.secondary_classifier = nn.Linear(
                    5, self.num_secondary_classes
                )

            def _freeze_and_unfreeze_layers(self):
                pass

            def _setup_classification_heads(self):
                pass

        model = ConcreteModel(10, 5)

        # Test that required methods exist
        assert hasattr(model, "get_trainable_params_info")
        assert hasattr(model, "log_model_info")
        assert hasattr(model, "forward")
        assert callable(model.get_trainable_params_info)
        assert callable(model.log_model_info)
        assert callable(model.forward)

    def test_parameter_info_calculation(self):
        """Test parameter information calculation."""

        class TestModel(BaseCustomModel):
            def __init__(
                self,
                num_primary_classes,
                num_secondary_classes,
                freeze_backbone=True,
                unfreeze_last_n_blocks=2,
            ):
                super(TestModel, self).__init__(
                    num_primary_classes=num_primary_classes,
                    num_secondary_classes=num_secondary_classes,
                    freeze_backbone=freeze_backbone,
                    unfreeze_last_n_blocks=unfreeze_last_n_blocks,
                )

                self._load_pretrained_model()
                self._freeze_and_unfreeze_layers()
                self._setup_classification_heads()

            def _load_pretrained_model(self):
                self.model = nn.Sequential(nn.Linear(100, 50))
                self.model.primary_classifier = nn.Linear(50, self.num_primary_classes)
                self.model.secondary_classifier = nn.Linear(
                    50, self.num_secondary_classes
                )

            def _freeze_and_unfreeze_layers(self):
                # Freeze the main model
                for param in self.model[0].parameters():
                    param.requires_grad = False

            def _setup_classification_heads(self):
                pass

        model = TestModel(10, 5)
        param_info = model.get_trainable_params_info()

        assert isinstance(param_info, dict)
        assert "total_params" in param_info
        assert "trainable_params" in param_info
        assert "frozen_params" in param_info
        assert "trainable_percentage" in param_info
        assert param_info["total_params"] > 0


class TestMobileNetV2:
    """Test CustomMobileNetV2 without loading actual weights."""

    @patch("recycla.train.CustomMobileNetV2.models.mobilenet_v2")
    def test_mobilenet_basic_creation(self, mock_mobilenet, mock_mobilenet_model):
        """Test basic MobileNet creation and properties."""
        mock_mobilenet.return_value = mock_mobilenet_model

        model = CustomMobileNetV2(
            num_primary_classes=10, num_secondary_classes=5, freeze_backbone=False
        )

        assert model.num_primary_classes == 10
        assert model.num_secondary_classes == 5
        assert model.freeze_backbone is False
        mock_mobilenet.assert_called_once()

    @patch("recycla.train.CustomMobileNetV2.models.mobilenet_v2")
    def test_mobilenet_inheritance(self, mock_mobilenet, mock_mobilenet_model):
        """Test that MobileNet inherits from BaseCustomModel."""
        mock_mobilenet.return_value = mock_mobilenet_model

        model = CustomMobileNetV2(10, 5)

        assert isinstance(model, BaseCustomModel)
        assert isinstance(model, nn.Module)
        assert hasattr(model, "get_trainable_params_info")
        assert hasattr(model, "forward")


class TestEfficientNetV2:
    """Test CustomEfficientNetV2."""

    @patch("recycla.train.CustomEfficientNetV2.models.efficientnet_v2_s")
    def test_efficientnet_creation(self, mock_efficientnet, mock_efficientnet_model):
        """Test EfficientNet creation."""
        mock_efficientnet.return_value = mock_efficientnet_model

        model = CustomEfficientNetV2(
            num_primary_classes=15, num_secondary_classes=8, model_size="s"
        )

        assert model.num_primary_classes == 15
        assert model.num_secondary_classes == 8
        assert model.model_size == "s"

    @patch("recycla.train.CustomEfficientNetV2.models.efficientnet_v2_s")
    def test_efficientnet_invalid_size(self, mock_efficientnet):
        """Test handling of invalid model size."""
        with pytest.raises(ValueError, match="Model size 'invalid' not supported"):
            CustomEfficientNetV2(10, 5, model_size="invalid")


class TestConvNeXt:
    """Test CustomConvNeXt."""

    def test_convnext_creation(self):
        """Test ConvNeXt creation."""
        model = CustomConvNeXt(
            num_primary_classes=12, num_secondary_classes=6, model_size="tiny"
        )

        assert model.num_primary_classes == 12
        assert model.num_secondary_classes == 6
        assert model.model_size == "tiny"


class TestRegNet:
    """Test CustomRegNet."""

    @patch("recycla.train.CustomRegNet.models.regnet_y_400mf")
    def test_regnet_creation(self, mock_regnet):
        """Test RegNet creation."""
        mock_model = MagicMock()
        mock_model.fc = nn.Linear(440, 1000)
        mock_model.trunk = MagicMock()
        mock_regnet.return_value = mock_model

        model = CustomRegNet(
            num_primary_classes=8, num_secondary_classes=4, model_variant="y_400mf"
        )

        assert model.num_primary_classes == 8
        assert model.num_secondary_classes == 4
        assert model.model_variant == "y_400mf"


class TestViT:
    """Test CustomViT."""

    @patch("recycla.train.CustomViT.models.vit_b_16")
    def test_vit_creation(self, mock_vit):
        """Test Vision Transformer creation."""
        mock_model = MagicMock()
        mock_model.heads = MagicMock()
        mock_model.heads.head = nn.Linear(768, 1000)
        mock_model.conv_proj = nn.Conv2d(3, 768, 16, 16)
        mock_model.encoder = MagicMock()
        mock_vit.return_value = mock_model

        model = CustomViT(
            num_primary_classes=20, num_secondary_classes=10, model_size="b_16"
        )

        assert model.num_primary_classes == 20
        assert model.num_secondary_classes == 10
        assert model.model_size == "b_16"


class TestResNet:
    """Test CustomResNet."""

    @patch("recycla.train.CustomResNet.models.resnet50")
    def test_resnet_creation(self, mock_resnet):
        """Test ResNet creation."""
        mock_model = MagicMock()
        mock_model.fc = nn.Linear(2048, 1000)
        mock_model.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3))
        mock_model.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3))
        mock_model.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3))
        mock_model.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3))
        mock_resnet.return_value = mock_model

        model = CustomResNet(
            num_primary_classes=25, num_secondary_classes=12, model_size="50"
        )

        assert model.num_primary_classes == 25
        assert model.num_secondary_classes == 12
        assert model.model_size == "50"


class TestModelConsistency:
    """Test consistency across all model implementations."""

    def test_all_models_inherit_from_base(self):
        """Test that all models inherit from BaseCustomModel."""
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

    def test_common_interface(self):
        """Test that all models have common interface methods."""
        # Use a simple mock approach to avoid complex patching
        common_methods = ["get_trainable_params_info", "log_model_info", "forward"]

        model_classes = [
            CustomMobileNetV2,
            CustomEfficientNetV2,
            CustomConvNeXt,
            CustomRegNet,
            CustomViT,
            CustomResNet,
        ]

        for model_class in model_classes:
            for method_name in common_methods:
                assert hasattr(model_class, method_name)
                assert callable(getattr(model_class, method_name))


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_model_sizes(self):
        """Test that invalid model sizes/variants raise appropriate errors."""
        test_cases = [
            (CustomEfficientNetV2, {"model_size": "invalid"}),
            (CustomConvNeXt, {"model_size": "invalid"}),
            (CustomRegNet, {"model_variant": "invalid"}),
            (CustomViT, {"model_size": "invalid"}),
            (CustomResNet, {"model_size": "invalid"}),
        ]

        for model_class, kwargs in test_cases:
            with pytest.raises(ValueError):
                # Mock the models module to avoid loading weights
                with patch(f"{model_class.__module__}.models"):
                    model_class(10, 5, **kwargs)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test that models accept valid parameters
        model = CustomMobileNetV2(
            num_primary_classes=10,
            num_secondary_classes=5,
            freeze_backbone=True,
            unfreeze_last_n_blocks=2,
        )

        assert model.num_primary_classes == 10
        assert model.num_secondary_classes == 5
        assert model.freeze_backbone is True
        assert model.unfreeze_last_n_blocks == 2


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
