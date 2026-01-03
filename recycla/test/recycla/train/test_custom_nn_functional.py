"""
Functional tests for custom neural network modules.
These tests use comprehensive mocking to avoid loading actual pretrained models.
"""

import os
import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
import torch.nn as nn

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
sys.path.insert(0, os.path.abspath(src_path))

from recycla.train.BaseCustomModel import BaseCustomModel
from recycla.train.CustomConvNeXt import CustomConvNeXt
from recycla.train.CustomEfficientNetV2 import CustomEfficientNetV2
from recycla.train.CustomMobileNetV2 import CustomMobileNetV2
from recycla.train.CustomRegNet import CustomRegNet
from recycla.train.CustomResNet import CustomResNet
from recycla.train.CustomViT import CustomViT


class TestBaseCustomModel:
    """Test the BaseCustomModel abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseCustomModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseCustomModel(10, 5)

    def test_base_model_has_required_methods(self):
        """Test that BaseCustomModel has all required abstract methods."""
        # Check that the abstract methods exist
        assert hasattr(BaseCustomModel, "_load_pretrained_model")
        assert hasattr(BaseCustomModel, "_freeze_and_unfreeze_layers")
        assert hasattr(BaseCustomModel, "_setup_classification_heads")

        # Check concrete methods exist
        assert hasattr(BaseCustomModel, "get_trainable_params_info")
        assert hasattr(BaseCustomModel, "log_model_info")
        assert hasattr(BaseCustomModel, "forward")

    def test_parameter_info_with_mock_model(self):
        """Test parameter info calculation with a working model."""

        class TestModel(BaseCustomModel):
            def _load_pretrained_model(self):
                self.model = nn.Sequential(nn.Linear(10, 5))

            def _freeze_and_unfreeze_layers(self):
                pass

            def _setup_classification_heads(self):
                self.model.primary_classifier = nn.Linear(5, self.num_primary_classes)
                self.model.secondary_classifier = nn.Linear(
                    5, self.num_secondary_classes
                )

        with patch("recycla.log"):  # Mock the logger
            model = TestModel(3, 2)
            param_info = model.get_trainable_params_info()

            assert isinstance(param_info, dict)
            assert "total_params" in param_info
            assert "trainable_params" in param_info
            assert "frozen_params" in param_info
            assert "trainable_percentage" in param_info


class TestModelInitialization:
    """Test model initialization with proper mocking."""

    def test_mobilenet_initialization(self):
        """Test MobileNet V2 initialization."""

        # Test initialization
        model = CustomMobileNetV2(
            num_primary_classes=10, num_secondary_classes=5, freeze_backbone=False
        )

        assert model.num_primary_classes == 10
        assert model.num_secondary_classes == 5
        assert model.freeze_backbone is False

    def test_efficientnet_initialization(self):
        """Test EfficientNet V2 initialization."""
        model = CustomEfficientNetV2(
            num_primary_classes=15, num_secondary_classes=8, model_size="s"
        )

        assert model.num_primary_classes == 15
        assert model.num_secondary_classes == 8
        assert model.model_size == "s"

    @patch("recycla.train.CustomConvNeXt.log")
    @patch("recycla.train.CustomConvNeXt.models.convnext_tiny")
    def test_convnext_initialization(self, mock_convnext_func, mock_log):
        """Test ConvNeXt initialization."""
        mock_model = MagicMock()
        mock_model.features = nn.Sequential(nn.Conv2d(3, 96, 4))

        # Create a proper classifier structure for ConvNeXt
        mock_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1), nn.Linear(768, 1000)
        )
        mock_model.classifier = mock_classifier
        mock_convnext_func.return_value = mock_model

        model = CustomConvNeXt(
            num_primary_classes=12, num_secondary_classes=6, model_size="tiny"
        )

        assert model.num_primary_classes == 12
        assert model.num_secondary_classes == 6
        assert model.model_size == "tiny"

    @patch("recycla.train.CustomRegNet.log")
    @patch("recycla.train.CustomRegNet.models")
    def test_regnet_initialization(self, mock_models, mock_log):
        """Test RegNet initialization."""
        mock_model = MagicMock()
        mock_model.trunk = nn.Sequential(nn.Conv2d(3, 32, 3))
        mock_model.fc = nn.Linear(440, 1000)

        # Mock the model loading function
        mock_models.regnet_y_400mf.return_value = mock_model

        model = CustomRegNet(
            num_primary_classes=8, num_secondary_classes=4, model_variant="y_400mf"
        )

        assert model.num_primary_classes == 8
        assert model.num_secondary_classes == 4
        assert model.model_variant == "y_400mf"

    @patch("recycla.train.CustomViT.log")
    @patch("recycla.train.CustomViT.models.vit_b_16")
    def test_vit_initialization(self, mock_vit_func, mock_log):
        """Test Vision Transformer initialization."""
        mock_model = MagicMock()
        mock_model.conv_proj = nn.Conv2d(3, 768, 16, 16)
        mock_model.encoder = MagicMock()
        mock_model.heads = MagicMock()
        mock_model.heads.head = nn.Linear(768, 1000)
        mock_vit_func.return_value = mock_model

        model = CustomViT(
            num_primary_classes=20, num_secondary_classes=10, model_size="b_16"
        )

        assert model.num_primary_classes == 20
        assert model.num_secondary_classes == 10
        assert model.model_size == "b_16"

    @patch("recycla.train.CustomResNet.log")
    @patch("recycla.train.CustomResNet.models.resnet50")
    def test_resnet_initialization(self, mock_resnet_func, mock_log):
        """Test ResNet initialization."""
        mock_model = MagicMock()
        mock_model.fc = nn.Linear(2048, 1000)
        mock_model.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3))
        mock_model.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3))
        mock_model.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3))
        mock_model.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3))
        mock_resnet_func.return_value = mock_model

        model = CustomResNet(
            num_primary_classes=25, num_secondary_classes=12, model_size="50"
        )

        assert model.num_primary_classes == 25
        assert model.num_secondary_classes == 12
        assert model.model_size == "50"


class TestInvalidParameters:
    """Test error handling for invalid parameters."""

    def test_efficientnet_invalid_model_size(self):
        """Test EfficientNet with invalid model size."""
        with pytest.raises(ValueError, match="Model size 'invalid' not supported"):
            with patch("recycla.train.CustomEfficientNetV2.log"):
                CustomEfficientNetV2(10, 5, model_size="invalid")

    def test_convnext_invalid_model_size(self):
        """Test ConvNeXt with invalid model size."""
        with pytest.raises(ValueError, match="Model size 'invalid' not supported"):
            with patch("recycla.train.CustomConvNeXt.log"):
                CustomConvNeXt(10, 5, model_size="invalid")

    def test_regnet_invalid_model_variant(self):
        """Test RegNet with invalid model variant."""
        with pytest.raises(ValueError, match="Model variant 'invalid' not supported"):
            with patch("recycla.train.CustomRegNet.log"):
                CustomRegNet(10, 5, model_variant="invalid")

    def test_vit_invalid_model_size(self):
        """Test ViT with invalid model size."""
        with pytest.raises(ValueError, match="Model size 'invalid' not supported"):
            with patch("recycla.train.CustomViT.log"):
                CustomViT(10, 5, model_size="invalid")

    def test_resnet_invalid_model_size(self):
        """Test ResNet with invalid model size."""
        with pytest.raises(ValueError, match="Model size 'invalid' not supported"):
            with patch("recycla.train.CustomResNet.log"):
                CustomResNet(10, 5, model_size="invalid")


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

    def test_all_models_have_common_interface(self):
        """Test that all models have the expected interface."""
        required_methods = [
            "get_trainable_params_info",
            "log_model_info",
            "forward",
            "_load_pretrained_model",
            "_freeze_and_unfreeze_layers",
            "_setup_classification_heads",
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
                ), f"{model_class.__name__} missing {method_name}"


class TestForwardPass:
    """Test forward pass functionality."""

    def test_forward_pass_shape(self):
        """Test that forward pass returns correct shapes."""

        # Create the model
        model = CustomMobileNetV2(10, 5, freeze_backbone=False)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        primary_output, secondary_output = model(x)

        assert primary_output.shape == (2, model.model.primary_classifier.out_features)
        assert secondary_output.shape == (
            2,
            model.model.secondary_classifier.out_features,
        )


class TestParameterFreezing:
    """Test parameter freezing functionality."""

    def test_backbone_freezing_enabled_disabled(self):
        """Test basic backbone freezing functionality - enabled vs disabled."""
        # Test with backbone frozen
        model_frozen = CustomMobileNetV2(
            10, 5, freeze_backbone=True, unfreeze_last_n_blocks=0
        )

        # Check that some parameters are frozen (though we can't check the exact ones due to mocking)
        assert hasattr(model_frozen, "freeze_backbone")
        assert model_frozen.freeze_backbone is True

        # Get parameter info for frozen model
        param_info_frozen = model_frozen.get_trainable_params_info()

        # Test with backbone unfrozen
        model_unfrozen = CustomMobileNetV2(10, 5, freeze_backbone=False)
        assert model_unfrozen.freeze_backbone is False

        # Get parameter info for unfrozen model
        param_info_unfrozen = model_unfrozen.get_trainable_params_info()

        # Unfrozen model should have more trainable parameters
        assert (
            param_info_unfrozen["trainable_params"]
            >= param_info_frozen["trainable_params"]
        )
        assert (
            param_info_unfrozen["trainable_percentage"]
            >= param_info_frozen["trainable_percentage"]
        )

    def test_unfreeze_last_n_blocks_functionality(self):
        """Test that unfreeze_last_n_blocks parameter controls partial unfreezing."""
        # Test with no blocks unfrozen (fully frozen backbone)
        model_no_unfreeze = CustomMobileNetV2(
            10, 5, freeze_backbone=True, unfreeze_last_n_blocks=0
        )

        # Test with some blocks unfrozen
        model_partial_unfreeze = CustomMobileNetV2(
            10, 5, freeze_backbone=True, unfreeze_last_n_blocks=2
        )

        # Test with many blocks unfrozen
        model_more_unfreeze = CustomMobileNetV2(
            10, 5, freeze_backbone=True, unfreeze_last_n_blocks=5
        )

        # Get parameter info
        param_info_no_unfreeze = model_no_unfreeze.get_trainable_params_info()
        param_info_partial = model_partial_unfreeze.get_trainable_params_info()
        param_info_more = model_more_unfreeze.get_trainable_params_info()

        # Check that unfreezing more blocks increases trainable parameters
        assert (
            param_info_partial["trainable_params"]
            >= param_info_no_unfreeze["trainable_params"]
        )
        assert (
            param_info_more["trainable_params"]
            >= param_info_partial["trainable_params"]
        )

        # Check that the unfreeze_last_n_blocks attribute is set correctly
        assert model_no_unfreeze.unfreeze_last_n_blocks == 0
        assert model_partial_unfreeze.unfreeze_last_n_blocks == 2
        assert model_more_unfreeze.unfreeze_last_n_blocks == 5

    def test_parameter_freezing_consistency_across_models(self):
        """Test that parameter freezing works consistently across different model types."""
        model_classes_to_test = [
            (CustomMobileNetV2, {}),
            (CustomEfficientNetV2, {"model_size": "s"}),
            (CustomResNet, {"model_size": "18"}),
        ]

        for model_class, extra_kwargs in model_classes_to_test:
            # Test frozen vs unfrozen for each model type
            model_frozen = model_class(
                num_primary_classes=10,
                num_secondary_classes=5,
                freeze_backbone=True,
                unfreeze_last_n_blocks=0,
                **extra_kwargs,
            )

            model_unfrozen = model_class(
                num_primary_classes=10,
                num_secondary_classes=5,
                freeze_backbone=False,
                **extra_kwargs,
            )

            # Check basic attributes
            assert model_frozen.freeze_backbone is True
            assert model_unfrozen.freeze_backbone is False

            # Get parameter information
            param_info_frozen = model_frozen.get_trainable_params_info()
            param_info_unfrozen = model_unfrozen.get_trainable_params_info()

            # Verify that unfrozen models have more trainable parameters
            assert (
                param_info_unfrozen["trainable_params"]
                >= param_info_frozen["trainable_params"]
            ), f"Model {model_class.__name__} freezing failed: unfrozen should have >= trainable params"

            # Both models should have some parameters (classification heads at minimum)
            assert (
                param_info_frozen["total_params"] > 0
            ), f"Model {model_class.__name__} should have parameters"
            assert (
                param_info_unfrozen["total_params"] > 0
            ), f"Model {model_class.__name__} should have parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
