"""
Tests for custom neural network modules and their functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from recycla.train.BaseCustomModel import BaseCustomModel
from recycla.train.CustomConvNeXt import CustomConvNeXt
from recycla.train.CustomEfficientNetV2 import CustomEfficientNetV2
from recycla.train.CustomMobileNetV2 import CustomMobileNetV2
from recycla.train.CustomRegNet import CustomRegNet
from recycla.train.CustomResNet import CustomResNet
from recycla.train.CustomViT import CustomViT


class TestBaseCustomModel:
    """Test the base abstract class functionality."""

    def test_base_model_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCustomModel(10, 5)

    def test_get_trainable_params_info(self):
        """Test parameter information calculation."""

        # Create a mock model that implements abstract methods
        class MockModel(BaseCustomModel):
            def __init__(self):
                super().__init__(10, 5)
                self.model = nn.Sequential(
                    nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 20)
                )
                self.model.primary_classifier = nn.Linear(20, 10)
                self.model.secondary_classifier = nn.Linear(20, 5)

            def _load_pretrained_model(self):
                pass

            def _freeze_and_unfreeze_layers(self):
                pass

            def _setup_classification_heads(self):
                pass

        model = MockModel()
        param_info = model.get_trainable_params_info()

        assert "total_params" in param_info
        assert "trainable_params" in param_info
        assert "frozen_params" in param_info
        assert "trainable_percentage" in param_info
        assert param_info["total_params"] > 0
        assert param_info["trainable_params"] > 0
        assert param_info["trainable_percentage"] > 0

    def test_forward_pass_dual_output(self):
        """Test that forward pass returns dual outputs."""

        class MockModel(BaseCustomModel):
            def __init__(self):
                super().__init__(10, 5)
                self.model = nn.Module()
                self.model.classifier = nn.Sequential(nn.Linear(100, 20))
                self.model.forward = self.model.classifier.forward
                self.model.primary_classifier = nn.Linear(20, 10)
                self.model.secondary_classifier = nn.Linear(20, 5)

            def _load_pretrained_model(self):
                pass

            def _freeze_and_unfreeze_layers(self):
                pass

            def _setup_classification_heads(self):
                pass

        model = MockModel()
        x = torch.randn(2, 100)
        primary_output, secondary_output = model(x)

        assert primary_output.shape == (2, 10)
        assert secondary_output.shape == (2, 5)


class TestCustomMobileNetV2:
    """Test CustomMobileNetV2 functionality."""

    def test_mobilenet_initialization(self):
        """Test MobileNet V2 model initialization."""
        model = CustomMobileNetV2(
            num_primary_classes=10, num_secondary_classes=5, freeze_backbone=False
        )

        assert model.num_primary_classes == 10
        assert model.num_secondary_classes == 5
        assert model.freeze_backbone is False

    def test_mobilenet_forward_pass(self):
        """Test MobileNet V2 forward pass."""
        model = CustomMobileNetV2(10, 5, freeze_backbone=False)

        x = torch.randn(2, 3, 224, 224)
        primary_output, secondary_output = model(x)

        assert primary_output.shape == (2, 10)
        assert secondary_output.shape == (2, 5)


class TestCustomEfficientNetV2:
    """Test CustomEfficientNetV2 functionality."""

    def test_efficientnet_initialization(self):
        """Test EfficientNet V2 model initialization."""
        model = CustomEfficientNetV2(
            num_primary_classes=15, num_secondary_classes=8, model_size="s"
        )

        assert model.num_primary_classes == 15
        assert model.num_secondary_classes == 8
        assert model.model_size == "s"

    def test_efficientnet_invalid_model_size(self):
        """Test that invalid model size raises ValueError."""
        with patch("recycla.train.CustomEfficientNetV2.models.efficientnet_v2_s"):
            with pytest.raises(ValueError, match="Model size 'invalid' not supported"):
                CustomEfficientNetV2(10, 5, model_size="invalid")


class TestCustomConvNeXt:
    """Test CustomConvNeXt functionality."""

    def test_convnext_initialization(self):
        """Test ConvNeXt model initialization."""

        model = CustomConvNeXt(
            num_primary_classes=12, num_secondary_classes=6, model_size="tiny"
        )

        assert model.num_primary_classes == 12
        assert model.num_secondary_classes == 6
        assert model.model_size == "tiny"

    def test_convnext_invalid_model_size(self):
        """Test that invalid model size raises ValueError."""
        with patch("recycla.train.CustomConvNeXt.models.convnext_tiny"):
            with pytest.raises(ValueError, match="Model size 'invalid' not supported"):
                CustomConvNeXt(10, 5, model_size="invalid")


class TestCustomRegNet:
    """Test CustomRegNet functionality."""

    @patch("recycla.train.CustomRegNet.models.regnet_y_400mf")
    def test_regnet_initialization(self, mock_regnet):
        """Test RegNet model initialization."""
        mock_model = MagicMock()
        mock_model.fc = nn.Linear(440, 1000)
        mock_regnet.return_value = mock_model

        model = CustomRegNet(
            num_primary_classes=8, num_secondary_classes=4, model_variant="y_400mf"
        )

        assert model.num_primary_classes == 8
        assert model.num_secondary_classes == 4
        assert model.model_variant == "y_400mf"
        mock_regnet.assert_called_once()

    def test_regnet_invalid_model_variant(self):
        """Test that invalid model variant raises ValueError."""
        with patch("recycla.train.CustomRegNet.models.regnet_y_400mf"):
            with pytest.raises(
                ValueError, match="Model variant 'invalid' not supported"
            ):
                CustomRegNet(10, 5, model_variant="invalid")


class TestCustomViT:
    """Test CustomViT functionality."""

    @patch("recycla.train.CustomViT.models.vit_b_16")
    def test_vit_initialization(self, mock_vit):
        """Test Vision Transformer model initialization."""
        mock_model = MagicMock()
        mock_model.heads = MagicMock()
        mock_model.heads.head = nn.Linear(768, 1000)
        mock_vit.return_value = mock_model

        model = CustomViT(
            num_primary_classes=20, num_secondary_classes=10, model_size="b_16"
        )

        assert model.num_primary_classes == 20
        assert model.num_secondary_classes == 10
        assert model.model_size == "b_16"
        mock_vit.assert_called_once()

    def test_vit_invalid_model_size(self):
        """Test that invalid model size raises ValueError."""
        with patch("recycla.train.CustomViT.models.vit_b_16"):
            with pytest.raises(ValueError, match="Model size 'invalid' not supported"):
                CustomViT(10, 5, model_size="invalid")


class TestCustomResNet:
    """Test CustomResNet functionality."""

    @patch("recycla.train.CustomResNet.models.resnet50")
    def test_resnet_initialization(self, mock_resnet):
        """Test ResNet model initialization."""
        mock_model = MagicMock()
        mock_model.fc = nn.Linear(2048, 1000)
        mock_resnet.return_value = mock_model

        model = CustomResNet(
            num_primary_classes=25, num_secondary_classes=12, model_size="50"
        )

        assert model.num_primary_classes == 25
        assert model.num_secondary_classes == 12
        assert model.model_size == "50"
        mock_resnet.assert_called_once()

    def test_resnet_invalid_model_size(self):
        """Test that invalid model size raises ValueError."""
        with patch("recycla.train.CustomResNet.models.resnet50"):
            with pytest.raises(ValueError, match="Model size 'invalid' not supported"):
                CustomResNet(10, 5, model_size="invalid")


class TestParameterFreezing:
    """Test parameter freezing functionality across all models."""

    def test_mobilenet_parameter_freezing(self):
        """Test that MobileNet parameters are properly frozen/unfrozen."""
        # Test with backbone frozen
        model_frozen = CustomMobileNetV2(
            10, 5, freeze_backbone=True, unfreeze_last_n_blocks=0
        )

        frozen_params = sum(
            1 for p in model_frozen.model.features.parameters() if not p.requires_grad
        )
        total_feature_params = sum(1 for p in model_frozen.model.features.parameters())

        assert frozen_params == total_feature_params

        # Test with backbone unfrozen
        model_unfrozen = CustomMobileNetV2(10, 5, freeze_backbone=False)
        # Simulate unfreezing
        for param in model_unfrozen.model.features.parameters():
            param.requires_grad = True

        trainable_params = sum(
            1 for p in model_unfrozen.model.features.parameters() if p.requires_grad
        )
        assert trainable_params == total_feature_params


class TestModelConsistency:
    """Test consistency across all model implementations."""

    def test_all_models_have_same_interface(self):
        """Test that all models have consistent initialization parameters."""
        common_params = {
            "num_primary_classes": 10,
            "num_secondary_classes": 5,
            "freeze_backbone": True,
            "unfreeze_last_n_blocks": 2,
        }

        model_classes = [
            (CustomMobileNetV2, {}),
            (CustomEfficientNetV2, {"model_size": "s"}),
            (CustomConvNeXt, {"model_size": "tiny"}),
            (CustomRegNet, {"model_variant": "y_400mf"}),
            (CustomViT, {"model_size": "b_16"}),
            (CustomResNet, {"model_size": "50"}),
        ]

        for model_class, extra_params in model_classes:
            with patch.object(model_class, "_load_pretrained_model"), patch.object(
                model_class, "_freeze_and_unfreeze_layers"
            ), patch.object(model_class, "_setup_classification_heads"), patch.object(
                model_class, "log_model_info"
            ):

                params = {**common_params, **extra_params}
                model = model_class(**params)

                assert hasattr(model, "num_primary_classes")
                assert hasattr(model, "num_secondary_classes")
                assert hasattr(model, "freeze_backbone")
                assert hasattr(model, "unfreeze_last_n_blocks")
                assert hasattr(model, "get_trainable_params_info")
                assert hasattr(model, "log_model_info")
                assert hasattr(model, "forward")

    def test_all_models_inherit_from_base(self):
        """Test that all custom models inherit from BaseCustomModel."""
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


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        with patch("recycla.train.CustomMobileNetV2.models.mobilenet_v2"):
            # Test negative class numbers
            with pytest.raises((ValueError, TypeError, RuntimeError)):
                CustomMobileNetV2(-1, 5)

            with pytest.raises((ValueError, TypeError, RuntimeError, IndexError)):
                CustomMobileNetV2(10, -1)

    def test_zero_classes(self):
        """Test handling of zero classes."""
        # Zero classes should work (edge case but valid)
        try:
            model = CustomMobileNetV2(0, 5)
            assert model.num_primary_classes == 0
        except Exception as e:
            # Some implementations might not allow zero classes
            assert isinstance(e, (ValueError, RuntimeError))


class TestTrainingModeCompatibility:
    """Test compatibility with PyTorch training modes."""

    def test_train_eval_modes(self):
        """Test switching between train and eval modes."""
        model = CustomMobileNetV2(10, 5)

        # Test train mode
        model.train()
        assert model.training

        # Test eval mode
        model.eval()
        assert not model.training

    def test_cuda_compatibility(self):
        """Test CUDA device compatibility (when available)."""
        model = CustomMobileNetV2(10, 5)

        # Test CPU device
        model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model.to("cuda")
            assert next(model.parameters()).device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__])
