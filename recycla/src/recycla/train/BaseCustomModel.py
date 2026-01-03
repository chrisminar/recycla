from abc import ABC, abstractmethod

import torch.nn as nn

from recycla import log


class BaseCustomModel(nn.Module, ABC):
    """
    Base class for custom dual-classification models.
    Provides common functionality for freezing/unfreezing backbone layers
    and dual classification heads.
    """

    def __init__(
        self,
        num_primary_classes: int,
        num_secondary_classes: int,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 2,
    ):
        super(BaseCustomModel, self).__init__()
        self.num_primary_classes = num_primary_classes
        self.num_secondary_classes = num_secondary_classes
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks

        # Will be set by subclasses
        self.model = None

    @abstractmethod
    def _load_pretrained_model(self, **kwargs):
        """Load the pretrained model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _freeze_and_unfreeze_layers(self):
        """Implement backbone freezing logic specific to each architecture."""
        pass

    @abstractmethod
    def _setup_classification_heads(self):
        """Setup the dual classification heads. Must be implemented by subclasses."""
        pass

    def get_trainable_params_info(self):
        """Returns information about trainable parameters for debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Handle case where there are no parameters
        trainable_percentage = (
            (trainable_params / total_params) * 100 if total_params > 0 else 0
        )

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_percentage": trainable_percentage,
        }

    def log_model_info(self, model_name: str):
        """Log model information including trainable parameters."""
        param_info = self.get_trainable_params_info()
        log.info(f"Using {model_name}")
        log.info(
            f"Model parameters: {param_info['trainable_params']:,} trainable / {param_info['total_params']:,} total ({param_info['trainable_percentage']:.1f}% trainable)"
        )

    def forward(self, x):
        """Forward pass through the model with dual classification heads."""
        features = self.model(x)
        primary_output = self.model.primary_classifier(features)
        secondary_output = self.model.secondary_classifier(features)
        return primary_output, secondary_output
