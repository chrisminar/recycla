import torch.nn as nn
from torchvision import models

from recycla import log
from recycla.train.BaseCustomModel import BaseCustomModel


class CustomConvNeXt(BaseCustomModel):
    def __init__(
        self,
        num_primary_classes,
        num_secondary_classes,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        model_size="tiny",  # 'tiny', 'small', 'base', 'large'
    ):
        super(CustomConvNeXt, self).__init__(
            num_primary_classes,
            num_secondary_classes,
            freeze_backbone,
            unfreeze_last_n_blocks,
        )
        self.model_size = model_size

        self._load_pretrained_model()
        self._freeze_and_unfreeze_layers()
        self._setup_classification_heads()
        self.log_model_info(f"ConvNeXt-{model_size}")

    def _load_pretrained_model(self):
        """Load the pretrained ConvNeXt model."""
        # Select model size
        if self.model_size == "tiny":
            self.model = models.convnext_tiny(weights="IMAGENET1K_V1")
        elif self.model_size == "small":
            self.model = models.convnext_small(weights="IMAGENET1K_V1")
        elif self.model_size == "base":
            self.model = models.convnext_base(weights="IMAGENET1K_V1")
        elif self.model_size == "large":
            self.model = models.convnext_large(weights="IMAGENET1K_V1")
        else:
            raise ValueError(
                f"Model size '{self.model_size}' not supported. Use 'tiny', 'small', 'base', or 'large'"
            )

    def _freeze_and_unfreeze_layers(self):
        """Implement ConvNeXt specific freezing logic."""
        if self.freeze_backbone:
            # First freeze all features
            for param in self.model.features.parameters():
                param.requires_grad = False

            # Then unfreeze the last n blocks if specified
            if self.unfreeze_last_n_blocks > 0:
                # ConvNeXt structure: features contain 4 main stages
                # Each stage has multiple ConvNeXt blocks
                feature_stages = list(self.model.features.children())
                total_stages = len(feature_stages)
                stages_to_unfreeze = min(self.unfreeze_last_n_blocks, total_stages)

                # Unfreeze the last n feature stages
                for stage in feature_stages[-stages_to_unfreeze:]:
                    for param in stage.parameters():
                        param.requires_grad = True

                log.info(
                    f"Frozen ConvNeXt backbone features, unfroze last {stages_to_unfreeze} feature stages"
                )
            else:
                log.info("Frozen ConvNeXt backbone features")

    def _setup_classification_heads(self):
        """Setup dual classification heads for ConvNeXt."""
        # Get the number of features from the classifier
        num_ftrs = self.model.classifier[2].in_features

        # Keep the normalization and flattening layers, remove final linear layer
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # LayerNorm2d
            self.model.classifier[1],  # Flatten
            # Remove the final Linear layer (classifier[2])
        )

        # Add custom classification heads
        self.model.primary_classifier = nn.Linear(num_ftrs, self.num_primary_classes)
        self.model.secondary_classifier = nn.Linear(
            num_ftrs, self.num_secondary_classes
        )
