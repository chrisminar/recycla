import torch.nn as nn
from torchvision import models

from recycla import log
from recycla.train.BaseCustomModel import BaseCustomModel


class CustomEfficientNetV2(BaseCustomModel):
    def __init__(
        self,
        num_primary_classes,
        num_secondary_classes,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        model_size="s",  # 's', 'm', or 'l'
    ):
        super(CustomEfficientNetV2, self).__init__(
            num_primary_classes,
            num_secondary_classes,
            freeze_backbone,
            unfreeze_last_n_blocks,
        )
        self.model_size = model_size

        self._load_pretrained_model()
        self._freeze_and_unfreeze_layers()
        self._setup_classification_heads()
        self.log_model_info(f"EfficientNet V2-{model_size.upper()}")

    def _load_pretrained_model(self):
        """Load the pretrained EfficientNet V2 model."""
        # Select model size
        if self.model_size == "s":
            self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        elif self.model_size == "m":
            self.model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
        elif self.model_size == "l":
            self.model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
        else:
            raise ValueError(
                f"Model size '{self.model_size}' not supported. Use 's', 'm', or 'l'"
            )

    def _freeze_and_unfreeze_layers(self):
        """Implement EfficientNet V2 specific freezing logic."""
        if self.freeze_backbone:
            # First freeze all features
            for param in self.model.features.parameters():
                param.requires_grad = False

            # Then unfreeze the last n blocks if specified
            if self.unfreeze_last_n_blocks > 0:
                # EfficientNet V2 structure: features contain multiple MBConv blocks
                # Get all the main feature blocks (stages)
                feature_stages = list(self.model.features.children())
                total_stages = len(feature_stages)
                stages_to_unfreeze = min(self.unfreeze_last_n_blocks, total_stages)

                # Unfreeze the last n feature stages
                for stage in feature_stages[-stages_to_unfreeze:]:
                    for param in stage.parameters():
                        param.requires_grad = True

                log.info(
                    f"Frozen EfficientNet V2 backbone features, unfroze last {stages_to_unfreeze} feature stages"
                )
            else:
                log.info("Frozen EfficientNet V2 backbone features")

    def _setup_classification_heads(self):
        """Setup dual classification heads for EfficientNet V2."""
        # Get the number of features from the classifier
        num_ftrs = self.model.classifier[1].in_features

        # Keep only the dropout layer from original classifier
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # Keep the dropout layer
        )

        # Add custom classification heads
        self.model.primary_classifier = nn.Linear(num_ftrs, self.num_primary_classes)
        self.model.secondary_classifier = nn.Linear(
            num_ftrs, self.num_secondary_classes
        )
