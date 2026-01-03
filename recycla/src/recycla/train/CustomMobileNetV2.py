import torch.nn as nn
from torchvision import models

from recycla import log
from recycla.train.BaseCustomModel import BaseCustomModel


class CustomMobileNetV2(BaseCustomModel):
    def __init__(
        self,
        num_primary_classes,
        num_secondary_classes,
        freeze_backbone=True,
        unfreeze_last_n_blocks=12,
    ):
        super(CustomMobileNetV2, self).__init__(
            num_primary_classes,
            num_secondary_classes,
            freeze_backbone,
            unfreeze_last_n_blocks,
        )

        self._load_pretrained_model()
        self._freeze_and_unfreeze_layers()
        self._setup_classification_heads()
        self.log_model_info("MobileNet V2")

    def _load_pretrained_model(self):
        """Load the pretrained MobileNet V2 model."""
        # Using the default weights gave significantly worse results than V1
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    def _freeze_and_unfreeze_layers(self):
        """Implement MobileNet V2 specific freezing logic."""
        if self.freeze_backbone:
            # First freeze all features
            for param in self.model.features.parameters():
                param.requires_grad = False

            # Then unfreeze the last n inverted residual blocks if specified
            if self.unfreeze_last_n_blocks > 0:
                # MobileNet V2 has inverted residual blocks starting from index 1
                # The structure is: [conv2d, InvertedResidual x17, conv2d]
                # We want to unfreeze the last n InvertedResidual blocks
                total_blocks = len(
                    [m for m in self.model.features if hasattr(m, "conv")]
                )
                blocks_to_unfreeze = min(self.unfreeze_last_n_blocks, total_blocks)

                # Find inverted residual blocks (they have 'conv' attribute)
                inverted_blocks = [
                    i for i, m in enumerate(self.model.features) if hasattr(m, "conv")
                ]

                # Unfreeze the last n blocks
                for i in inverted_blocks[-blocks_to_unfreeze:]:
                    for param in self.model.features[i].parameters():
                        param.requires_grad = True

                log.info(
                    f"Frozen MobileNet V2 backbone features, unfroze last {blocks_to_unfreeze} inverted residual blocks"
                )
            else:
                log.info("Frozen MobileNet V2 backbone features")

    def _setup_classification_heads(self):
        """Setup dual classification heads for MobileNet V2."""
        num_ftrs = self.model.classifier[1].in_features

        # Remove the linear layer from the classifier
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier.children())[:-1]
        )

        # make new output layers
        self.model.primary_classifier = nn.Linear(num_ftrs, self.num_primary_classes)
        self.model.secondary_classifier = nn.Linear(
            num_ftrs, self.num_secondary_classes
        )
