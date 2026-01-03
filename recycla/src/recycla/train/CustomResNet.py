import torch.nn as nn
from torchvision import models

from recycla import log

from .BaseCustomModel import BaseCustomModel


class CustomResNet(BaseCustomModel):
    def __init__(
        self,
        num_primary_classes,
        num_secondary_classes,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        model_size="50",  # '18', '34', '50', '101', '152'
    ):
        self.model_size = model_size
        super(CustomResNet, self).__init__(
            num_primary_classes=num_primary_classes,
            num_secondary_classes=num_secondary_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        self._load_pretrained_model()
        self._freeze_and_unfreeze_layers()
        self._setup_classification_heads()
        self.log_model_info(f"ResNet-{model_size}")

    def _load_pretrained_model(self):
        """Load the pretrained ResNet model."""
        # Select model size
        model_map = {
            "18": models.resnet18,
            "34": models.resnet34,
            "50": models.resnet50,
            "101": models.resnet101,
            "152": models.resnet152,
        }

        if self.model_size not in model_map:
            raise ValueError(
                f"Model size '{self.model_size}' not supported. Use one of: {list(model_map.keys())}"
            )

        self.model = model_map[self.model_size](weights="IMAGENET1K_V1")

    def _freeze_and_unfreeze_layers(self):
        """Implement ResNet specific freezing logic."""
        if self.freeze_backbone:
            # First freeze all layers except the final classifier
            for name, param in self.model.named_parameters():
                if (
                    "fc" not in name
                ):  # Don't freeze the final fc layer (we'll replace it anyway)
                    param.requires_grad = False

            # Then unfreeze the last n residual layers if specified
            if self.unfreeze_last_n_blocks > 0:
                # ResNet structure: has layer1, layer2, layer3, layer4
                # Each layer contains multiple residual blocks
                layers = [
                    self.model.layer1,
                    self.model.layer2,
                    self.model.layer3,
                    self.model.layer4,
                ]
                total_layers = len(layers)
                layers_to_unfreeze = min(self.unfreeze_last_n_blocks, total_layers)

                # Unfreeze the last n layers
                for layer in layers[-layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True

                log.info(
                    f"Frozen ResNet backbone, unfroze last {layers_to_unfreeze} residual layers"
                )
            else:
                log.info("Frozen ResNet backbone")

    def _setup_classification_heads(self):
        """Setup dual classification heads for ResNet."""
        # Get the number of features from the classifier
        num_ftrs = self.model.fc.in_features

        # Remove the final fully connected layer
        self.model.fc = nn.Identity()

        # Add custom classification heads
        self.model.primary_classifier = nn.Linear(num_ftrs, self.num_primary_classes)
        self.model.secondary_classifier = nn.Linear(
            num_ftrs, self.num_secondary_classes
        )
