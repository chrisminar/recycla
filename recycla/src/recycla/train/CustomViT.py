import torch.nn as nn
from torchvision import models

from recycla import log

from .BaseCustomModel import BaseCustomModel


class CustomViT(BaseCustomModel):
    def __init__(
        self,
        num_primary_classes,
        num_secondary_classes,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        model_size="b_16",  # 'b_16', 'b_32', 'l_16', 'l_32', 'h_14'
    ):
        self.model_size = model_size
        super(CustomViT, self).__init__(
            num_primary_classes=num_primary_classes,
            num_secondary_classes=num_secondary_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

        self._load_pretrained_model()
        self._freeze_and_unfreeze_layers()
        self._setup_classification_heads()
        self.log_model_info(f"ViT-{model_size}")

    def _load_pretrained_model(self):
        """Load the pretrained Vision Transformer model."""
        # Select model size
        model_map = {
            "b_16": models.vit_b_16,
            "b_32": models.vit_b_32,
            "l_16": models.vit_l_16,
            "l_32": models.vit_l_32,
            "h_14": models.vit_h_14,
        }

        if self.model_size not in model_map:
            raise ValueError(
                f"Model size '{self.model_size}' not supported. Use one of: {list(model_map.keys())}"
            )

        self.model = model_map[self.model_size](weights="IMAGENET1K_V1")

    def _freeze_and_unfreeze_layers(self):
        """Implement Vision Transformer specific freezing logic."""
        if self.freeze_backbone:
            # First freeze the patch embedding and encoder
            for param in self.model.conv_proj.parameters():
                param.requires_grad = False

            # Freeze all transformer encoder layers
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            # Then unfreeze the last n transformer layers if specified
            if self.unfreeze_last_n_blocks > 0:
                # ViT structure: encoder contains multiple transformer layers
                encoder_layers = list(self.model.encoder.layers.children())
                total_layers = len(encoder_layers)
                layers_to_unfreeze = min(self.unfreeze_last_n_blocks, total_layers)

                # Unfreeze the last n transformer layers
                for layer in encoder_layers[-layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True

                log.info(
                    f"Frozen ViT backbone, unfroze last {layers_to_unfreeze} transformer layers"
                )
            else:
                log.info("Frozen ViT backbone")

    def _setup_classification_heads(self):
        """Setup dual classification heads for Vision Transformer."""
        # Get the number of features from the classifier
        num_ftrs = self.model.heads.head.in_features

        # Remove the classification head
        self.model.heads.head = nn.Identity()

        # Add custom classification heads
        self.model.primary_classifier = nn.Linear(num_ftrs, self.num_primary_classes)
        self.model.secondary_classifier = nn.Linear(
            num_ftrs, self.num_secondary_classes
        )
