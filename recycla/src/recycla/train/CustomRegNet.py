import torch.nn as nn
from torchvision import models

from recycla import log
from recycla.train.BaseCustomModel import BaseCustomModel


class CustomRegNet(BaseCustomModel):
    def __init__(
        self,
        num_primary_classes,
        num_secondary_classes,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        model_variant="y_400mf",  # 'y_400mf', 'y_800mf', 'y_1_6gf', 'y_3_2gf', 'x_400mf', 'x_800mf', 'x_1_6gf', 'x_3_2gf'
    ):
        super(CustomRegNet, self).__init__(
            num_primary_classes,
            num_secondary_classes,
            freeze_backbone,
            unfreeze_last_n_blocks,
        )
        self.model_variant = model_variant

        self._load_pretrained_model()
        self._freeze_and_unfreeze_layers()
        self._setup_classification_heads()
        self.log_model_info(f"RegNet-{model_variant}")

    def _load_pretrained_model(self):
        """Load the pretrained RegNet model."""
        # Select model variant
        model_map = {
            "y_400mf": models.regnet_y_400mf,
            "y_800mf": models.regnet_y_800mf,
            "y_1_6gf": models.regnet_y_1_6gf,
            "y_3_2gf": models.regnet_y_3_2gf,
            "x_400mf": models.regnet_x_400mf,
            "x_800mf": models.regnet_x_800mf,
            "x_1_6gf": models.regnet_x_1_6gf,
            "x_3_2gf": models.regnet_x_3_2gf,
        }

        if self.model_variant not in model_map:
            raise ValueError(
                f"Model variant '{self.model_variant}' not supported. Use one of: {list(model_map.keys())}"
            )

        self.model = model_map[self.model_variant](weights="IMAGENET1K_V1")

    def _freeze_and_unfreeze_layers(self):
        """Implement RegNet specific freezing logic."""
        if self.freeze_backbone:
            # First freeze all trunk (backbone) features
            for param in self.model.trunk.parameters():
                param.requires_grad = False

            # Then unfreeze the last n blocks if specified
            if self.unfreeze_last_n_blocks > 0:
                # RegNet structure: trunk contains multiple stages (s1, s2, s3, s4)
                # Each stage has multiple blocks
                trunk_stages = []
                for name, module in self.model.trunk.named_children():
                    if name.startswith("s"):  # Stage modules (s1, s2, s3, s4)
                        trunk_stages.append(module)

                total_stages = len(trunk_stages)
                stages_to_unfreeze = min(self.unfreeze_last_n_blocks, total_stages)

                # Unfreeze the last n stages
                for stage in trunk_stages[-stages_to_unfreeze:]:
                    for param in stage.parameters():
                        param.requires_grad = True

                log.info(
                    f"Frozen RegNet backbone features, unfroze last {stages_to_unfreeze} stages"
                )
            else:
                log.info("Frozen RegNet backbone features")

    def _setup_classification_heads(self):
        """Setup dual classification heads for RegNet."""
        # Get the number of features from the classifier
        num_ftrs = self.model.fc.in_features

        # Remove the final fully connected layer
        self.model.fc = nn.Identity()

        # Add custom classification heads
        self.model.primary_classifier = nn.Linear(num_ftrs, self.num_primary_classes)
        self.model.secondary_classifier = nn.Linear(
            num_ftrs, self.num_secondary_classes
        )
