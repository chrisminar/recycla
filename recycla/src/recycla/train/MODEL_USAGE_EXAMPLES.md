"""
Usage Examples for Custom Neural Network Models
===============================================

This file demonstrates how to use the new ModelType enum and updated train function.

## Available Model Types

The ModelType enum includes all custom neural network architectures:

### MobileNet V2
- ModelType.MOBILENET_V2

### EfficientNet V2
- ModelType.EFFICIENTNET_V2_S (Small)
- ModelType.EFFICIENTNET_V2_M (Medium)  
- ModelType.EFFICIENTNET_V2_L (Large)

### ConvNeXt
- ModelType.CONVNEXT_TINY
- ModelType.CONVNEXT_SMALL
- ModelType.CONVNEXT_BASE
- ModelType.CONVNEXT_LARGE

### RegNet (Y and X variants)
- ModelType.REGNET_Y_400MF, REGNET_Y_800MF, REGNET_Y_1_6GF, etc.
- ModelType.REGNET_X_400MF, REGNET_X_800MF, REGNET_X_1_6GF, etc.

### Vision Transformer (ViT)
- ModelType.VIT_B_16, VIT_B_32 (Base models)
- ModelType.VIT_L_16, VIT_L_32 (Large models)
- ModelType.VIT_H_14 (Huge model)

### ResNet
- ModelType.RESNET_18, RESNET_34, RESNET_50, RESNET_101, RESNET_152

## Command Line Usage

```bash
# Train with default MobileNet V2
python -m recycla.train.train --data-dir data/labeled_data --nepochs 20

# Train with EfficientNet V2 Small
python -m recycla.train.train --data-dir data/labeled_data --model-type efficientnet_v2_s --nepochs 20

# Train with ResNet-50 and frozen backbone
python -m recycla.train.train --data-dir data/labeled_data --model-type resnet_50 --freeze-backbone --nepochs 20

# Train with Vision Transformer
python -m recycla.train.train --data-dir data/labeled_data --model-type vit_b_16 --nepochs 20

# Train with class weights and specific unfreeze blocks
python -m recycla.train.train --data-dir data/labeled_data --model-type convnext_tiny --weight-labels --unfreeze-last-n-blocks 3 --nepochs 20
```

## Programmatic Usage

```python
from recycla.train.ModelType import ModelType
from recycla.train.train import _train

# Train with different model types
model_types = [
    ModelType.MOBILENET_V2,
    ModelType.EFFICIENTNET_V2_S,
    ModelType.RESNET_50,
    ModelType.VIT_B_16
]

for model_type in model_types:
    print(f"Training with {model_type.value}...")
    _train(
        data_dir="data/labeled_data",
        nepochs=10,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        model_type=model_type
    )
```

## Getting Model Classes Programmatically

```python
from recycla.train.ModelType import ModelType

# Get model class and kwargs for any model type
model_class, kwargs = ModelType.get_model_class_and_kwargs(ModelType.EFFICIENTNET_V2_M)

# Create model instance
model = model_class(
    num_primary_classes=10,
    num_secondary_classes=5,
    freeze_backbone=True,
    unfreeze_last_n_blocks=2,
    **kwargs
)
```

## Model Type Conversion

```python
from recycla.train.ModelType import ModelType

# Convert from string
model_type = ModelType.from_string("resnet_50")

# Convert to string
model_string = str(ModelType.RESNET_50)  # Returns "resnet_50"
```

## Default Behavior

- Default model type: `ModelType.MOBILENET_V2`
- The train function maintains backward compatibility
- All existing parameters work with any model type
- Model-specific parameters (like model size) are handled automatically

