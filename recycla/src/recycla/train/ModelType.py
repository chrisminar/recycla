"""
Enum for custom neural network model types.
"""

from enum import Enum

from recycla.train.CustomConvNeXt import CustomConvNeXt
from recycla.train.CustomEfficientNetV2 import CustomEfficientNetV2
from recycla.train.CustomMobileNetV2 import CustomMobileNetV2
from recycla.train.CustomRegNet import CustomRegNet
from recycla.train.CustomResNet import CustomResNet
from recycla.train.CustomViT import CustomViT


class ModelType(Enum):
    """Enum for selecting which custom neural network model to use."""

    MOBILENET_V2 = "mobilenet_v2"
    EFFICIENTNET_V2_S = "efficientnet_v2_s"
    EFFICIENTNET_V2_M = "efficientnet_v2_m"
    EFFICIENTNET_V2_L = "efficientnet_v2_l"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"
    CONVNEXT_BASE = "convnext_base"
    CONVNEXT_LARGE = "convnext_large"
    REGNET_Y_400MF = "regnet_y_400mf"
    REGNET_Y_800MF = "regnet_y_800mf"
    REGNET_Y_1_6GF = "regnet_y_1_6gf"
    REGNET_Y_3_2GF = "regnet_y_3_2gf"
    REGNET_Y_8GF = "regnet_y_8gf"
    REGNET_Y_16GF = "regnet_y_16gf"
    REGNET_Y_32GF = "regnet_y_32gf"
    REGNET_X_400MF = "regnet_x_400mf"
    REGNET_X_800MF = "regnet_x_800mf"
    REGNET_X_1_6GF = "regnet_x_1_6gf"
    REGNET_X_3_2GF = "regnet_x_3_2gf"
    REGNET_X_8GF = "regnet_x_8gf"
    REGNET_X_16GF = "regnet_x_16gf"
    REGNET_X_32GF = "regnet_x_32gf"
    VIT_B_16 = "vit_b_16"
    VIT_B_32 = "vit_b_32"
    VIT_L_16 = "vit_l_16"
    VIT_L_32 = "vit_l_32"
    VIT_H_14 = "vit_h_14"
    RESNET_18 = "resnet_18"
    RESNET_34 = "resnet_34"
    RESNET_50 = "resnet_50"
    RESNET_101 = "resnet_101"
    RESNET_152 = "resnet_152"

    @classmethod
    def get_model_class_and_kwargs(cls, model_type):
        """
        Get the model class and initialization kwargs for a given model type.

        Args:
            model_type (ModelType): The model type enum value

        Returns:
            tuple: (model_class, kwargs_dict)
        """
        model_mapping = {
            # MobileNet V2
            cls.MOBILENET_V2: (CustomMobileNetV2, {}),
            # EfficientNet V2
            cls.EFFICIENTNET_V2_S: (CustomEfficientNetV2, {"model_size": "s"}),
            cls.EFFICIENTNET_V2_M: (CustomEfficientNetV2, {"model_size": "m"}),
            cls.EFFICIENTNET_V2_L: (CustomEfficientNetV2, {"model_size": "l"}),
            # ConvNeXt
            cls.CONVNEXT_TINY: (CustomConvNeXt, {"model_size": "tiny"}),
            cls.CONVNEXT_SMALL: (CustomConvNeXt, {"model_size": "small"}),
            cls.CONVNEXT_BASE: (CustomConvNeXt, {"model_size": "base"}),
            cls.CONVNEXT_LARGE: (CustomConvNeXt, {"model_size": "large"}),
            # RegNet Y variants
            cls.REGNET_Y_400MF: (CustomRegNet, {"model_variant": "y_400mf"}),
            cls.REGNET_Y_800MF: (CustomRegNet, {"model_variant": "y_800mf"}),
            cls.REGNET_Y_1_6GF: (CustomRegNet, {"model_variant": "y_1_6gf"}),
            cls.REGNET_Y_3_2GF: (CustomRegNet, {"model_variant": "y_3_2gf"}),
            cls.REGNET_Y_8GF: (CustomRegNet, {"model_variant": "y_8gf"}),
            cls.REGNET_Y_16GF: (CustomRegNet, {"model_variant": "y_16gf"}),
            cls.REGNET_Y_32GF: (CustomRegNet, {"model_variant": "y_32gf"}),
            # RegNet X variants
            cls.REGNET_X_400MF: (CustomRegNet, {"model_variant": "x_400mf"}),
            cls.REGNET_X_800MF: (CustomRegNet, {"model_variant": "x_800mf"}),
            cls.REGNET_X_1_6GF: (CustomRegNet, {"model_variant": "x_1_6gf"}),
            cls.REGNET_X_3_2GF: (CustomRegNet, {"model_variant": "x_3_2gf"}),
            cls.REGNET_X_8GF: (CustomRegNet, {"model_variant": "x_8gf"}),
            cls.REGNET_X_16GF: (CustomRegNet, {"model_variant": "x_16gf"}),
            cls.REGNET_X_32GF: (CustomRegNet, {"model_variant": "x_32gf"}),
            # Vision Transformer
            cls.VIT_B_16: (CustomViT, {"model_size": "b_16"}),
            cls.VIT_B_32: (CustomViT, {"model_size": "b_32"}),
            cls.VIT_L_16: (CustomViT, {"model_size": "l_16"}),
            cls.VIT_L_32: (CustomViT, {"model_size": "l_32"}),
            cls.VIT_H_14: (CustomViT, {"model_size": "h_14"}),
            # ResNet
            cls.RESNET_18: (CustomResNet, {"model_size": "18"}),
            cls.RESNET_34: (CustomResNet, {"model_size": "34"}),
            cls.RESNET_50: (CustomResNet, {"model_size": "50"}),
            cls.RESNET_101: (CustomResNet, {"model_size": "101"}),
            cls.RESNET_152: (CustomResNet, {"model_size": "152"}),
        }

        if model_type not in model_mapping:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model_mapping[model_type]

    @classmethod
    def n_params(cls, model_type) -> int:
        """
        Get the number of parameters for a given model type.

        Args:
            model_type (ModelType): The model type enum value

        Returns:
            int: The number of parameters in millions
        """

        model_mapping = {
            cls.MOBILENET_V2: 2.26,
            cls.EFFICIENTNET_V2_S: 20.21,
            cls.EFFICIENTNET_V2_M: 52.89,
            cls.EFFICIENTNET_V2_L: 117.27,
            cls.CONVNEXT_TINY: 27.84,
            cls.CONVNEXT_SMALL: 49.48,
            cls.CONVNEXT_BASE: 87.60,
            cls.CONVNEXT_LARGE: 196.27,
            cls.REGNET_Y_400MF: 3.92,
            cls.REGNET_Y_800MF: 5.67,
            cls.REGNET_Y_1_6GF: 10.34,
            cls.REGNET_Y_3_2GF: 17.97,
            cls.REGNET_Y_8GF: 3.92,
            cls.REGNET_Y_16GF: 3.92,
            cls.REGNET_Y_32GF: 3.92,
            cls.REGNET_X_400MF: 5.11,
            cls.REGNET_X_800MF: 6.61,
            cls.REGNET_X_1_6GF: 8.30,
            cls.REGNET_X_3_2GF: 14.32,
            cls.REGNET_X_8GF: 3.92,
            cls.REGNET_X_16GF: 3.92,
            cls.REGNET_X_32GF: 3.92,
            cls.VIT_B_16: 85.82,
            cls.VIT_B_32: 87.48,
            cls.VIT_L_16: 303.33,
            cls.VIT_L_32: 305.54,
            cls.VIT_H_14: 85.82,
            cls.RESNET_18: 11.19,
            cls.RESNET_34: 21.30,
            cls.RESNET_50: 23.57,
            cls.RESNET_101: 42.56,
            cls.RESNET_152: 58.20,
        }

        if model_type not in model_mapping:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model_mapping[model_type] * 1e6

    @classmethod
    def from_string(cls, model_str: str):
        """
        Create ModelType from string value.

        Args:
            model_str (str): String representation of the model type

        Returns:
            ModelType: The corresponding enum value
        """
        for model_type in cls:
            if model_type.value == model_str:
                return model_type
        raise ValueError(f"Unknown model type: {model_str}")

    def __str__(self):
        return self.value
