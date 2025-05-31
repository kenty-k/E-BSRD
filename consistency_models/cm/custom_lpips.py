from typing import List
import torch
from piq import LPIPS  # 元のLPIPSクラスがあるモジュールを適切にインポート

class CustomLPIPS(LPIPS):
    def __init__(self, replace_pooling: bool = False, distance: str = "mse", reduction: str = "mean",
                 mean: List[float] = None, std: List[float] = None, model_path: str = None) -> None:
        """
        Custom LPIPS class to allow custom weights for the LPIPS model.

        Args:
            replace_pooling (bool): Replace MaxPooling with AveragePooling.
            distance (str): Distance metric ('mse' or 'mae').
            reduction (str): Reduction method ('none', 'mean', or 'sum').
            mean (List[float]): Mean values for normalization. Defaults to ImageNet mean if None.
            std (List[float]): Std values for normalization. Defaults to ImageNet std if None.
            model_path (str): Path to custom weights file. If None, defaults to pretrained weights.
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # Default ImageNet mean
        if std is None:
            std = [0.229, 0.224, 0.225]  # Default ImageNet std
        # Load custom weights if provided
        if model_path is not None:
            lpips_weights = torch.load(model_path, map_location="cpu")
            lpips_weights = tuple(lpips_weights)

        else:
            # Use default weights from the parent class
            lpips_weights = torch.hub.load_state_dict_from_url(
                self._weights_url, progress=False
            )
        breakpoint()
        # Initialize the parent class
        super().__init__(
            replace_pooling=replace_pooling,
            distance=distance,
            reduction=reduction,
            mean=mean,
            std=std,
        )

        # Override weights with custom weights if provided
        self.weights = lpips_weights
