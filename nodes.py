import torch
from torchvision.transforms.functional import gaussian_blur
from comfy.cli_args import args
from comfy.model_management import soft_empty_cache
from comfy.utils import common_upscale
from comfy_extras.nodes_upscale_model import *

class GradientBlurNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0}),
                "direction": (["custom", "top_to_bottom", "bottom_to_top", 
                              "left_to_right", "right_to_left"],),
                "auto_center": ("BOOLEAN", {"default": True}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "sharp_edge": ("BOOLEAN", {"default": True}),
                "bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0}),  # Новый параметр
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "gradient_mask")
    FUNCTION = "apply_gradient_blur"
    CATEGORY = "image/blur"

    def apply_gradient_blur(self, image, intensity, direction, auto_center, center_x, center_y, sharp_edge, bias):
        if intensity <= 0:
            return (image, image)

        image = image.permute(0, 3, 1, 2)
        batch_size, channels, height, width = image.shape
        device = image.device

        # Определение направления градиента
        if direction == "custom":
            if auto_center:
                center_x = 0.5
                center_y = 0.5
        else:
            pass  # Предопределённые направления игнорируют center_x/y

        # Создание координатной сетки
        x = torch.linspace(0, 1, width, device=device)
        y = torch.linspace(0, 1, height, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        # Создание градиентной маски
        if sharp_edge:
            if direction == "top_to_bottom":
                distance = grid_y
            elif direction == "bottom_to_top":
                distance = 1 - grid_y
            elif direction == "left_to_right":
                distance = grid_x
            elif direction == "right_to_left":
                distance = 1 - grid_x
            elif direction == "custom":
                distance = torch.sqrt((grid_x - center_x)**2 + (grid_y - center_y)**2)
        else:
            distance = torch.sqrt((grid_x - center_x)**2 + (grid_y - center_y)**2)

        max_distance = distance.max()
        mask = distance / max_distance
        mask = torch.clamp(mask + bias, 0, 1)  # Применение bias

        # Применение размытия
        kernel_size = max(1, int(intensity) * 2 + 1)
        blurred = gaussian_blur(image, kernel_size=kernel_size, sigma=(intensity, intensity))

        # Интерполяция между оригинал и размытым
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height, width)
        result = image * (1 - mask_expanded) + blurred * mask_expanded

        # Преобразование маски для предпросмотра
        gradient_preview = mask.unsqueeze(-1).repeat(1, 1, channels).cpu().numpy()
        gradient_preview = torch.from_numpy(gradient_preview).unsqueeze(0).permute(0, 3, 1, 2)

        # Возврат к исходному формату
        result = result.permute(0, 2, 3, 1)
        gradient_preview = gradient_preview.permute(0, 2, 3, 1)

        return (result, gradient_preview)
