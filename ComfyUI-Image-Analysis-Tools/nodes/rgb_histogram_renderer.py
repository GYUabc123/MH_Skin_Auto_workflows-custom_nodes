import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
from comfy_api.latest import io as io_comfy # Rename to avoid conflict with io module

class RGBHistogramRenderer(io_comfy.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io_comfy.Schema(
            node_id="RGBHistogramRenderer",
            display_name="RGB Histogram Renderer",
            category="Image Analysis",
            inputs=[
                io_comfy.Image.Input("image"),
            ],
            outputs=[
                io_comfy.Image.Output()
            ]
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    CATEGORY = "Image Analysis"

    @classmethod
    def execute(cls, image):
        try:
            # ComfyUI image batches are [B, H, W, C]
            img_tensor = image[0]

            np_img = img_tensor.cpu().numpy()

            # The original code had: np_img = image[0].cpu().numpy().transpose(1, 2, 0)
            # This assumed input was CHW, but ComfyUI usually passes HWC for IMAGE type.
            # We fix it by removing transpose if it's already HWC.

            # Check dimensions to be safe
            if np_img.ndim == 3:
                # If shape is [3, H, W], transpose to [H, W, 3]
                if np_img.shape[0] == 3 and np_img.shape[2] > 3:
                     np_img = np_img.transpose(1, 2, 0)
                # Else assume [H, W, 3] - correct for ComfyUI

            red = np_img[:, :, 0]
            green = np_img[:, :, 1]
            blue = np_img[:, :, 2]

            # Plot histograms for R, G, B on same axes
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.hist(red.ravel(), bins=256, color='red', alpha=0.5, label='Red')
            ax.hist(green.ravel(), bins=256, color='green', alpha=0.5, label='Green')
            ax.hist(blue.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
            ax.set_title("RGB Histogram")
            ax.legend()
            fig.tight_layout()

            # Save plot to buffer as RGB PNG
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, transparent=False, facecolor='white')
            plt.close(fig)
            buf.seek(0)
            pil_image = Image.open(buf).convert("RGB")  # remove alpha if present

            # Convert to tensor in format (1, H, W, 3) float32 0-1
            img_np = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W, 3)

            return (img_tensor,)

        except Exception as e:
            print(f"[RGBHistogramInlineViewer] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (fallback,)
