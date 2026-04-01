# nodes.py
import torch
import numpy as np
from PIL import Image, ImageColor, ImageOps
from .presets import PRESETS, RESIZE_ALGOS, get_size_from_preset

# ------------- 通用工具 -------------
def _resample(method: str):
    method = (method or "lanczos").lower()
    if method == "nearest":
        return Image.Resampling.NEAREST
    if method == "bilinear":
        return Image.Resampling.BILINEAR
    return Image.Resampling.LANCZOS


def _safe_color(color_text: str):
    try:
        return ImageColor.getrgb(color_text)
    except Exception:
        return (0, 0, 0)


def _ratio_wh(aspect_ratio: str, src_w: int, src_h: int):
    ratio_map = {
        "1:1": (1, 1),
        "2:3": (2, 3),
        "3:2": (3, 2),
        "3:4": (3, 4),
        "4:3": (4, 3),
        "9:16": (9, 16),
        "16:9": (16, 9),
        "1:2": (1, 2),
        "2:1": (2, 1),
        "1:3": (1, 3),
        "3:1": (3, 1),
    }
    if aspect_ratio == "original":
        return max(src_w, 1), max(src_h, 1)
    return ratio_map.get(aspect_ratio, (1, 1))


def _round_to_multiple(v: int, multiple: int):
    multiple = max(int(multiple), 1)
    if multiple == 1:
        return max(1, int(v))
    return max(multiple, int(round(v / multiple) * multiple))


def _target_size(rw: float, rh: float, p_w: float, p_h: float, side_mode: str, side_len: int, src_w: int, src_h: int, multiple: int):
    rw = max(float(rw), 1e-6)
    rh = max(float(rh), 1e-6)
    p_w = max(float(p_w), 1e-6)
    p_h = max(float(p_h), 1e-6)

    # proportional ratio refinement (lets user tweak ratio quickly)
    rw *= p_w
    rh *= p_h

    if side_mode == "longest":
        long_side = max(int(side_len), 1)
        if rw >= rh:
            w = long_side
            h = int(long_side * rh / rw)
        else:
            h = long_side
            w = int(long_side * rw / rh)
    elif side_mode == "shortest":
        short_side = max(int(side_len), 1)
        if rw >= rh:
            h = short_side
            w = int(short_side * rw / rh)
        else:
            w = short_side
            h = int(short_side * rh / rw)
    else:
        # None: keep source area scale but enforce chosen ratio
        src_area = max(src_w * src_h, 1)
        ratio = rw / rh
        w = int((src_area * ratio) ** 0.5)
        h = int(w / ratio)

    w = _round_to_multiple(max(w, 1), multiple)
    h = _round_to_multiple(max(h, 1), multiple)
    return w, h


def _fit_image_and_pad_mask(pil_img: Image.Image, target_w: int, target_h: int, fit: str, method: str, background_color: str):
    res = _resample(method)
    fit = (fit or "letterbox").lower()

    if fit == "stretch":
        return pil_img.resize((target_w, target_h), resample=res), None, None

    if fit == "crop":
        return ImageOps.fit(pil_img, (target_w, target_h), method=res, centering=(0.5, 0.5)), None, None

    # letterbox
    src_w, src_h = pil_img.size
    scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = pil_img.resize((new_w, new_h), resample=res)

    bg = _safe_color(background_color) if pil_img.mode != "L" else 0
    canvas = Image.new(pil_img.mode, (target_w, target_h), color=bg)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas.paste(resized, (ox, oy))

    # pad mask: padded=255, content=0
    pad_mask = Image.new("L", (target_w, target_h), color=255)
    pad_mask.paste(0, (ox, oy, ox + new_w, oy + new_h))
    bbox = (ox, oy, new_w, new_h)
    return canvas, pad_mask, bbox

# ------------- 简单图像尺寸 -------------
class EasySizeSimpleImage:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([
                    "original",
                    "1:1",
                    "2:3",
                    "3:2",
                    "3:4",
                    "4:3",
                    "9:16",
                    "16:9",
                    "1:2",
                    "2:1",
                    "1:3",
                    "3:1",
                ], {"default": "original"}),
                "proportional_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "proportional_height": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "fit": (["letterbox", "crop", "stretch"], {"default": "letterbox"}),
                "method": (RESIZE_ALGOS, {"default": "lanczos"}),
                "round_to_multiple": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "scale_to_side": (["None", "longest", "shortest"], {"default": "None"}),
                "scale_to_length": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "background_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("图像", "遮罩", "宽", "高")
    FUNCTION = "run"
    CATEGORY = "ScaleByRatio"

    def run(self, aspect_ratio, proportional_width, proportional_height, fit, method,
            round_to_multiple, scale_to_side, scale_to_length, background_color,
            图像=None, 遮罩=None):
        if 图像 is not None:
            _, src_h, src_w, _ = 图像.shape
        else:
            src_w, src_h = 1024, 1024

        rw, rh = _ratio_wh(aspect_ratio, src_w, src_h)
        side_mode = None if scale_to_side == "None" else scale_to_side
        out_w, out_h = _target_size(
            rw,
            rh,
            proportional_width,
            proportional_height,
            side_mode,
            scale_to_length,
            src_w,
            src_h,
            round_to_multiple,
        )

        # Image branch
        if 图像 is not None:
            arr = (图像.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(arr)
            out_img_pil, pad_mask, bbox = _fit_image_and_pad_mask(
                pil_img, out_w, out_h, fit, method, background_color
            )
            out_img_np = np.array(out_img_pil).astype(np.float32) / 255.0
            if out_img_np.ndim == 2:
                out_img_np = np.stack([out_img_np] * 3, axis=-1)
            图像_out = torch.from_numpy(out_img_np).unsqueeze(0)
        else:
            图像_out = torch.zeros((1, out_h, out_w, 3), dtype=torch.float32)
            pad_mask, bbox = (None, None)

        # Mask branch
        if 遮罩 is not None:
            m_arr = (遮罩.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            pil_msk = Image.fromarray(m_arr, mode="L")

            if fit == "stretch":
                out_msk_pil = pil_msk.resize((out_w, out_h), resample=_resample(method))
            elif fit == "crop":
                out_msk_pil = ImageOps.fit(pil_msk, (out_w, out_h), method=_resample(method), centering=(0.5, 0.5))
            else:
                # letterbox: union(existing mask, padded fill mask)
                if bbox is None:
                    # Keep behavior deterministic even if image input is absent.
                    # Build both content mask and pad mask, then union via max.
                    content_pil, pad_pil, _ = _fit_image_and_pad_mask(
                        pil_msk, out_w, out_h, "letterbox", method, "#000000"
                    )
                    content_np = np.array(content_pil).astype(np.float32) / 255.0
                    pad_np = np.array(pad_pil).astype(np.float32) / 255.0 if pad_pil is not None else np.zeros((out_h, out_w), dtype=np.float32)
                    out_msk_np = np.maximum(content_np, pad_np)
                    遮罩_out = torch.from_numpy(out_msk_np).unsqueeze(0)
                    return (图像_out, 遮罩_out, out_w, out_h)
                else:
                    ox, oy, new_w, new_h = bbox
                    resized = pil_msk.resize((new_w, new_h), resample=_resample(method))

                    # Content mask placed in content box.
                    content_pil = Image.new("L", (out_w, out_h), color=0)
                    content_pil.paste(resized, (ox, oy))
                    content_np = np.array(content_pil).astype(np.float32) / 255.0

                    # Padded fill mask uses pad_mask from image branch when available.
                    if pad_mask is not None:
                        pad_np = np.array(pad_mask).astype(np.float32) / 255.0
                    else:
                        pad_np = np.ones((out_h, out_w), dtype=np.float32)
                        pad_np[oy:oy + new_h, ox:ox + new_w] = 0.0

                    out_msk_np = np.maximum(content_np, pad_np)
                    遮罩_out = torch.from_numpy(out_msk_np).unsqueeze(0)
                    return (图像_out, 遮罩_out, out_w, out_h)

            out_msk_np = np.array(out_msk_pil).astype(np.float32) / 255.0
            遮罩_out = torch.from_numpy(out_msk_np).unsqueeze(0)
        else:
            if fit == "letterbox":
                # no input mask: content=0.0, padded=1.0
                base = np.ones((out_h, out_w), dtype=np.float32)
                if bbox is not None:
                    ox, oy, new_w, new_h = bbox
                    base[oy:oy + new_h, ox:ox + new_w] = 0.0
                else:
                    # if no image, nothing to pad against -> all content mask 0
                    base[:, :] = 0.0
                遮罩_out = torch.from_numpy(base).unsqueeze(0)
            else:
                遮罩_out = torch.zeros((1, out_h, out_w), dtype=torch.float32)

        return (图像_out, 遮罩_out, out_w, out_h)

# ------------- 简单图像尺寸-Latent -------------
class EasySizeSimpleLatent:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_dict = {k: ["关"] + [t[0] for t in PRESETS[k]] for k in PRESETS}
        return {
            "required": {
                **{k: (v, {"default": "关"}) for k, v in preset_dict.items()},
                "启用自定义尺寸": ("BOOLEAN", {"default": False}),
                "宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "ScaleByRatio"

    def run(self, **kwargs):
        use_custom = kwargs["启用自定义尺寸"]
        if use_custom:
            w, h = kwargs["宽度"], kwargs["高度"]
        else:
            choices = {k: kwargs[k] for k in PRESETS}
            w, h = get_size_from_preset(choices)
        latent = torch.zeros([1, 4, h // 8, w // 8])
        return ({"samples": latent},)

# ------------- 简单尺寸设置 -------------
class EasySizeSimpleSetting:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_dict = {k: ["关"] + [t[0] for t in PRESETS[k]] for k in PRESETS}
        return {
            "required": {
                **{k: (v, {"default": "关"}) for k, v in preset_dict.items()},
                "启用自定义尺寸": ("BOOLEAN", {"default": False}),
                "宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("宽度", "高度")
    FUNCTION = "run"
    CATEGORY = "ScaleByRatio"

    def run(self, **kwargs):
        use_custom = kwargs["启用自定义尺寸"]
        if use_custom:
            return (kwargs["宽度"], kwargs["高度"])
        choices = {k: kwargs[k] for k in PRESETS}
        w, h = get_size_from_preset(choices)
        return (w, h)

# -------------- 注册 --------------
NODE_CLASS_MAPPINGS = {
    # "EasySizeSimpleImage":   EasySizeSimpleImage,
    # "EasySizeSimpleLatent":  EasySizeSimpleLatent,
    # "EasySizeSimpleSetting": EasySizeSimpleSetting,
    # Backward-compatible aliases for existing workflows.
    "ScaleByRatioImage":     EasySizeSimpleImage,
    "ScaleByRatioLatent":    EasySizeSimpleLatent,
    "ScaleByRatioSetting":   EasySizeSimpleSetting,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "EasySizeSimpleImage":   "ScaleByRatio",
    # "EasySizeSimpleLatent":  "ScaleByRatio",
    # "EasySizeSimpleSetting": "ScaleByRatio",
    "ScaleByRatioImage":     "ScaleByRatio",
    "ScaleByRatioLatent":    "ScaleByRatio",
    "ScaleByRatioSetting":   "ScaleByRatio",
}