"""preprocessing: degradations for the Amazon robustness experiment."""

import json
import numpy as np
from config import CONDITIONS, CorruptionConfig
from hashlib import sha256
from scipy.ndimage import gaussian_filter, sobel
from skimage.transform import resize


def image_random_generator(image_id, operation, seed):
    """Return deterministic randomness independent of row order and Python hash().

    The text token contains image ID, operation family and experiment seed.
    All noise severities use operation='noise', so they share the same draw.
    All mask sizes use operation='occlusion', so they share the same location.
    """
    token = json.dumps([str(image_id), operation, int(seed)], separators=(",", ":"))
    number = int.from_bytes(sha256(token.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(number)


def apply_corruption(image, condition, image_id, seed=51, config=CorruptionConfig()):
    """Return (new float32 RGBN image, per-band noise-clipping fractions).

    Input and output both have shape (H, W, 4). The source array is never modified.
    Spatial transforms act on each band at matching coordinates. The four-item
    clipping array measures noise values outside the DN range BEFORE clipping;
    it is zero for conditions without added noise. It is quality evidence only.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}.")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 4 or min(array.shape[:2]) < 4:
        raise ValueError("Expected an H×W×4 RGBN image with sides at least 4 pixels.")
    if (not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all()
            or np.any(array < 0) or np.any(array > config.dn_scale)):
        raise ValueError("Pixels must be finite nonnegative DN within the encoding range.")
    output = array.astype(np.float32, copy=True)
    clipped = np.zeros(4, dtype=float)
    if condition == "clean":
        return output, clipped
    height, width = output.shape[:2]
    if condition.startswith("occlusion_"):
        area = {"occlusion_05": 0.05, "occlusion_10": 0.10, "occlusion_20": 0.20}[condition]
        side = max(1, min(height, width, int(round(np.sqrt(height * width * area)))))
        largest = max(1, min(height, width, int(round(np.sqrt(height * width * 0.20)))))
        generator = image_random_generator(image_id, "occlusion", seed)
        # Find the 20% square first, then centre smaller squares inside it.
        top = int(generator.integers(0, height - largest + 1)) + (largest - side) // 2
        left = int(generator.integers(0, width - largest + 1)) + (largest - side) // 2
        output[top:top + side, left:left + side, :] = config.occlusion_fill
    elif condition == "resolution":
        smaller_shape = (max(1, int(round(height * config.resolution_scale))),
                         max(1, int(round(width * config.resolution_scale))))
        for band in range(4):
            small = resize(output[..., band], smaller_shape, order=1, mode="reflect",
                           anti_aliasing=True, preserve_range=True)
            output[..., band] = resize(small, (height, width), order=1, mode="reflect",
                                      anti_aliasing=False, preserve_range=True)
    elif condition == "blur":
        # The final zero means do not blur along the spectral-band axis.
        output = gaussian_filter(output, sigma=(config.blur_sigma, config.blur_sigma, 0.0),
                                 mode="reflect")
    else:
        fraction = {"noise_low": config.noise_low_fraction,
                    "noise_high": config.noise_high_fraction,
                    "noise_stress": config.noise_stress_fraction}[condition]
        band_sd = np.full(4, fraction * config.dn_scale, dtype=float)
        noise = image_random_generator(image_id, "noise", seed).normal(size=output.shape) * band_sd
        output += noise.astype(np.float32)
        clipped = np.mean((output < 0) | (output > config.dn_scale), axis=(0, 1))
        np.clip(output, 0, config.dn_scale, out=output)
    return np.ascontiguousarray(output, dtype=np.float32), clipped


def corrupt_rgbn(image, condition, image_id, seed=51, config=CorruptionConfig()):
    """Convenience interface for previews: return only the transformed image."""
    return apply_corruption(image, condition, image_id, seed, config)[0]
