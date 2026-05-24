import argparse
import gc
import os
from io import BytesIO
import cv2
import numpy as np
import requests
from PIL import Image

AUTO_MODELS = ["birefnet-general-lite", "birefnet-general", "isnet-general-use", "u2net"]


def load_image(path):
    if path.startswith("http"):
        print("[Download]", path)
        resp = requests.get(path, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGBA")
        temp_path = "input_temp.png"
        img.save(temp_path)
        return temp_path

    if not os.path.exists(path):
        print("[ERR] File not found:", path)
        return None

    return path


def load_rembg_session(model_name="birefnet-general-lite"):
    from rembg import new_session

    print("[Model]", model_name)
    return new_session(model_name)


def resolve_model_names(model_name):
    name = (model_name or "").strip().lower()
    if name == "auto":
        return AUTO_MODELS
    if "," in name:
        return [x.strip() for x in name.split(",") if x.strip()]
    if not name:
        return ["birefnet-general-lite"]
    return [name]


def refine_alpha(alpha):
    alpha = alpha.astype(np.uint8)
    alpha = cv2.medianBlur(alpha, 3)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return alpha


def resize_rgba_max_side(rgba, max_side):
    w, h = rgba.size
    long_side = max(w, h)
    if long_side <= max_side:
        return rgba, (w, h)

    scale = float(max_side) / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = rgba.resize((new_w, new_h), Image.LANCZOS)
    return resized, (w, h)


def build_size_attempts(orig_w, orig_h, preferred_max_side):
    long_side = max(orig_w, orig_h)
    sizes = [preferred_max_side, 1280, 1024, 896, 768, 640, 512]
    attempts = []
    for s in sizes:
        s = int(s)
        if s <= 0:
            continue
        if s > long_side:
            s = long_side
        if s not in attempts:
            attempts.append(s)
    return attempts


def merge_alpha_maps(alpha_maps):
    if len(alpha_maps) == 1:
        return alpha_maps[0].astype(np.uint8)

    stack = np.stack(alpha_maps).astype(np.float32)
    alpha_max = np.max(stack, axis=0)
    alpha_mean = np.mean(stack, axis=0)
    votes = np.sum(stack > 120, axis=0)

    merged = np.where(votes >= 2, np.maximum(alpha_mean, alpha_max * 0.95), alpha_mean * 0.75)
    return np.clip(merged, 0, 255).astype(np.uint8)


def keep_near_main_subject(alpha):
    mask = (alpha > 18).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return alpha

    main = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = (labels == main).astype(np.uint8)
    near_main = cv2.dilate(
        main_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
        iterations=1
    ) > 0

    keep = np.zeros_like(mask)
    for label in range(1, n):
        comp = labels == label
        if np.any(comp & near_main):
            keep[comp] = 1

    out = np.zeros_like(alpha)
    out[keep > 0] = alpha[keep > 0]
    return out


def grabcut_center_mask(rgb):
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (
        int(w * 0.08),
        int(h * 0.06),
        int(w * 0.84),
        int(h * 0.88),
    )
    cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.medianBlur(fg, 5)
    return fg


def recover_flag_region(rgb, alpha):
    fg = alpha > 20
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return alpha

    h, w = alpha.shape
    upper = ys < int(h * 0.60)
    if np.any(upper):
        ys_u = ys[upper]
        xs_u = xs[upper]
    else:
        ys_u = ys
        xs_u = xs

    top_y = int(ys_u.min())
    top_idx = np.where(ys_u <= top_y + 3)[0]
    top_x = int(np.median(xs_u[top_idx]))
    anchor = np.array([top_x, top_y], dtype=np.int32)

    radius = max(70, int(min(h, w) * 0.18))
    x1 = max(0, anchor[0] - radius)
    y1 = max(0, anchor[1] - radius)
    x2 = min(w, anchor[0] + radius)
    y2 = min(h, anchor[1] + radius)

    roi = np.zeros((h, w), dtype=np.uint8)
    roi[y1:y2, x1:x2] = 1

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 40), (12, 255, 255)) > 0
    red2 = cv2.inRange(hsv, (165, 80, 40), (179, 255, 255)) > 0
    yellow = cv2.inRange(hsv, (15, 70, 40), (45, 255, 255)) > 0
    color = (red1 | red2 | yellow)

    candidate = (color & (roi > 0)).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )

    seed = np.zeros_like(candidate, dtype=np.uint8)
    cv2.circle(seed, (int(anchor[0]), int(anchor[1])), 18, 1, -1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros_like(candidate, dtype=np.uint8)
    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 20 or area > int(h * w * 0.05):
            continue
        comp = labels == label
        ys_c, xs_c = np.where(comp)
        if len(xs_c) == 0:
            continue

        min_dist = np.min(
            (xs_c.astype(np.float32) - float(anchor[0])) ** 2 +
            (ys_c.astype(np.float32) - float(anchor[1])) ** 2
        ) ** 0.5

        cx = float(np.mean(xs_c))
        cy = float(np.mean(ys_c))
        center_dist = ((cx - float(anchor[0])) ** 2 + (cy - float(anchor[1])) ** 2) ** 0.5

        is_upper = cy < float(h) * 0.45
        near_anchor = (min_dist < radius * 0.75) or (center_dist < radius * 0.70)
        touch_seed = np.any(comp & (seed > 0))

        if is_upper and (touch_seed or near_anchor):
            keep[comp] = 1

    out = alpha.copy()
    out[keep > 0] = 255
    return out


def remove_bg_pro(
    image_path,
    output_path,
    model_name="birefnet-general-lite",
    recover_flag=False,
    ensemble=False,
    max_side=1024
):
    print("[BG] Removing background")

    try:
        from rembg import remove

        with Image.open(image_path) as img:
            rgba = img.convert("RGBA")

        src_rgb_full = np.array(rgba.convert("RGB"))
        model_names = resolve_model_names(model_name)
        size_attempts = build_size_attempts(rgba.size[0], rgba.size[1], max_side)
        alpha_maps = []

        for name in model_names:
            model_ok = False
            for attempt_side in size_attempts:
                try:
                    session = load_rembg_session(name)
                    rgba_infer, orig_size = resize_rgba_max_side(rgba, attempt_side)
                    result = remove(rgba_infer, session=session)
                    if not isinstance(result, Image.Image):
                        result = Image.fromarray(result)
                    result = result.convert("RGBA")
                    alpha_small = np.array(result)[:, :, 3].astype(np.uint8)
                    alpha_full = cv2.resize(
                        alpha_small,
                        orig_size,
                        interpolation=cv2.INTER_LINEAR
                    )
                    alpha_maps.append(alpha_full.astype(np.uint8))
                    print("[Mask] OK:", name, "max_side=", attempt_side)
                    model_ok = True
                    break
                except Exception as model_error:
                    msg = str(model_error)
                    is_oom = (
                        ("AllocateRawInternal" in msg) or
                        ("bad allocation" in msg) or
                        ("Failed to allocate memory" in msg)
                    )
                    if is_oom:
                        print("[Mask] OOM:", name, "max_side=", attempt_side, "-> retry smaller")
                    else:
                        print("[Mask] FAIL:", name, model_error)
                        break
                finally:
                    # Release ONNX objects aggressively to avoid bad allocation on next model.
                    session = None
                    result = None
                    rgba_infer = None
                    gc.collect()

            if model_ok and not ensemble:
                break

        if not alpha_maps:
            print("[ERR] No model produced a mask")
            return False

        alpha = merge_alpha_maps(alpha_maps)
        alpha = keep_near_main_subject(alpha)

        coverage = float(np.mean(alpha > 18))
        if coverage > 0.80:
            print("[Mask] rembg too broad, applying GrabCut fallback")
            gc_alpha = grabcut_center_mask(src_rgb_full)
            alpha = np.minimum(alpha, gc_alpha)

        if recover_flag:
            alpha = recover_flag_region(src_rgb_full, alpha)
        arr = np.array(rgba.convert("RGBA"))
        arr[:, :, 3] = alpha
        arr[:, :, 3] = refine_alpha(arr[:, :, 3])

        Image.fromarray(arr, "RGBA").save(output_path)
        print("[OK] Saved:", output_path)
        return True

    except Exception as e:
        print("[ERR] rembg failed:", e)
        return False


def process_image(
    image_input,
    model_name="birefnet-general-lite",
    output_path=None,
    recover_flag=False,
    ensemble=False,
    max_side=1024
):
    image_path = load_image(image_input)
    if image_path is None:
        return False

    base = os.path.splitext(os.path.basename(image_path))[0]
    if output_path is None:
        output_path = f"{base}_HD.png"

    ok = remove_bg_pro(
        image_path,
        output_path,
        model_name=model_name,
        recover_flag=recover_flag,
        ensemble=ensemble,
        max_side=max_side
    )

    if image_input.startswith("http") and image_path == "input_temp.png" and os.path.exists(image_path):
        os.remove(image_path)

    return ok


def main():
    parser = argparse.ArgumentParser(description="Background remover using rembg")
    parser.add_argument("input", nargs="*", help="Input image path or URL")
    parser.add_argument(
        "--model",
        default="auto",
        help="Model name, comma-list, or 'auto' (tries lightweight-first)"
    )
    parser.add_argument("--max-side", type=int, default=1024, help="Max inference side length to reduce memory usage")
    parser.add_argument("--ensemble", action="store_true", help="Combine masks from all successful models (uses more memory)")
    parser.add_argument("--recover-flag", action="store_true", help="Recover missing red/yellow flag near raised hand")
    parser.add_argument("--output", "-o", default=None, help="Output path")

    args = parser.parse_args()

    if not args.input:
        args.input = ["C:/Users/Admin/Desktop/BT AI/anhcanha.jpg"]

    for img in args.input:
        print(f"\n[Process] {img}")
        process_image(
            img,
            model_name=args.model,
            output_path=args.output,
            recover_flag=args.recover_flag,
            ensemble=args.ensemble,
            max_side=args.max_side
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
