import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
BLACK_ID = 0
def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
def encode_rgb24(rgb: np.ndarray) -> np.ndarray:
    rgb32 = rgb.astype(np.uint32, copy=False)
    return (rgb32[..., 0] << 16) | (rgb32[..., 1] << 8) | rgb32[..., 2]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Ground truth mask folder")
    ap.add_argument("--pred", required=True, help="Predicted mask folder")
    args = ap.parse_args()
    gt_dir = Path(args.gt)
    pr_dir = Path(args.pred)
    gt_files = {p.name for p in gt_dir.iterdir() if p.is_file()}
    pr_files = {p.name for p in pr_dir.iterdir() if p.is_file()}
    common = sorted(gt_files & pr_files)
    sum_iou_macro = 0.0
    sum_dice_macro = 0.0
    n_gt_objects = 0
    n_images = 0
    pix_correct = 0
    pix_total = 0
    for name in tqdm(common):
        gt_id = encode_rgb24(load_rgb(gt_dir / name)).ravel()
        pr_id = encode_rgb24(load_rgb(pr_dir / name)).ravel()
        pix_correct += int(np.count_nonzero(gt_id == pr_id))
        pix_total += int(gt_id.size)
        gt_u, gt_c = np.unique(gt_id, return_counts=True)
        pr_u, pr_c = np.unique(pr_id, return_counts=True)
        gt_counts = dict(zip(gt_u.tolist(), gt_c.tolist()))
        pr_counts = dict(zip(pr_u.tolist(), pr_c.tolist()))
        pair = (gt_id.astype(np.uint64) << 24) | pr_id.astype(np.uint64)
        pair_u, pair_c = np.unique(pair, return_counts=True)
        joint_counts = dict(zip(pair_u.tolist(), pair_c.tolist()))
        gt_colors = gt_u[gt_u != BLACK_ID]
        for c_np in gt_colors:
            c = int(c_np)
            gt_sum = gt_counts[c]
            pr_sum = pr_counts.get(c, 0)
            cc_key = (c << 24) | c
            inter = joint_counts.get(cc_key, 0)
            union = gt_sum + pr_sum - inter
            denom = gt_sum + pr_sum
            iou = (inter / union) if union else 0.0
            dice = ((2 * inter) / denom) if denom else 0.0
            sum_iou_macro += iou
            sum_dice_macro += dice
            n_gt_objects += 1
        n_images += 1
    mean_iou = (sum_iou_macro / n_gt_objects) if n_gt_objects else 0.0
    mean_dice = (sum_dice_macro / n_gt_objects) if n_gt_objects else 0.0
    pix_acc = (pix_correct / pix_total) if pix_total else 0.0
    print("=== Color-correspondence instance evaluation ===")
    print(f"Images evaluated:        {n_images}")
    print(f"GT objects (total):      {n_gt_objects}")
    print("\n--- Per-object segmentation (macro over GT objects) ---")
    print(f"Mean IoU:                {mean_iou:.4f}")
    print(f"Mean Dice:               {mean_dice:.4f}")
    print("\n--- Pixel accuracy (exact RGB match) ---")
    print(f"Pixel accuracy:          {pix_acc:.4f}")
if __name__ == "__main__":
    main()
