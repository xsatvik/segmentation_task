"""
Train a YOLO bounding-box (detection) model using Ultralytics on a Roboflow-exported dataset.

Example:
  python scripts/train_yolo_bbox.py \
    --data /Users/satvikahuja13/Downloads/Drywall_data/data.yaml \
    --model yolov8n.pt \
    --device mps

W&B:
  export WANDB_API_KEY=...        # recommended (non-interactive)
  export WANDB_PROJECT=drywall    # optional
  python scripts/train_yolo_bbox.py --wandb
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any

_STATE: dict[str, Any] = {
    "epoch": None,  # 0-indexed
    "save_period": None,
    "val_iou_sum": 0.0,
    "val_iou_gt": 0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO bbox model (Ultralytics).")
    p.add_argument(
        "--data",
        type=str,
        default=os.environ.get("DRYWALL_DATA_YAML", ""),
        help="Path to YOLO data.yaml (absolute path recommended). Can also set DRYWALL_DATA_YAML env var.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("YOLO_MODEL", "yolov8n.pt"),
        help="Ultralytics model (e.g., yolov8n.pt, yolov8n.yaml, or a local .pt).",
    )
    p.add_argument("--epochs", type=int, default=int(os.environ.get("YOLO_EPOCHS", "50")))
    p.add_argument("--imgsz", type=int, default=int(os.environ.get("YOLO_IMGSZ", "640")))
    p.add_argument("--batch", type=int, default=int(os.environ.get("YOLO_BATCH", "16")))
    p.add_argument(
        "--device",
        type=str,
        default=os.environ.get("YOLO_DEVICE", "auto"),
        help="Device: auto/cpu/cuda/mps or CUDA index like 0.",
    )
    p.add_argument(
        "--project",
        type=str,
        default=os.environ.get("YOLO_PROJECT", "outputs/yolo_bbox"),
        help="Output project directory.",
    )
    p.add_argument(
        "--name",
        type=str,
        default=os.environ.get("YOLO_RUN_NAME", "drywall_join"),
        help="Run name (subdir under project).",
    )
    p.add_argument(
        "--exist-ok",
        action="store_true",
        help="Overwrite existing run directory if present.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint (weights/last.pt) if available.",
    )
    p.add_argument(
        "--save-period",
        type=int,
        default=int(os.environ.get("YOLO_SAVE_PERIOD", "3")),
        help="Save a checkpoint every N epochs (in addition to best/last). -1 disables periodic saving.",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Save training curves/plots (e.g., losses and metrics) into the run directory.",
    )
    p.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        default=os.environ.get("YOLO_WANDB", "1").lower() not in {"0", "false", "no"},
        help="Enable Weights & Biases logging (default: enabled).",
    )
    p.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable Weights & Biases logging.",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "drywall_yolo_bbox"),
        help="W&B project name.",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", ""),
        help="W&B entity/team (optional).",
    )
    p.add_argument(
        "--wandb-mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
        help="W&B mode. If no API key is available, we auto-fallback to offline to avoid interactive prompts.",
    )
    return p.parse_args()

def _enable_ultralytics_wandb() -> None:
    """
    Best-effort: enable W&B logger inside Ultralytics across versions.
    If the APIs differ, Ultralytics may still auto-enable W&B when wandb is installed.
    """
    try:
        from ultralytics.utils import SETTINGS  # type: ignore

        if isinstance(SETTINGS, dict):
            SETTINGS["wandb"] = True
    except Exception:
        pass

    try:
        # Some versions expose settings via ultralytics import
        from ultralytics import settings  # type: ignore

        try:
            settings.update({"wandb": True})  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def _configure_wandb(args: argparse.Namespace) -> None:
    if not args.wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return

    # Avoid W&B ever prompting interactively mid-training.
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key and args.wandb_mode == "online":
        print("WANDB_API_KEY not set; switching WANDB_MODE=offline to avoid interactive login prompts.")
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ["WANDB_MODE"] = args.wandb_mode

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    os.environ.setdefault("WANDB_SILENT", "true")

    # Non-interactive login if key is provided.
    if api_key:
        try:
            import wandb  # type: ignore

            wandb.login(key=api_key, relogin=True)
        except Exception as e:
            print(f"Warning: W&B login failed ({e}). Training will continue; logging may be offline/disabled.")

    _enable_ultralytics_wandb()


def _greedy_mean_iou(gt_xyxy, gt_cls, pred_xyxy, pred_cls) -> tuple[float, int]:
    """Compute mean IoU over GT boxes using greedy 1-1 matching within each class."""
    import torch
    from ultralytics.utils.metrics import box_iou  # type: ignore

    if gt_xyxy.numel() == 0:
        return 0.0, 0

    total_iou = 0.0
    total_gt = int(gt_xyxy.shape[0])

    classes = torch.unique(gt_cls).tolist()
    for c in classes:
        gt_mask = gt_cls == c
        pr_mask = pred_cls == c
        gt_c = gt_xyxy[gt_mask]
        pr_c = pred_xyxy[pr_mask]
        if gt_c.numel() == 0:
            continue
        if pr_c.numel() == 0:
            continue  # contributes 0 for all those GTs

        ious = box_iou(gt_c, pr_c)  # [G, P]
        # Greedy match highest IoU pairs without reuse
        g_used = set()
        p_used = set()
        flat = torch.argsort(ious.reshape(-1), descending=True)
        G, P = ious.shape
        for k in flat.tolist():
            g = k // P
            p = k % P
            if g in g_used or p in p_used:
                continue
            g_used.add(g)
            p_used.add(p)
            total_iou += float(ious[g, p].item())
            if len(g_used) == G or len(p_used) == P:
                break

    return total_iou, total_gt


def _on_train_epoch_start(trainer) -> None:
    _STATE["epoch"] = int(getattr(trainer, "epoch", 0))
    _STATE["save_period"] = int(getattr(trainer, "save_period", -1))


def _on_val_start(validator) -> None:
    _STATE["val_iou_sum"] = 0.0
    _STATE["val_iou_gt"] = 0


def _on_val_batch_end(validator) -> None:
    """
    Accumulate a simple bbox IoU metric on the validation set.

    Notes:
    - This is NOT mIoU. It's a mean IoU over GT boxes using greedy 1-1 matching within each class.
    - Uses internal validator loop variables (`batch`, `preds`) via frame inspection.
    """
    try:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None or frame.f_back.f_back is None:
            return
        v = frame.f_back.f_back.f_locals
        batch = v.get("batch")
        preds = v.get("preds")
        if batch is None or preds is None:
            return

        import torch
        from ultralytics.utils import ops  # type: ignore

        imgsz = batch["img"].shape[2:]  # (h, w)
        scale = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], device=batch["bboxes"].device)

        bs = len(preds)
        for si in range(bs):
            idx = batch["batch_idx"] == si
            gt_cls = batch["cls"][idx].squeeze(-1)
            gt_xyxy = ops.xywh2xyxy(batch["bboxes"][idx]) * scale

            pr = preds[si]
            pred_xyxy = pr["bboxes"]
            pred_cls = pr["cls"]

            iou_sum, gt_n = _greedy_mean_iou(gt_xyxy, gt_cls, pred_xyxy, pred_cls)
            _STATE["val_iou_sum"] += iou_sum
            _STATE["val_iou_gt"] += gt_n
    except Exception:
        # Never break validation if our custom metric fails.
        return


def _on_val_end(validator) -> None:
    gt_n = int(_STATE.get("val_iou_gt") or 0)
    if gt_n <= 0:
        return

    mean_iou = float(_STATE["val_iou_sum"]) / gt_n
    epoch = int(_STATE.get("epoch") or 0) + 1

    # Log every epoch (W&B will show it); you can filter epochs 3,6,9,... if save_period=3.
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            wandb.log({"val/iou": mean_iou, "epoch": epoch})
    except Exception:
        pass

    # Also print a clean line for terminal logs.
    print(f"val/iou={mean_iou:.4f} (epoch={epoch})")


def main() -> None:
    args = parse_args()

    if not args.data:
        raise SystemExit(
            "Missing --data. Provide a path to your dataset data.yaml, e.g. "
            "--data /Users/satvikahuja13/Downloads/Drywall_data/data.yaml"
        )

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    _configure_wandb(args)

    # Import here so reading --help doesn't require ultralytics installed.
    from ultralytics import YOLO  # type: ignore

    model = YOLO(args.model)
    model.add_callback("on_train_epoch_start", _on_train_epoch_start)
    model.add_callback("on_val_start", _on_val_start)
    model.add_callback("on_val_batch_end", _on_val_batch_end)
    model.add_callback("on_val_end", _on_val_end)
    train_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        val=True,
        exist_ok=args.exist_ok,
        resume=args.resume,
        save_period=args.save_period,
        plots=args.plots,
    )
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

