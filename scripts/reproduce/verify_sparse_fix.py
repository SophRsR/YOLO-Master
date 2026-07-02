"""
Verify the ES_MOE sparse-eval renormalization fix on a trained checkpoint.

Runs validation twice on the same weights:
  - dense  : use_sparse_inference=False on every ES_MOE (matches training path)
  - sparse : use_sparse_inference=True  (the inference path this branch fixes)

If the renorm fix works, the sparse mAP should jump from the collapsed value
(~0.04 mAP50-95 on VisDrone before the fix) toward the dense ceiling.

Usage:
  python scripts/reproduce/verify_sparse_fix.py \
      --weights "/path/to/best (1).pt" --data VisDrone.yaml --device mps
"""
from __future__ import annotations
import argparse
from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE


def set_sparse(model, enabled: bool) -> int:
    n = 0
    for m in model.model.modules():
        if isinstance(m, ES_MOE):
            m.use_sparse_inference = enabled
            n += 1
    return n


def run(weights, data, device, imgsz, sparse: bool):
    model = YOLO(weights)
    n = set_sparse(model, sparse)
    metrics = model.val(data=data, imgsz=imgsz, device=device, verbose=False)
    return n, metrics.box.map50, metrics.box.map


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="VisDrone.yaml")
    p.add_argument("--device", default="mps")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    print(f"[verify] weights={args.weights} data={args.data} device={args.device}\n")

    n, d50, d = run(args.weights, args.data, args.device, args.imgsz, sparse=False)
    print(f"[dense ] {n} ES_MOE modules | mAP50={d50:.4f}  mAP50-95={d:.4f}")

    n, s50, s = run(args.weights, args.data, args.device, args.imgsz, sparse=True)
    print(f"[sparse] {n} ES_MOE modules | mAP50={s50:.4f}  mAP50-95={s:.4f}")

    print("\n===== summary =====")
    print(f"{'path':<8}{'mAP50':>10}{'mAP50-95':>12}")
    print(f"{'dense':<8}{d50:>10.4f}{d:>12.4f}")
    print(f"{'sparse':<8}{s50:>10.4f}{s:>12.4f}")
    print(f"\nsparse/dense mAP50-95 ratio = {s / d:.3f}  (1.0 = fix fully recovers)")


if __name__ == "__main__":
    main()
