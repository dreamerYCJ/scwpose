"""Run SCW-Pose on one image or a directory of images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from scwpose import SCWPredictor

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Image file or image directory")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/scwpose.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or 0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/predictions")
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f"Source does not exist: {args.source}")
    paths = ([args.source] if args.source.is_file() else
             sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
    if not paths:
        parser.error("No input images found")
    torch.set_num_threads(args.threads)
    predictor = SCWPredictor(checkpoint=args.weights, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, 1):
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode image: {path}")
        result = predictor.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        drawn = frame.copy()
        for box, endpoints, score in zip(boxes, points, scores):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (255, 170, 50), 2)
            p1, p2 = np.rint(endpoints[:, :2]).astype(int)
            cv2.line(drawn, tuple(p1), tuple(p2), (40, 220, 240), 2, cv2.LINE_AA)
            for i, point in enumerate((p1, p2), 1):
                cv2.circle(drawn, tuple(point), 5, (60, 220, 80), -1, cv2.LINE_AA)
                cv2.putText(drawn, f"p{i}", (point[0] + 7, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 80), 2, cv2.LINE_AA)
            cv2.putText(drawn, f"walnut {score:.2f}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 170, 50), 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".png", drawn)
        if not ok:
            raise RuntimeError(f"Cannot encode prediction image: {path.name}")
        encoded.tofile(args.output / f"{path.stem}.png")
        payload = dict(image=path.name, boxes_xyxy=boxes.tolist(), box_scores=scores.tolist(),
                       endpoints_xy_conf=points.tolist())
        (args.output / f"{path.stem}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[{index}/{len(paths)}] {path.name}: {len(boxes)} walnuts", flush=True)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
