"""Evaluate SCW-Pose on YOLO-format two-endpoint annotations."""
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


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def read_labels(path, width, height):
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation: {path}")
    if not path.read_text(encoding="utf-8").strip():
        return np.empty((0, 4)), np.empty((0, 2, 3))
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.shape[1] != 11 or not np.isfinite(values).all():
        raise ValueError(f"Expected 11 finite values per annotation row: {path}")
    if not (values[:, 0] == 0).all():
        raise ValueError(f"Expected class 0 (walnut): {path}")
    center = values[:, 1:3] * [width, height]
    size = values[:, 3:5] * [width, height]
    boxes = np.concatenate((center - size / 2, center + size / 2), axis=1)
    points = values[:, 5:].reshape(-1, 2, 3).copy()
    points[..., :2] *= [width, height]
    return boxes, points


def box_iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    area_a = np.maximum(0, a[2:] - a[:2]).prod()
    area_b = np.maximum(0, b[2:] - b[:2]).prod()
    return float(intersection / max(area_a + area_b - intersection, 1e-9))


def match_boxes(gt_boxes, pred_boxes, threshold):
    candidates = sorted(
        ((box_iou(a, b), gi, pi) for gi, a in enumerate(gt_boxes)
         for pi, b in enumerate(pred_boxes)), reverse=True
    )
    matches, used_pred = {}, set()
    for overlap, gi, pi in candidates:
        if overlap < threshold:
            break
        if gi not in matches and pi not in used_pred:
            matches[gi] = pi
            used_pred.add(pi)
    return matches


def localization_errors(gt_boxes, gt_points, pred_boxes, pred_points, threshold=0.5):
    matches = match_boxes(gt_boxes, pred_boxes, threshold)
    distances, angles = [], []
    for gi, pi in matches.items():
        gt = gt_points[gi]
        predicted = pred_points[pi, :, :2]
        visible = gt[:, 2] > 0
        # The axis is undirected: use the endpoint correspondence with the
        # smaller total pixel distance, rather than a fixed predicted order.
        direct = np.linalg.norm(predicted - gt[:, :2], axis=1)
        reverse = np.linalg.norm(predicted[::-1] - gt[:, :2], axis=1)
        if reverse[visible].sum() < direct[visible].sum():
            predicted = predicted[::-1]
        valid = visible & np.isfinite(predicted).all(axis=1)
        diagonal = max(float(np.linalg.norm(gt_boxes[gi, 2:] - gt_boxes[gi, :2])), 1.0)
        distances.extend((np.linalg.norm(predicted[valid] - gt[valid, :2], axis=1) / diagonal).tolist())
        if visible.all() and np.isfinite(predicted).all():
            p_axis = predicted[1] - predicted[0]
            g_axis = gt[1, :2] - gt[0, :2]
            p_length, g_length = np.linalg.norm(p_axis), np.linalg.norm(g_axis)
            if p_length > 1e-6 and g_length > 1e-6:
                cosine = np.clip(abs(np.dot(p_axis / p_length, g_axis / g_length)), 0.0, 1.0)
                angles.append(float(np.degrees(np.arccos(cosine))))
    return matches, distances, angles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/scwpose.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/test",
                        help="Directory containing images/ and labels/.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or 0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.55, help="NMS IoU threshold")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    images = sorted(p for p in (args.data / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"No images in {args.data / 'images'}")
    predictor = SCWPredictor(checkpoint=args.weights, device=args.device)
    rows, distances, angles = [], [], []
    for index, path in enumerate(images, 1):
        frame = read_image(path)
        gt_boxes, gt_points = read_labels(args.data / "labels" / (path.stem + ".txt"),
                                          frame.shape[1], frame.shape[0])
        result = predictor.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou)[0]
        pb = result.boxes.xyxy.cpu().numpy().astype(np.float64)
        pp = result.keypoints.data.cpu().numpy().astype(np.float64)
        matches, de, ae = localization_errors(gt_boxes, gt_points, pb, pp, args.match_iou)
        distances.extend(de)
        angles.extend(ae)
        rows.append(dict(image=path.name, gt_instances=len(gt_boxes),
                         predictions=len(pb), matched_instances=len(matches),
                         normalized_endpoint_errors=de, angular_errors_degrees=ae))
        if index % 20 == 0 or index == len(images):
            print(f"Evaluated {index}/{len(images)} images", flush=True)
    d, a = np.asarray(distances), np.asarray(angles)
    metrics = dict(
        images=len(images), gt_instances=sum(r["gt_instances"] for r in rows),
        predictions=sum(r["predictions"] for r in rows),
        matched_instances=sum(r["matched_instances"] for r in rows),
        evaluated_endpoints=len(d), evaluated_axes=len(a),
        axis_mae_deg=float(a.mean()) if len(a) else None,
        axis_rmse_deg=float(np.sqrt(np.mean(a ** 2))) if len(a) else None,
        axis_p95_deg=float(np.percentile(a, 95)) if len(a) else None,
        endpoint_nme_percent=float(d.mean() * 100) if len(d) else None,
        pck_005_percent=float((d < 0.05).mean() * 100) if len(d) else None,
    )
    report = dict(metrics=metrics, settings=dict(imgsz=args.imgsz, confidence=args.conf,
                  nms_iou=args.iou, matching_iou=args.match_iou,
                  device=str(next(predictor.model.parameters()).device)), images=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
