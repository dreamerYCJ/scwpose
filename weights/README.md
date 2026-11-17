# SCW-Pose weights

`scwpose.pt` contains the complete trained model: the YOLO26n-Pose backbone,
semantic refinement modules, and dual-gated contour correction modules.
It includes the architecture dictionary and all 923 state entries; no separate
baseline checkpoint or architecture file is required.

Load it with `SCWPose.load()` or `SCWPredictor`, as used by `evaluate.py`
and `predict.py`. It is a tensor checkpoint, not an `ultralytics.YOLO()` export.

- Parameters: 2,816,282.
- Format: `scwpose_v1`.
- SHA-256: `4da005e9cd1220985e8f230144a93ddc8b66e1994a2996d8c6b60554c94d55bb`.

The published checkpoint preserves every trained tensor and removes
machine-specific training metadata.
