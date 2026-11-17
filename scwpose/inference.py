"""Single-image BGR inference and optional instance-mask reconstruction."""

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

from .model import SCWPose, expanded_rois


class SCWPredictor:
    """Predict full SCW-Pose endpoints on uint8 OpenCV BGR images.

    Device selection occurs when loading the model. Returned Results contain
    original-image pixel coordinates, boxes, and two keypoints per instance.
    """

    def __init__(self, checkpoint=None, model=None, device="auto"):
        if (checkpoint is None) == (model is None):
            raise ValueError("Provide exactly one of checkpoint or model")
        self.model = model if model is not None else SCWPose.load(checkpoint, device=device)
        self.model.eval()
        self.last_rois = self.last_masks = self.last_constrained = self.last_alpha = None

    @torch.inference_mode()
    def predict(self, frame, imgsz=1280, conf=0.3, iou=0.55, max_det=300):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an H x W x 3 BGR numpy array")
        if frame.dtype != np.uint8 or min(frame.shape[:2]) == 0:
            raise ValueError("frame must be a nonempty uint8 BGR image")
        self.model.eval()
        prepared = LetterBox((imgsz, imgsz), auto=True, stride=32)(image=frame)
        tensor = torch.from_numpy(np.ascontiguousarray(prepared[..., ::-1].transpose(2, 0, 1)))
        tensor = tensor.to(next(self.model.parameters()).device).float()[None] / 255
        decoded = self.model(tensor)[0]
        if decoded.shape[1] != 11:
            raise RuntimeError("Expected one walnut class and two three-component keypoints")
        pred = non_max_suppression(decoded, conf_thres=conf, iou_thres=iou,
                                   nc=1, max_det=max_det)[0]
        refined = pred[:, 6:].reshape(-1, 2, 3).clone()
        self.last_rois = self.last_masks = self.last_constrained = self.last_alpha = None
        if len(pred):
            rois = expanded_rois(pred[:, :4], pred.new_zeros(len(pred)), tensor.shape[-2:])
            out = self.model.refine(rois, refined)
            refined[..., :2] = out["points"] * (rois[:, 3:5] - rois[:, 1:3])[:, None] + rois[:, 1:3][:, None]
            self.last_masks = out["mask"][:, 0].cpu().numpy()
            self.last_alpha = out["alpha"].cpu().numpy()
            self.last_constrained = (out["alpha"] > 0).cpu().numpy()
            gain = min(tensor.shape[-2] / frame.shape[0], tensor.shape[-1] / frame.shape[1])
            pad = np.array([round((tensor.shape[-1] - frame.shape[1] * gain) / 2 - 0.1),
                            round((tensor.shape[-2] - frame.shape[0] * gain) / 2 - 0.1)])
            self.last_rois = (rois[:, 1:].cpu().numpy() - np.tile(pad, 2)) / gain
            ops.scale_coords(tensor.shape[-2:], refined[..., :2], frame.shape)
            ops.scale_boxes(tensor.shape[-2:], pred[:, :4], frame.shape)
        return [Results(frame, path="", names={0: "walnut"},
                        boxes=pred[:, :6], keypoints=refined)]

    def masks_original(self, shape):
        """Return one H x W probability mask per detected walnut."""
        result = []
        if self.last_masks is None:
            return result
        h, w = shape[:2]
        for box, probability in zip(self.last_rois, self.last_masks):
            x1, y1 = np.floor(box[:2]).astype(int)
            x2, y2 = np.ceil(box[2:]).astype(int)
            resized = cv2.resize(probability, (max(1, x2 - x1), max(1, y2 - y1)))
            canvas = np.zeros((h, w), dtype=np.float32)
            a, b, c, d = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if c > a and d > b:
                canvas[b:d, a:c] = resized[b-y1:d-y1, a-x1:c-x1]
            result.append(canvas)
        return result


def predictor(model):
    return SCWPredictor(model=model)
