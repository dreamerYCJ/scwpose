"""SCW-Pose model for inference with the released full-model checkpoint."""

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics.nn.tasks import PoseModel


CHECKPOINT_FORMAT = "scwpose_v1"
SUPPORTED_FORMATS = (CHECKPOINT_FORMAT, "walnut_yolo26_A_v1")
RADIUS = 0.08


def resolve_device(device="auto"):
    """Resolve 'auto', 'cpu', CUDA device strings, or an integer GPU index."""
    if device is None or str(device) == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int) or str(device).isdigit():
        device = f"cuda:{device}"
    return torch.device(device)


def pixel_grid(size, device, dtype=torch.float32):
    values = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size
    y, x = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((x, y), -1)


def expected_points(logits):
    probability = logits.float().flatten(2).softmax(-1)
    grid = pixel_grid(logits.shape[-1], logits.device).reshape(-1, 2)
    return probability @ grid


def soft_boundary(probability):
    return (F.max_pool2d(probability, 3, 1, 1)
            + F.max_pool2d(-probability, 3, 1, 1)).clamp_min(0)


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                         nn.GroupNorm(8, cout), nn.SiLU())


def expanded_rois(boxes, batch_indices, image_shape):
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(4) * 1.12
    xy1, xy2 = center - wh / 2, center + wh / 2
    limit = boxes.new_tensor([image_shape[1], image_shape[0]])
    xy1 = torch.minimum(xy1.clamp_min(0), limit - 1)
    xy2 = torch.minimum(torch.maximum(xy2, xy1 + 1), limit)
    return torch.cat((batch_indices[:, None].float(), xy1, xy2), 1)


class RefinementHead(nn.Module):
    """Semantic endpoint refinement and bounded dual-endpoint correction."""

    def __init__(self):
        super().__init__()
        self.size = 64
        self.seg = nn.Sequential(block(64, 32), block(32, 32))
        self.mask = nn.Conv2d(32, 1, 1)
        self.pose = nn.Sequential(block(64 + 16 + 4, 32), block(32, 32))
        self.fusion = nn.Sequential(block(32 + 32 + 1 + 1 + 2, 32), block(32, 32))
        self.residual = nn.Conv2d(32, 2, 1)
        # Retained solely for strict compatibility with the released state dictionary.
        self.gate = nn.Linear(32, 2)
        self.gate.requires_grad_(False)
        self.accept = nn.Sequential(nn.Linear(69, 32), nn.SiLU(), nn.Linear(32, 1))
        self.joint_accept = nn.Sequential(nn.Linear(14, 32), nn.SiLU(), nn.Linear(32, 2))

    def forward(self, shared, pose, coarse):
        n = len(shared)
        grid = pixel_grid(self.size, shared.device, shared.dtype)
        prior_logits = -(grid[None, None] - coarse[:, :, None, None]).square().sum(-1) / (2 * 0.14**2)
        priors = prior_logits.exp()
        coords = grid.permute(2, 0, 1)[None].expand(n, -1, -1, -1)
        seg = self.seg(shared)
        mask_logits = self.mask(seg)
        mask = mask_logits.sigmoid()
        boundary = soft_boundary(mask)
        pose_features = self.pose(torch.cat((shared, pose, priors, coords), 1))
        fused = self.fusion(torch.cat((pose_features, seg, mask, boundary, priors), 1))
        semantic = prior_logits + self.residual(fused)
        semantic_points = expected_points(semantic)

        offset = grid[None, None] - semantic_points.detach()[:, :, None, None]
        dist2 = offset.square().sum(-1)
        local = dist2 <= RADIUS**2
        boundary_support = (boundary.expand(-1, 2, -1, -1) * local).amax((-1, -2))
        occupancy = mask.mean((1, 2, 3))
        valid = ((boundary_support >= 0.1) & local.any((-1, -2))
                 & (occupancy[:, None] >= 0.02) & (occupancy[:, None] <= 0.97))
        logits = semantic + boundary.clamp_min(1e-5).log() - dist2 / (2 * (RADIUS / 2)**2)
        logits = logits.masked_fill(~local, -1e4)
        candidate = expected_points(logits)
        candidate = torch.where(valid[..., None], candidate, semantic_points)

        def sample(points):
            return F.grid_sample(fused, points[:, None] * 2 - 1,
                                 align_corners=False, padding_mode="border")[:, :, 0].transpose(1, 2)

        delta = candidate - semantic_points
        distance = delta.norm(dim=-1)
        extra = torch.cat((delta, distance[..., None], boundary_support[..., None],
                           valid.float()[..., None]), -1)
        local_gate = self.accept(torch.cat((sample(semantic_points), sample(candidate), extra), -1))[..., 0]
        pair = semantic_points[:, 1] - semantic_points[:, 0]
        length = pair.norm(dim=-1, keepdim=True)
        context = torch.cat((semantic_points.flatten(1), delta.flatten(1), pair, length,
                             valid.float(), mask.mean((1, 2, 3))[:, None]), 1)
        gate_logits = local_gate + self.joint_accept(context)
        alpha = gate_logits.sigmoid() * valid.float()
        points = semantic_points + alpha[..., None] * delta
        return dict(mask_logits=mask_logits, mask=mask, boundary=boundary,
                    semantic=semantic, semantic_points=semantic_points,
                    candidate=candidate, valid_candidate=valid,
                    gate_logits=gate_logits, alpha=alpha, points=points)


class SCWPose(nn.Module):
    """Complete backbone and refinement network; no baseline weights are needed."""

    def __init__(self, base_yaml):
        super().__init__()
        self.base = PoseModel(cfg=base_yaml, ch=3, nc=1, data_kpt_shape=(2, 3), verbose=False)
        pose_head = self.base.model[-1]
        if type(pose_head).__name__ != "Pose26" or pose_head.end2end:
            raise ValueError("SCW-Pose requires the YOLO26 Pose26 head with end2end=False")
        self.head = RefinementHead()
        self.detail = nn.Sequential(block(64, 32), nn.Conv2d(32, 64, 1))
        self.shared_feature = self.pose_feature = self.detail_feature = None
        self._hooks = [pose_head.register_forward_pre_hook(self._capture_shared),
                       pose_head.cv4_kpts[0].register_forward_pre_hook(self._capture_pose),
                       self.base.model[2].register_forward_hook(self._capture_detail)]
        self.metadata = {}

    def _capture_shared(self, module, inputs):
        self.shared_feature = inputs[0][0]

    def _capture_pose(self, module, inputs):
        self.pose_feature = inputs[0]

    def _capture_detail(self, module, inputs, output):
        self.detail_feature = output

    def forward(self, images):
        return self.base(images)

    def refine(self, rois, coarse_pixels):
        origin = rois[:, 1:3]
        wh = (rois[:, 3:5] - origin).clamp_min(1)
        coarse = (coarse_pixels[..., :2] - origin[:, None]) / wh[:, None]
        shared = roi_align(self.shared_feature.float(), rois.float(), 64,
                           spatial_scale=1 / 8, sampling_ratio=2, aligned=True)
        pose = roi_align(self.pose_feature.float(), rois.float(), 64,
                         spatial_scale=1 / 8, sampling_ratio=2, aligned=True)
        fine = roi_align(self.detail_feature.float(), rois.float(), 64,
                         spatial_scale=1 / 4, sampling_ratio=2, aligned=True)
        return self.head(shared + self.detail(fine), pose, coarse.detach())

    @classmethod
    def load(cls, path, device="auto"):
        """Load the complete tensor checkpoint without importing experiment classes."""
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        if data.get("format") not in SUPPORTED_FORMATS or data.get("arm", "A") != "A":
            raise ValueError("Expected a full SCW-Pose checkpoint")
        if not isinstance(data.get("base_yaml"), dict):
            raise ValueError("The checkpoint must contain the model architecture dictionary")
        model = cls(data["base_yaml"])
        model.load_state_dict(data["state"], strict=True)
        model.metadata = data.get("metadata", {})
        return model.to(resolve_device(device)).eval()


def load_model(path, device="auto"):
    return SCWPose.load(path, device=device)
