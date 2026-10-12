"""Semantic segmentation backends behind one interface: segment(points) -> (labels, probs).

points: (N, 4) [x, y, z, intensity] in the sensor frame. Returns AdaptMap labels (N,)
and AdaptMap class probabilities (N, NUM_CLASSES).
"""
import torch

from . import classes


class MinkUNetSegmenter:
    """Sparse 3D CNN (MinkUNet, spconv backend) pretrained on SemanticKITTI, via mmdet3d.

    half=True runs the network in fp16 (spconv supports it natively): ~2.4x faster on an RTX 4050
    with 99.95 % of point labels unchanged. The head may output the 19 SemanticKITTI classes (mapped
    to AdaptMap in probability space) or AdaptMap's classes directly (fine-tuned "adapt10" head).
    """

    def __init__(self, config, checkpoint, device="cuda", half=True):
        from mmdet3d.apis import init_model
        from mmdet3d.structures import Det3DDataSample, PointData

        self.model = init_model(config, checkpoint, device="cuda:0" if device == "cuda" else device)
        self.model.eval()
        self.half = half and device != "cpu"
        if self.half:
            self.model.half()
        self.device = device
        self._sample_cls, self._point_data = Det3DDataSample, PointData

    @torch.no_grad()
    def segment(self, points):
        pts = points.to(self.device, torch.float32)
        sample = self._sample_cls()
        sample.gt_pts_seg = self._point_data()
        data = self.model.data_preprocessor({"inputs": {"points": [pts]}, "data_samples": [sample]}, False)
        if self.half:
            data["inputs"]["voxels"]["voxels"] = data["inputs"]["voxels"]["voxels"].half()
        feats = self.model.extract_feat(data["inputs"])
        logits = self.model.decode_head.predict(feats, data["data_samples"])[0]  # (N, C) per point
        probs = torch.softmax(logits.float(), dim=1)
        if probs.shape[1] != classes.NUM_CLASSES:
            probs = classes.kitti_probs_to_adapt(probs)
        return probs.argmax(1), probs


class GeometricSegmenter:
    """Model-free fallback: ground vs obstacle by height above the local minimum.

    Used for tests / when no checkpoint is available. Not a learned model.
    """

    def __init__(self, sensor_height=1.73, cell=0.5, ground_tol=0.25, device="cuda"):
        self.h, self.cell, self.tol, self.device = sensor_height, cell, ground_tol, device

    @torch.no_grad()
    def segment(self, points):
        xyz = points[:, :3].to(self.device)
        ij = torch.floor(xyz[:, :2] / self.cell).long()
        ij -= ij.min(0).values
        key = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
        _, inv = torch.unique(key, return_inverse=True)
        zmin = torch.full((int(inv.max()) + 1,), float("inf"), device=self.device)
        zmin.scatter_reduce_(0, inv, xyz[:, 2], "amin")
        above = xyz[:, 2] - zmin[inv]
        labels = torch.full((xyz.shape[0],), classes.NAMES.index("other_obstacle"), device=self.device)
        labels[above < self.tol] = classes.NAMES.index("road")
        probs = torch.nn.functional.one_hot(labels, classes.NUM_CLASSES).float() * 0.6 + 0.04
        return labels, probs


def build_segmenter(cfg):
    m = cfg["model"]
    if m["backend"] == "mmdet3d_minkunet":
        return MinkUNetSegmenter(m["config"], m["checkpoint"], cfg["device"], half=m.get("half", True))
    return GeometricSegmenter(device=cfg["device"])
