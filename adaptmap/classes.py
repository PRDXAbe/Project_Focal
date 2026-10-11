"""AdaptMap semantic classes and the SemanticKITTI -> AdaptMap mapping."""
import torch

# id, name, RGB, dynamic
ADAPT_CLASSES = [
    (0, "unknown",        (128, 128, 128), False),
    (1, "road",           (255, 0, 255),   False),
    (2, "sidewalk_curb",  (75, 0, 75),     False),
    (3, "terrain",        (150, 240, 80),  False),
    (4, "vegetation",     (0, 175, 0),     False),
    (5, "building_wall",  (255, 200, 0),   False),
    (6, "pole",           (255, 240, 150), False),
    (7, "vehicle",        (100, 150, 245), True),
    (8, "pedestrian",     (255, 30, 30),   True),
    (9, "other_obstacle", (255, 120, 50),  False),
]
NUM_CLASSES = len(ADAPT_CLASSES)
NAMES = [c[1] for c in ADAPT_CLASSES]
COLORS = torch.tensor([c[2] for c in ADAPT_CLASSES], dtype=torch.uint8)
DYNAMIC = torch.tensor([c[3] for c in ADAPT_CLASSES])
GROUND = torch.tensor([n in ("road", "sidewalk_curb", "terrain") for n in NAMES])

# SemanticKITTI 19 train classes in mmdet3d / standard learning-map order.
KITTI_CLASSES = [
    "car", "bicycle", "motorcycle", "truck", "bus", "person", "bicyclist",
    "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building",
    "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
]
_KITTI_TO_ADAPT = {
    "car": "vehicle", "bicycle": "vehicle", "motorcycle": "vehicle",
    "truck": "vehicle", "bus": "vehicle", "person": "pedestrian",
    "bicyclist": "pedestrian", "motorcyclist": "pedestrian", "road": "road",
    "parking": "road", "sidewalk": "sidewalk_curb", "other-ground": "terrain",
    "building": "building_wall", "fence": "building_wall",
    "vegetation": "vegetation", "trunk": "pole", "terrain": "terrain",
    "pole": "pole", "traffic-sign": "pole",
}
KITTI_TO_ADAPT = torch.tensor([NAMES.index(_KITTI_TO_ADAPT[k]) for k in KITTI_CLASSES])


def kitti_probs_to_adapt(probs):
    """Sum (N, 19) SemanticKITTI softmax into (N, NUM_CLASSES) AdaptMap probs."""
    out = torch.zeros(probs.shape[0], NUM_CLASSES, device=probs.device, dtype=probs.dtype)
    out.index_add_(1, KITTI_TO_ADAPT.to(probs.device), probs)
    return out
