"""Torchvision training core for the Train Model operator.

Owns the torchvision-specific work for classification, detection, and semantic
segmentation: the per-task head swap, a shared ``step_fn``-parametrized training
loop, the FiftyOne -> PyTorch bridge, and a ``record_run`` spec whose ``fit``
trains and yields an in-memory model wrapped for inference.

Inference rides FiftyOne's native :class:`fiftyone.utils.torch.TorchImageModel`
plus the matching built-in ``OutputProcessor`` (classifier / detector /
semantic-segmenter), so there is no hand-rolled decode: the just-trained net is
wrapped via the config's direct ``model`` slot and applied with the same
weights, no reload. Detection feeds the model a raw list of tensors (a tiny
``_forward_pass`` override moves them to the device); segmentation adds a thin
output-processor subclass that maps contiguous channel ids back to the dataset's
``mask_targets`` ids.

The form scaffolding, view resolution, checkpoint placement, and engine calls
come from ``training_helpers``. Only this core's three tasks are exposed; the
roster is deliberately one model per task (the breadth-across-tasks first cut).
"""

import importlib.util
import os

import fiftyone.core.labels as fol
import fiftyone.core.storage as fos
import fiftyone.operators.types as types
import fiftyone.utils.torch as fout_torch
import fiftyone.utils.training as fout

from . import training_helpers as th

# torchvision is an optional dependency; check without importing the heavy
# package at plugin-load time (the real imports happen in the fit closures).
HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None

# Ground-truth label types accepted per task (read by the form to filter the
# label field + framework choices). Segmentation is restricted to semantic
# ``Segmentation`` masks so the welded eval (evaluate_segmentations) compares
# like-for-like.
TASK_LABEL_TYPES = {
    "detection": (fol.Detections,),
    "classification": (fol.Classification,),
    "segmentation": (fol.Segmentation,),
}

# One model per task (closed set, rendered as a dropdown). The roster is
# intentionally small -- the picks maximize architectural spread while keeping
# the implementation to the minimum needed to prove the framework lands the
# same run record across all three tasks. Faster R-CNN is chosen for detection
# because its N+1 / background-at-0 label convention is the well-documented one.
_TV_MODELS = {
    "classification": ["resnet50"],
    "detection": ["fasterrcnn_resnet50_fpn_v2"],
    "segmentation": ["deeplabv3_resnet50"],
}

# ImageNet normalization (segmentation models expect it; classification uses the
# weights' own transforms, which already normalize).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Fixed square the segmentation image + mask are resized to (the model upsamples
# its output to the input size; the seg eval resizes the prediction back to the
# ground-truth size, so this only needs to let batches stack).
_SEG_SIZE = 520


def tv_model_inputs(inputs, ctx, task):
    """Add the torchvision model + hyperparameter fields for the chosen ``task``
    (task / label field / splits are collected by the operator). Returns
    ``False`` if a prerequisite is missing (e.g. segmentation mask_targets)."""
    if not HAS_TORCHVISION:
        inputs.view(
            "tv_error",
            types.Error(
                label="torchvision not installed",
                description="Install it with: pip install torch torchvision",
            ),
        )
        return False

    if task not in _TV_MODELS:
        inputs.view(
            "tv_task_error",
            types.Error(
                label="Unsupported task",
                description=f"The torchvision core can't train '{task}'",
            ),
        )
        return False

    if task == "segmentation" and not _mask_targets(
        ctx.dataset, ctx.params.get("label_field")
    ):
        inputs.view(
            "seg_warn",
            types.Warning(
                label="No mask_targets set",
                description="Set dataset.mask_targets so classes can be mapped",
            ),
        )
        return False

    models = _TV_MODELS[task]
    inputs.enum(
        "model_name",
        models,
        default=models[0],
        required=True,
        view=types.DropdownView(),
        label="Torchvision model",
    )

    if task == "detection":
        inputs.float(
            "confidence",
            default=0.25,
            label="Confidence threshold",
            description="Minimum confidence for the predictions written back",
        )

    inputs.int("epochs", default=5, label="Epochs")
    inputs.int("batch_size", default=4, label="Batch size")
    inputs.float("learning_rate", default=1e-4, label="Learning rate")
    inputs.bool(
        "freeze_backbone",
        default=True,
        label="Freeze backbone",
        description="Train only the swapped head (faster; recommended)",
        view=types.CheckboxView(),
    )
    return True


def train_torchvision(ctx, train_key):
    """Build the ``record_run`` spec for a torchvision run."""
    task = ctx.params.get("task", "classification")
    if task == "detection":
        return _train_detection(ctx)
    if task == "segmentation":
        return _train_segmentation(ctx)
    return _train_classification(ctx)


# --- shared core helpers ---------------------------------------------------


def _config(ctx, task):
    """The opaque ``train_config`` recorded on the run."""
    cfg = {
        "model": ctx.params["model_name"],
        "task": task,
        "epochs": ctx.params.get("epochs", 5),
        "batch_size": ctx.params.get("batch_size", 4),
        "learning_rate": ctx.params.get("learning_rate", 1e-4),
        "freeze_backbone": bool(ctx.params.get("freeze_backbone", True)),
    }
    if ctx.user_id:
        cfg["triggered_by"] = ctx.user_id
    return cfg


def _spec(ctx, train_config, splits, fit):
    """Assemble the ``record_run`` spec shared by every torchvision task."""
    train_arg, val_arg, test_arg = splits
    return dict(
        train_arg=train_arg,
        val_arg=val_arg,
        test_arg=test_arg,
        gt_field=ctx.params["label_field"],
        pred_field=ctx.params.get("pred_field") or "torchvision_predictions",
        auto_eval=None,
        project_url=ctx.params.get("project_url"),
        train_config=train_config,
        fit=fit,
    )


def _mask_targets(dataset, label_field):
    if not label_field:
        return None
    targets = dataset.mask_targets.get(label_field) if dataset.mask_targets else None
    return targets or dataset.default_mask_targets or None


def _union_classes(ctx, path, *args):
    """Sorted, non-null class names across every provided split.

    A class present only in val/test (but not train) must still be known, or
    the model's class space wouldn't cover predictions made on those splits.
    """
    classes = set()
    for arg in args:
        view = fout.resolve_view(ctx.dataset, arg)
        if view is not None:
            classes.update(v for v in view.distinct(path) if v is not None)
    return sorted(classes)


def _loader(ctx, view, get_item, collate_fn=None):
    """A training DataLoader over ``view`` using the form's batch size."""
    import torch

    return torch.utils.data.DataLoader(
        view.to_torch(get_item),
        batch_size=ctx.params.get("batch_size", 4),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _apply_freeze(model, head_modules):
    """Freeze every parameter, then re-enable grad on the swapped head(s)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for head in head_modules:
        for p in head.parameters():
            p.requires_grad_(True)


def _run_training(ctx, model, loader, step_fn):
    """One generic loop shared by all tasks. Reads epochs/lr from the form and
    picks the device; freezing is just a flag upstream, since the optimizer
    only sees ``requires_grad`` params."""
    import torch

    device = _device()
    epochs = max(ctx.params.get("epochs", 5), 1)
    lr = ctx.params.get("learning_rate", 1e-4)

    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            loss = step_fn(model, batch, device)
            loss.backward()
            opt.step()
        sched.step()


def _save_checkpoint(ctx, model, metadata):
    """Save ``state_dict`` + rebuild metadata to a single ``.pt`` and place it
    (local or cloud); return its URI. The metadata is everything a future
    reload path needs to reconstruct the architecture and label space."""
    import torch

    save_dir = fos.make_temp_dir()
    path = os.path.join(save_dir, "best.pt")
    torch.save({"state_dict": model.state_dict(), **metadata}, path)
    return th.place_checkpoint_file(ctx, path)


# --- classification --------------------------------------------------------


def _swap_classifier_head(model, num_classes):
    """Replace the classifier head to emit ``num_classes`` and return it (for
    selective unfreezing). Covers ResNet (``.fc``), ViT (``.heads.head``), and
    ConvNeXt/EfficientNet/MobileNet (last ``Linear`` in ``.classifier``)."""
    import torch.nn as nn

    if isinstance(getattr(model, "fc", None), nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return [model.fc]

    heads = getattr(model, "heads", None)
    if heads is not None and isinstance(getattr(heads, "head", None), nn.Linear):
        heads.head = nn.Linear(heads.head.in_features, num_classes)
        return [heads.head]

    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        for i in reversed(range(len(classifier))):
            if isinstance(classifier[i], nn.Linear):
                classifier[i] = nn.Linear(
                    classifier[i].in_features, num_classes
                )
                return [classifier[i]]

    raise ValueError("Could not locate a classifier head to replace")


def _train_classification(ctx):
    label_field = ctx.params["label_field"]
    train_arg, val_arg, test_arg = th.resolve_split_args(ctx)
    train_config = _config(ctx, "classification")

    def fit():
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision.models import get_model, get_model_weights
        from fiftyone.utils.torch import GetItem

        model_name = ctx.params["model_name"]
        classes = _union_classes(
            ctx, f"{label_field}.label", train_arg, val_arg, test_arg
        )
        if len(classes) < 2:
            raise ValueError(
                f"Need at least 2 classes to fine-tune, found {len(classes)} "
                f"in '{label_field}'"
            )
        class2idx = {c: i for i, c in enumerate(classes)}

        weights = get_model_weights(model_name).DEFAULT
        transform = weights.transforms()
        model = get_model(model_name, weights=weights)
        head = _swap_classifier_head(model, len(classes))
        if ctx.params.get("freeze_backbone", True):
            _apply_freeze(model, head)

        class _Get(GetItem):
            def __init__(self):
                super().__init__(field_mapping={"classification": label_field})

            @property
            def required_keys(self):
                return ["filepath", "classification"]

            def __call__(self, d):
                image = Image.open(d["filepath"]).convert("RGB")
                return (
                    transform(image),
                    torch.tensor(
                        class2idx[d["classification"].label], dtype=torch.long
                    ),
                )

        def step_fn(model, batch, device):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            return F.cross_entropy(model(images), labels)

        train_view = fout.resolve_view(ctx.dataset, train_arg)
        loader = _loader(ctx, train_view.exists(f"{label_field}.label"), _Get())
        _run_training(ctx, model, loader, step_fn)

        checkpoint_uri = _save_checkpoint(
            ctx,
            model,
            {"task": "classification", "arch": model_name, "classes": classes},
        )

        config = fout_torch.TorchImageModelConfig(
            {
                "model": model.eval(),
                "transforms": transform,
                "ragged_batches": False,
                "output_processor_cls": (
                    "fiftyone.utils.torch.ClassifierOutputProcessor"
                ),
                "classes": classes,
            }
        )
        return fout_torch.TorchImageModel(config), checkpoint_uri, None

    return _spec(ctx, train_config, (train_arg, val_arg, test_arg), fit)


# --- detection -------------------------------------------------------------


def _swap_detection_head(model, num_classes):
    """Faster R-CNN head swap. ``num_classes`` INCLUDES background (index 0),
    so foreground labels are ``1..N``. Returns the new head for unfreezing."""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return [model.roi_heads.box_predictor]


class _TVDetectionModel(fout_torch.TorchImageModel):
    """Torchvision detectors take a raw ``List[Tensor[C, H, W]]`` and return a
    list of prediction dicts. The base ``_forward_pass`` doesn't move raw-input
    lists to the device (only the stacked path does), so move them here."""

    def _forward_pass(self, imgs):
        imgs = [img.to(self._device) for img in imgs]
        return self._model(imgs)


def _train_detection(ctx):
    label_field = ctx.params["label_field"]
    train_arg, val_arg, test_arg = th.resolve_split_args(ctx)
    train_config = _config(ctx, "detection")

    def fit():
        import torch
        from PIL import Image
        from torchvision.models import get_model, get_model_weights
        from torchvision.transforms import ToTensor
        from fiftyone.utils.torch import GetItem

        model_name = ctx.params["model_name"]
        classes = _union_classes(
            ctx, f"{label_field}.detections.label", train_arg, val_arg, test_arg
        )
        if not classes:
            raise ValueError(f"No detection labels found in '{label_field}'")
        # Foreground labels are 1..N; index 0 is background.
        class2idx = {c: i + 1 for i, c in enumerate(classes)}
        # The decode classes list is index-aligned: classes[label] is the name,
        # with a placeholder at 0 the model never emits as a kept detection.
        decode_classes = ["__background__"] + classes

        weights = get_model_weights(model_name).DEFAULT
        model = get_model(model_name, weights=weights)
        head = _swap_detection_head(model, len(classes) + 1)
        if ctx.params.get("freeze_backbone", True):
            _apply_freeze(model, head)

        to_tensor = ToTensor()

        class _Get(GetItem):
            def __init__(self):
                super().__init__(field_mapping={"detections": label_field})

            @property
            def required_keys(self):
                return ["filepath", "detections"]

            def __call__(self, d):
                image = Image.open(d["filepath"]).convert("RGB")
                img = to_tensor(image)  # CHW float [0, 1], original size
                _, h, w = img.shape
                boxes, labels = [], []
                for det in d["detections"].detections:
                    rx, ry, rw, rh = det.bounding_box
                    boxes.append([rx * w, ry * h, (rx + rw) * w, (ry + rh) * h])
                    labels.append(class2idx[det.label])
                target = {
                    "boxes": torch.tensor(
                        boxes, dtype=torch.float32
                    ).reshape(-1, 4),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
                return img, target

        def _collate(batch):
            return tuple(zip(*batch))

        def step_fn(model, batch, device):
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]
            loss_dict = model(images, targets)
            return sum(loss_dict.values())

        train_view = fout.resolve_view(ctx.dataset, train_arg)
        loader = _loader(
            ctx,
            train_view.exists(f"{label_field}.detections"),
            _Get(),
            collate_fn=_collate,
        )
        _run_training(ctx, model, loader, step_fn)

        checkpoint_uri = _save_checkpoint(
            ctx,
            model,
            {"task": "detection", "arch": model_name, "classes": classes},
        )

        config = fout_torch.TorchImageModelConfig(
            {
                "model": model.eval(),
                "transforms": to_tensor,
                "raw_inputs": True,
                "ragged_batches": True,
                "output_processor_cls": (
                    "fiftyone.utils.torch.DetectorOutputProcessor"
                ),
                "classes": decode_classes,
                "confidence_thresh": ctx.params.get("confidence", 0.25),
            }
        )
        # batch_size=1 keeps the per-image frame_size correct (the output
        # processor reads it from the first image of each batch).
        return _TVDetectionModel(config), checkpoint_uri, {"batch_size": 1}

    return _spec(ctx, train_config, (train_arg, val_arg, test_arg), fit)


# --- semantic segmentation -------------------------------------------------


def _swap_seg_head(model, num_classes):
    """Replace the segmentation head (and aux head, if present) to emit
    ``num_classes`` channels; return the head module(s) for unfreezing."""
    import torch.nn as nn

    heads = []
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    heads.append(model.classifier)
    aux = getattr(model, "aux_classifier", None)
    if aux is not None:
        in_aux = aux[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_aux, num_classes, kernel_size=1)
        heads.append(model.aux_classifier)
    return heads


class _SegOutputProcessor(fout_torch.SemanticSegmenterOutputProcessor):
    """Maps the model's contiguous channel ids (0..C-1) back to the dataset's
    original ``mask_targets`` ids, so the welded eval compares like-for-like.
    The base processor's argmax is at the model's input size; the segmentation
    eval resizes the prediction to the ground-truth size, so no resize here."""

    def __init__(self, lut, **kwargs):
        super().__init__(**kwargs)
        self._lut = lut

    def __call__(self, output, *args, **kwargs):
        segs = super().__call__(output, *args, **kwargs)
        for seg in segs:
            seg.mask = self._lut[seg.mask]
        return segs


def _train_segmentation(ctx):
    label_field = ctx.params["label_field"]
    train_arg, val_arg, test_arg = th.resolve_split_args(ctx)
    train_config = _config(ctx, "segmentation")

    def fit():
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision.models import get_model, get_model_weights
        import torchvision.transforms as T
        from fiftyone.utils.torch import GetItem

        model_name = ctx.params["model_name"]
        targets = _mask_targets(ctx.dataset, label_field)
        if not targets:
            raise ValueError(
                f"No mask_targets found for '{label_field}'. Set "
                "dataset.mask_targets so classes can be mapped."
            )
        # Original (possibly sparse) ids -> contiguous 0..C-1 for training, and
        # the inverse lookup table for decoding back to the dataset's ids.
        orig_ids = sorted(int(k) for k in targets)
        orig2contig = {orig: i for i, orig in enumerate(orig_ids)}
        lut = np.array(orig_ids, dtype=np.int64)  # contig -> orig
        num_classes = len(orig_ids)
        if num_classes < 2:
            raise ValueError("Need at least 2 segmentation classes to fine-tune")

        weights = get_model_weights(model_name).DEFAULT
        model = get_model(model_name, weights=weights, aux_loss=True)
        head = _swap_seg_head(model, num_classes)
        if ctx.params.get("freeze_backbone", True):
            _apply_freeze(model, head)

        remap = np.full(max(orig_ids) + 1, 255, dtype=np.int64)
        for orig, contig in orig2contig.items():
            remap[orig] = contig

        image_tf = T.Compose(
            [
                T.Resize((_SEG_SIZE, _SEG_SIZE)),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

        class _Get(GetItem):
            def __init__(self):
                super().__init__(field_mapping={"segmentation": label_field})

            @property
            def required_keys(self):
                return ["filepath", "segmentation"]

            def __call__(self, d):
                image = Image.open(d["filepath"]).convert("RGB")
                mask = _load_seg_mask(d["segmentation"])
                # Clip out-of-range ids before remap, then map to contiguous.
                mask = np.clip(mask, 0, len(remap) - 1)
                mask = remap[mask]
                mask_img = Image.fromarray(mask.astype(np.int32), mode="I")
                mask_img = mask_img.resize(
                    (_SEG_SIZE, _SEG_SIZE), Image.NEAREST
                )
                return (
                    image_tf(image),
                    torch.from_numpy(np.array(mask_img)).long(),
                )

        def step_fn(model, batch, device):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            out = model(images)
            loss = F.cross_entropy(out["out"], masks, ignore_index=255)
            if "aux" in out:
                loss = loss + 0.4 * F.cross_entropy(
                    out["aux"], masks, ignore_index=255
                )
            return loss

        train_view = fout.resolve_view(ctx.dataset, train_arg)
        loader = _loader(ctx, train_view.exists(label_field), _Get())
        _run_training(ctx, model, loader, step_fn)

        checkpoint_uri = _save_checkpoint(
            ctx,
            model,
            {
                "task": "segmentation",
                "arch": model_name,
                "mask_targets": {int(k): v for k, v in targets.items()},
            },
        )

        config = fout_torch.TorchImageModelConfig(
            {
                "model": model.eval(),
                "transforms": image_tf,
                "ragged_batches": False,
                "output_processor": _SegOutputProcessor(lut),
            }
        )
        return fout_torch.TorchImageModel(config), checkpoint_uri, None

    return _spec(ctx, train_config, (train_arg, val_arg, test_arg), fit)


def _load_seg_mask(seg):
    """Class-index mask array for a ``Segmentation`` label (in-memory or disk)."""
    import numpy as np
    from PIL import Image

    if seg.mask is not None:
        mask = seg.mask
        return mask[:, :, 0] if mask.ndim == 3 else mask

    mask_path = getattr(seg, "mask_path", None)
    if mask_path is None:
        raise ValueError("Segmentation label has neither 'mask' nor 'mask_path'")
    img = Image.open(mask_path)
    if img.mode == "P":
        return np.array(img)
    arr = np.array(img)
    return arr[:, :, 0] if arr.ndim == 3 else arr
