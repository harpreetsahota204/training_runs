"""HuggingFace ``transformers`` training core for the Train Model operator.

Owns the HF-specific work for detection, classification, and semantic
segmentation: building the FiftyOne -> PyTorch bridge, fitting a
``transformers`` AutoModel, and returning a ``record_run`` spec whose ``fit``
trains and yields the in-memory model. ``fiftyone.apply_model`` auto-converts
that model object, so the predictions written back use the same weights that
were just trained -- no reload required.

The form scaffolding, view resolution, checkpoint placement, and engine calls
come from ``training_helpers``. Training writes to a local temp directory (the
HF ``Trainer`` needs local I/O); the saved model directory is then copied to
the user's chosen output directory, which may be local or cloud.
"""

import importlib.util
import json
import os
import sys

import fiftyone.core.labels as fol
import fiftyone.core.storage as fos
import fiftyone.operators.types as types
import fiftyone.utils.training as fout

from . import training_helpers as th

# transformers is an optional dependency; check without importing the heavy
# package at plugin-load time (the real imports happen in the fit closures).
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

# Ground-truth label types accepted per HF task (read by the form to filter the
# label field + framework choices). Segmentation is restricted to semantic
# ``Segmentation`` masks so the welded eval (evaluate_segmentations) compares
# like-for-like.
TASK_LABEL_TYPES = {
    "detection": (fol.Detections,),
    "classification": (fol.Classification,),
    "segmentation": (fol.Segmentation,),
}

# A sensible default model id per task (free-text; any compatible id works).
_DEFAULT_MODELS = {
    "detection": "facebook/detr-resnet-50",
    "classification": "google/vit-base-patch16-224",
    "segmentation": "nvidia/mit-b0",
}

_MODEL_HINTS = {
    "detection": "Any AutoModelForObjectDetection id, e.g. facebook/detr-resnet-50",
    "classification": "Any AutoModelForImageClassification id, e.g. google/vit-base-patch16-224",
    "segmentation": "Any AutoModelForSemanticSegmentation id, e.g. nvidia/mit-b0",
}


def hf_model_inputs(inputs, ctx, task):
    """Add the HuggingFace model + hyperparameter fields for the chosen ``task``
    (task / label field / splits are collected by the operator). Returns
    ``False`` if a prerequisite is missing (e.g. segmentation mask_targets)."""
    if not HAS_TRANSFORMERS:
        inputs.view(
            "hf_error",
            types.Error(
                label="transformers not installed",
                description="Install it with: pip install transformers timm accelerate",
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

    inputs.str(
        "model_name",
        default=_DEFAULT_MODELS[task],
        required=True,
        label="Model id",
        description=_MODEL_HINTS[task],
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
    inputs.float("learning_rate", default=5e-5, label="Learning rate")

    inputs.bool(
        "push_to_hub",
        default=False,
        label="Push to HuggingFace Hub",
        description="Upload the model after training (needs an HF_TOKEN secret)",
        view=types.CheckboxView(),
    )
    if ctx.params.get("push_to_hub"):
        inputs.str(
            "hub_model_id",
            required=True,
            label="Hub model id",
            description="Repository id on the Hub, e.g. your-username/model-name",
        )
    return True


def train_hf(ctx, train_key):
    """Build the ``record_run`` spec for a HuggingFace run.

    Forces single-GPU training: the HF ``Trainer`` wraps models in
    ``nn.DataParallel`` when it sees multiple GPUs, which breaks models whose
    ``forward()`` reads ``self.device`` (e.g. DETR).
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if sys.platform == "darwin":
        # Apple's MPS backend lacks some detector ops (e.g. deformable
        # attention); fall back to CPU for those instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    task = ctx.params.get("task", "detection")
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
        "learning_rate": ctx.params.get("learning_rate", 5e-5),
    }
    if ctx.user_id:
        cfg["triggered_by"] = ctx.user_id
    return cfg


def _spec(ctx, train_config, splits, fit):
    """Assemble the ``record_run`` spec shared by every HF task."""
    train_arg, val_arg, test_arg = splits
    return dict(
        train_arg=train_arg,
        val_arg=val_arg,
        test_arg=test_arg,
        gt_field=ctx.params["label_field"],
        pred_field=ctx.params.get("pred_field") or "hf_predictions",
        auto_eval=None,
        project_url=ctx.params.get("project_url"),
        train_config=train_config,
        fit=fit,
    )


def _training_args(ctx, output_dir):
    """Build ``TrainingArguments`` shared across the HF tasks. Hub fields are
    read from the form; the token comes from the HF_TOKEN secret."""
    import torch
    from transformers import TrainingArguments

    push = bool(ctx.params.get("push_to_hub"))
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=ctx.params.get("epochs", 5),
        per_device_train_batch_size=ctx.params.get("batch_size", 4),
        per_device_eval_batch_size=ctx.params.get("batch_size", 4),
        learning_rate=ctx.params.get("learning_rate", 5e-5),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        remove_unused_columns=False,
        push_to_hub=push,
        hub_model_id=ctx.params.get("hub_model_id") if push else None,
        hub_token=ctx.secrets.get("HF_TOKEN") if push else None,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )


def _save_and_place(ctx, trainer, processor, mapping):
    """Save model + processor + class mapping, optionally push to the Hub, and
    copy the saved directory to the user's output directory; return its URI."""
    save_dir = fos.make_temp_dir()
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "class_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)
    if ctx.params.get("push_to_hub"):
        trainer.push_to_hub()
    return th.place_checkpoint_dir(ctx, save_dir)


def _union_classes(ctx, path, *args):
    """Sorted, non-null class names across every provided split.

    The class map must cover all splits: a class present only in val/test (but
    not train) must still be known, or the bridge's strict label lookup fails.
    """
    classes = set()
    for arg in args:
        view = fout.resolve_view(ctx.dataset, arg)
        if view is not None:
            classes.update(v for v in view.distinct(path) if v is not None)
    return sorted(classes)


def _index_classes(classes):
    """``(label2id, id2label)`` for a sorted class list."""
    label2id = {c: i for i, c in enumerate(classes)}
    return label2id, {i: c for c, i in label2id.items()}


def _mapping(label2id, id2label):
    """The ``class_mapping.json`` payload saved alongside the model (``json``
    coerces the int ``id2label`` keys to strings on dump)."""
    return {"label2id": label2id, "id2label": id2label}


def _resolve_train_val(ctx, train_arg, val_arg):
    """The concrete train view and an eval view (val, or train if no val)."""
    train_view = fout.resolve_view(ctx.dataset, train_arg)
    val_view = fout.resolve_view(ctx.dataset, val_arg) or train_view
    return train_view, val_view


# --- classification --------------------------------------------------------


def _train_classification(ctx):
    label_field = ctx.params["label_field"]
    train_arg, val_arg, test_arg = th.resolve_split_args(ctx)
    train_config = _config(ctx, "classification")

    def fit():
        import torch
        from PIL import Image
        from fiftyone.utils.torch import GetItem
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            Trainer,
        )

        classes = _union_classes(
            ctx, f"{label_field}.label", train_arg, val_arg, test_arg
        )
        if len(classes) < 2:
            raise ValueError(
                f"Need at least 2 classes to fine-tune, found {len(classes)} "
                f"in '{label_field}'"
            )
        label2id, id2label = _index_classes(classes)

        processor = AutoImageProcessor.from_pretrained(ctx.params["model_name"])
        model = AutoModelForImageClassification.from_pretrained(
            ctx.params["model_name"],
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        class _Get(GetItem):
            def __init__(self):
                super().__init__(field_mapping={"classification": label_field})

            @property
            def required_keys(self):
                return ["filepath", "classification"]

            def __call__(self, d):
                image = Image.open(d["filepath"]).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                out = {k: v.squeeze(0) for k, v in inputs.items()}
                out["labels"] = torch.tensor(
                    label2id[d["classification"].label], dtype=torch.long
                )
                return out

        train_view, val_view = _resolve_train_val(ctx, train_arg, val_arg)
        trainer = Trainer(
            model=model,
            args=_training_args(ctx, fos.make_temp_dir()),
            train_dataset=train_view.exists(f"{label_field}.label").to_torch(_Get()),
            eval_dataset=val_view.exists(f"{label_field}.label").to_torch(_Get()),
        )
        trainer.train()

        mapping = _mapping(label2id, id2label)
        return trainer.model, _save_and_place(ctx, trainer, processor, mapping), None

    return _spec(ctx, train_config, (train_arg, val_arg, test_arg), fit)


# --- detection -------------------------------------------------------------


def _annotation(det, label2id, w, h):
    """A COCO-detection annotation dict for one FiftyOne detection.

    Every HuggingFace object-detection image processor (DETR / YOLOS / RT-DETR
    and the rest of the ``AutoModelForObjectDetection`` family) expects COCO
    ``[x, y, w, h]`` in absolute pixels and converts to the model's internal
    normalized ``cxcywh`` itself (``do_convert_annotations=True``). So the only
    transform we do is FiftyOne's normalized box -> absolute pixels; there is no
    per-model bbox format to choose.
    """
    rx, ry, rw, rh = det.bounding_box
    bbox = [rx * w, ry * h, rw * w, rh * h]
    return {
        "bbox": bbox,
        "category_id": label2id[det.label],
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
    }


def _train_detection(ctx):
    label_field = ctx.params["label_field"]
    train_arg, val_arg, test_arg = th.resolve_split_args(ctx)
    train_config = _config(ctx, "detection")

    def fit():
        import torch
        from PIL import Image
        from fiftyone.utils.torch import GetItem
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
            Trainer,
        )

        classes = _union_classes(
            ctx, f"{label_field}.detections.label", train_arg, val_arg, test_arg
        )
        if not classes:
            raise ValueError(f"No detection labels found in '{label_field}'")
        label2id, id2label = _index_classes(classes)

        processor = AutoImageProcessor.from_pretrained(ctx.params["model_name"])
        model = AutoModelForObjectDetection.from_pretrained(
            ctx.params["model_name"],
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        train_view, val_view = _resolve_train_val(ctx, train_arg, val_arg)
        train_view = train_view.exists(f"{label_field}.detections")
        val_view = val_view.exists(f"{label_field}.detections")
        train_view.compute_metadata(overwrite=False)
        val_view.compute_metadata(overwrite=False)

        class _Get(GetItem):
            def __init__(self):
                super().__init__(field_mapping={"detections": label_field})

            @property
            def required_keys(self):
                return ["filepath", "detections", "metadata"]

            def __call__(self, d):
                image = Image.open(d["filepath"]).convert("RGB")
                meta = d.get("metadata")
                w, h = (meta.width, meta.height) if meta is not None else image.size
                annotations = [
                    _annotation(det, label2id, w, h)
                    for det in d["detections"].detections
                ]
                inputs = processor(
                    images=image,
                    annotations=[{"image_id": 0, "annotations": annotations}],
                    return_tensors="pt",
                )
                out = {"pixel_values": inputs["pixel_values"].squeeze(0)}
                if "pixel_mask" in inputs:
                    out["pixel_mask"] = inputs["pixel_mask"].squeeze(0)
                out["labels"] = inputs["labels"][0]
                return out

        def _collate(batch):
            pixel_values = [b["pixel_values"] for b in batch]
            max_h = max(pv.shape[-2] for pv in pixel_values)
            max_w = max(pv.shape[-1] for pv in pixel_values)
            images, masks = [], []
            for pv in pixel_values:
                _, ph, pw = pv.shape
                images.append(
                    torch.nn.functional.pad(pv, (0, max_w - pw, 0, max_h - ph), value=0)
                )
                mask = torch.zeros(max_h, max_w, dtype=torch.long)
                mask[:ph, :pw] = 1
                masks.append(mask)
            return {
                "pixel_values": torch.stack(images),
                "pixel_mask": torch.stack(masks),
                "labels": [b["labels"] for b in batch],
            }

        trainer = Trainer(
            model=model,
            args=_training_args(ctx, fos.make_temp_dir()),
            data_collator=_collate,
            train_dataset=train_view.to_torch(_Get()),
            eval_dataset=val_view.to_torch(_Get()),
            processing_class=processor,
        )
        trainer.train()

        mapping = _mapping(label2id, id2label)
        apply_kwargs = {"confidence_thresh": ctx.params.get("confidence", 0.25)}
        return (
            trainer.model,
            _save_and_place(ctx, trainer, processor, mapping),
            apply_kwargs,
        )

    return _spec(ctx, train_config, (train_arg, val_arg, test_arg), fit)


# --- semantic segmentation -------------------------------------------------


def _mask_targets(dataset, label_field):
    if not label_field:
        return None
    targets = dataset.mask_targets.get(label_field) if dataset.mask_targets else None
    return targets or dataset.default_mask_targets or None


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


def _train_segmentation(ctx):
    label_field = ctx.params["label_field"]
    train_arg, val_arg, test_arg = th.resolve_split_args(ctx)
    train_config = _config(ctx, "segmentation")

    def fit():
        import numpy as np
        import torch
        from PIL import Image
        from fiftyone.utils.torch import GetItem
        from transformers import (
            AutoImageProcessor,
            AutoModelForSemanticSegmentation,
            Trainer,
        )

        targets = _mask_targets(ctx.dataset, label_field)
        if not targets:
            raise ValueError(
                f"No mask_targets found for '{label_field}'. Set "
                "dataset.mask_targets so classes can be mapped."
            )
        id2label = {int(k): v for k, v in targets.items()}
        label2id = {v: int(k) for k, v in targets.items()}
        if len(id2label) < 2:
            raise ValueError("Need at least 2 segmentation classes to fine-tune")

        processor = AutoImageProcessor.from_pretrained(ctx.params["model_name"])
        model = AutoModelForSemanticSegmentation.from_pretrained(
            ctx.params["model_name"],
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
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
                inputs = processor(
                    images=image,
                    segmentation_maps=Image.fromarray(mask.astype(np.uint8)),
                    return_tensors="pt",
                )
                return {k: v.squeeze(0) for k, v in inputs.items()}

        def _collate(batch):
            return {
                "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
            }

        train_view, val_view = _resolve_train_val(ctx, train_arg, val_arg)
        trainer = Trainer(
            model=model,
            args=_training_args(ctx, fos.make_temp_dir()),
            data_collator=_collate,
            train_dataset=train_view.exists(label_field).to_torch(_Get()),
            eval_dataset=val_view.exists(label_field).to_torch(_Get()),
            processing_class=processor,
        )
        trainer.train()

        mapping = _mapping(label2id, id2label)
        return trainer.model, _save_and_place(ctx, trainer, processor, mapping), None

    return _spec(ctx, train_config, (train_arg, val_arg, test_arg), fit)
