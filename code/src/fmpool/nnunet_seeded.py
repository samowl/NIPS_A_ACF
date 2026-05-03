"""Seeded nnU-Net trainer subclasses.

This module is the project-level divergence from upstream nnU-Net. nnU-Net v2
does not expose a seed argument in the CLI, so we use trainer subclasses that
seed Python, NumPy, and PyTorch before the upstream trainer starts.

Important: official nnU-Net discovers custom trainers by recursively scanning
the installed ``nnunetv2.training.nnUNetTrainer`` package on disk. Dynamic
classes attached only to this module are useful for introspection, but they are
not enough for ``nnUNetv2_train -tr ...``. Call
``ensure_nnunet_discovery()`` once in the active environment to write and
verify the small compatibility shim that nnU-Net can discover.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Type

from fmpool import determinism

logger = logging.getLogger(__name__)

# Seeds 13/37 reproduce the released nnU-Net summary JSONs. Seeds 42--45 are
# retained for compatibility with the rest of the FMPool seed grid.
SEEDS: tuple[int, ...] = (13, 37, 42, 43, 44, 45)
BASE_TRAINERS: tuple[str, ...] = (
    "nnUNetTrainer_100epochs",
    "nnUNetTrainer_1000epochs",
)
DISCOVERY_MODULE = "nnUNetTrainer_fmpool_seeded.py"


def make_seeded_trainer_cls(base_cls: Type, seed: int) -> Type:
    """Return a subclass of ``base_cls`` that seeds RNGs in ``on_train_start``."""

    class _SeededTrainer(base_cls):  # type: ignore[misc,valid-type]
        fmpool_seed: int = int(seed)

        def on_train_start(self) -> None:  # type: ignore[override]
            determinism.set_seed(self.fmpool_seed)
            logger.info(
                "nnUNetTrainer seeded with %d (class=%s)",
                self.fmpool_seed,
                type(self).__name__,
            )
            super().on_train_start()

    _SeededTrainer.__name__ = f"{base_cls.__name__}_Seed{seed}"
    _SeededTrainer.__qualname__ = _SeededTrainer.__name__
    return _SeededTrainer


def released_alias_name(base_name: str, seed: int) -> str:
    """Class name convention used by the released nnU-Net result summaries."""

    if not base_name.startswith("nnUNetTrainer_"):
        raise ValueError(f"unexpected nnU-Net trainer base name: {base_name}")
    return f"nnUNetTrainerSeed{seed}_{base_name.removeprefix('nnUNetTrainer_')}"


def _load_upstream_base(base_name: str) -> Type | None:
    try:
        from nnunetv2.training.nnUNetTrainer.variants.training_length import (  # type: ignore
            nnUNetTrainer_Xepochs,
        )
    except ImportError:
        logger.info("nnunetv2 not importable; skipping seeded trainer registration.")
        return None

    base = getattr(nnUNetTrainer_Xepochs, base_name, None)
    if base is not None:
        return base
    if base_name == "nnUNetTrainer_1000epochs":
        try:
            import torch
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (  # type: ignore
                nnUNetTrainer,
            )
        except ImportError:
            return None

        class nnUNetTrainer_1000epochs(nnUNetTrainer):  # type: ignore[no-redef]
            def __init__(
                self,
                plans: dict,
                configuration: str,
                fold: int,
                dataset_json: dict,
                device: torch.device = torch.device("cuda"),
            ) -> None:
                super().__init__(plans, configuration, fold, dataset_json, device)
                self.num_epochs = 1000

        return nnUNetTrainer_1000epochs

    return None


def register_all_seeded_trainers(
    base_names: tuple[str, ...] = BASE_TRAINERS,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Type]:
    """Attach seeded variants to this module for local introspection."""

    registered: dict[str, Type] = {}
    module = sys.modules[__name__]
    for base_name in base_names:
        base_cls = _load_upstream_base(base_name)
        if base_cls is None:
            logger.warning("Upstream trainer %s not found in nnunetv2; skipping.", base_name)
            continue
        setattr(module, base_name, base_cls)
        for seed in seeds:
            variant = make_seeded_trainer_cls(base_cls, seed)
            setattr(module, variant.__name__, variant)
            registered[variant.__name__] = variant
            alias_name = released_alias_name(base_name, seed)
            alias = type(
                alias_name,
                (variant,),
                {
                    "__module__": __name__,
                    "__doc__": f"Released-result alias for {variant.__name__}.",
                },
            )
            setattr(module, alias_name, alias)
            registered[alias_name] = alias
    logger.info("registered %d seeded nnU-Net variant(s)", len(registered))
    return registered


def _shim_source() -> str:
    seed_classes = []
    for base in BASE_TRAINERS:
        for seed in SEEDS:
            canonical = f"{base}_Seed{seed}"
            alias = released_alias_name(base, seed)
            seed_classes.append(
                f"class {canonical}(_SeedMixin, {base}):\n"
                f"    fmpool_seed = {seed}\n\n"
                f"class {alias}({canonical}):\n"
                f"    pass\n"
            )
    return (
        '"""FMPool seeded nnU-Net trainer shim.\n\n'
        "Generated by fmpool.nnunet_seeded.ensure_nnunet_discovery(). "
        "Classes inherit official nnU-Net trainers and only seed RNGs before "
        "training starts.\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "import random\n"
        "import numpy as np\n"
        "import torch\n"
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n"
        "from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs\n\n"
        "class nnUNetTrainer_1000epochs(nnUNetTrainer):\n"
        "    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):\n"
        "        super().__init__(plans, configuration, fold, dataset_json, device)\n"
        "        self.num_epochs = 1000\n\n"
        "def _seed_all(seed: int) -> None:\n"
        "    random.seed(seed)\n"
        "    np.random.seed(seed)\n"
        "    torch.manual_seed(seed)\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.manual_seed_all(seed)\n"
        "    torch.backends.cudnn.deterministic = True\n"
        "    torch.backends.cudnn.benchmark = False\n\n"
        "class _SeedMixin:\n"
        "    fmpool_seed: int\n"
        "    def on_train_start(self) -> None:\n"
        "        _seed_all(int(self.fmpool_seed))\n"
        "        super().on_train_start()\n\n"
        + "\n".join(seed_classes)
        + "\n"
    )


def _nnunet_trainer_root() -> Path:
    import nnunetv2  # type: ignore

    return Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"


def _discovered_class(name: str) -> Type | None:
    from nnunetv2.utilities.find_class_by_name import (  # type: ignore
        recursive_find_python_class,
    )

    return recursive_find_python_class(
        str(_nnunet_trainer_root()), name, "nnunetv2.training.nnUNetTrainer"
    )


def ensure_nnunet_discovery(write: bool = True) -> dict[str, str]:
    """Install and verify the official nnU-Net trainer-discovery shim.

    The function writes one module into the active nnU-Net installation under
    ``training/nnUNetTrainer/variants/training_length`` and verifies that
    ``recursive_find_python_class`` can resolve all documented seeded trainer
    names plus the 1000-epoch base trainer. It does not start training.
    """

    target = _nnunet_trainer_root() / "variants" / "training_length" / DISCOVERY_MODULE
    if write:
        target.write_text(_shim_source(), encoding="utf-8")
        importlib.invalidate_caches()

    expected = []
    for base in BASE_TRAINERS:
        for seed in SEEDS:
            expected.append(f"{base}_Seed{seed}")
            expected.append(released_alias_name(base, seed))
    expected.append("nnUNetTrainer_1000epochs")
    missing = [name for name in expected if _discovered_class(name) is None]
    if missing:
        raise RuntimeError(
            "nnU-Net seeded trainer discovery failed for "
            + ", ".join(missing)
            + f". Tried shim path: {target}"
        )
    return {"path": str(target), "registered": str(len(expected))}


_REGISTERED: dict[str, Type] = register_all_seeded_trainers()


__all__ = [
    "SEEDS",
    "BASE_TRAINERS",
    "make_seeded_trainer_cls",
    "register_all_seeded_trainers",
    "ensure_nnunet_discovery",
]
