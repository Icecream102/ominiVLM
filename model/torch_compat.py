import torch


def ensure_torch_transformers_compat():
    # Transformers treats scikit-learn as an optional dependency. Some managed
    # GPU images contain an incompatible sklearn/scipy/numpy combination: the
    # package is discoverable, but importing it raises before AutoTokenizer is
    # available. Disable only that broken optional integration; MiniMind-V does
    # not use sklearn for training, generation, or caption evaluation.
    try:
        import sklearn  # noqa: F401
    except Exception:
        try:
            from transformers.utils import import_utils

            import_utils._sklearn_available = False
        except Exception:
            pass

    try:
        import scipy  # noqa: F401
        from scipy import optimize  # noqa: F401
    except Exception:
        try:
            from transformers.utils import import_utils

            import_utils._scipy_available = False
        except Exception:
            pass

    pytree = getattr(torch.utils, "_pytree", None)
    if pytree is None:
        return
    if hasattr(pytree, "register_pytree_node"):
        return
    legacy_register = getattr(pytree, "_register_pytree_node", None)
    if legacy_register is None:
        return
    pytree.register_pytree_node = legacy_register
