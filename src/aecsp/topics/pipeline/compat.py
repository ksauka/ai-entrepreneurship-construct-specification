"""Compatibility helpers for topic modeling dependencies."""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def patch_hdbscan_sklearn_compat() -> bool:
    """
    Patch HDBSCAN to work with newer scikit-learn versions.

    Newer scikit-learn renamed `force_all_finite` to `ensure_all_finite` in
    validation helpers. Older HDBSCAN builds still call `check_array` with the
    old keyword, which raises a TypeError during BERTopic clustering.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    try:
        from sklearn.utils import validation as sk_validation
    except Exception as exc:
        logger.debug("Skipping sklearn compatibility patch: %s", exc)
        return False

    try:
        signature = inspect.signature(sk_validation.check_array)
    except Exception as exc:
        logger.debug("Unable to inspect sklearn.check_array: %s", exc)
        return False

    if "force_all_finite" in signature.parameters:
        _PATCH_APPLIED = True
        return True

    original_check_array = sk_validation.check_array

    def compat_check_array(*args: Any, force_all_finite: Any = None, **kwargs: Any):
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return original_check_array(*args, **kwargs)

    sk_validation.check_array = compat_check_array

    try:
        import hdbscan.hdbscan_ as hdbscan_module
        hdbscan_module.check_array = compat_check_array
    except Exception as exc:
        logger.debug("Unable to patch hdbscan.check_array directly: %s", exc)

    _PATCH_APPLIED = True
    logger.info("Applied HDBSCAN/scikit-learn compatibility patch for check_array")
    return True
