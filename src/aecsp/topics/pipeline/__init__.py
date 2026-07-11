"""Initialize topic-modeling compatibility without loading pipeline modules.

Inputs: installed HDBSCAN and scikit-learn packages.
Outputs: a patched dependency interface used by topic pipeline modules.
"""

from aecsp.topics.pipeline.compat import patch_hdbscan_sklearn_compat

patch_hdbscan_sklearn_compat()
