"""Topic modeling pipeline (BERTopic + hybrid phrase detection).

Ported verbatim from the reference implementation at
source_projects/AI-Entrepreneurship-SKOS-Ontology/src/theory_elaboration/topic_modeling/.

Import submodules directly (e.g. `from aecsp.topics.pipeline import training`);
this package init only applies the hdbscan/scikit-learn compatibility patch so
that importing one submodule does not force-load the whole pipeline.
"""

from aecsp.topics.pipeline.compat import patch_hdbscan_sklearn_compat

patch_hdbscan_sklearn_compat()
