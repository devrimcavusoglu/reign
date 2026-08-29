"""DAPFAM patent prior-art benchmark adapter (real-world long-document case study).

``build_dataset`` converts the upstream DAPFAM release into the BEIR/MTEB layout the
retrieval harness consumes; ``split_qrels`` derives query-disjoint train/val/test
qrels from the eval-only upstream split.
"""
