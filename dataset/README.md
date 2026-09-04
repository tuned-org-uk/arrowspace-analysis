# Shared calibrated dataset

License: MIT — see `LICENSE` in this directory.

`eigenmaps_controlled.parquet` — 1000 items x 128 dims, 3 orthonormal
clusters of increasing sparsity (dense, medium, sparse).

Provenance: adapted verbatim (values byte-identical) from the
pyarrowspace calibration fixture `tests/data/eigenmaps_controlled.parquet`
(seed 11; `make_datasets.py` there). The only transformation is schema:
pyarrowspace stores `f0..f127` + `cluster`; genefold-vd's native ingest
layouts are `vector: FixedSizeList<Float64>[128]` or wide `col_0..col_127`.
This copy is the wide `col_*` layout with `cluster` dropped.

Required build hyperparameters: the default `eps=0.001, sigma=None` leaves
this unit-norm dataset disconnected (adjacency = 0 nnz, every lambda ~0,
lancefmt rejects the empty artifact). Build with the same graph params that
keep the pyarrowspace tests meaningful — `eps=0.5, sigma=0.5, k=12`
(`p=2.0`, `topk` per test). Do not regenerate; reuse only.
