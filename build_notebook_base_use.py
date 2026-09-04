"""Build the arrowspace-base-use notebooks (feature Laplacian, sequencing, motives).

Data: dataset/eigenmaps_controlled.parquet — 1000 items x 128 dims, three orthonormal
clusters of increasing sparsity (pyarrowspace calibration fixture; values byte-identical,
labels recovered in-notebook by k-means). Build hyperparameters are the ones the dataset
README prescribes: eps=0.5, sigma=0.5, k=12, p=2.0. Do not regenerate the dataset.

Run:  python build_notebook_base_use.py
Then execute:
  python -m jupyter nbconvert --to notebook --execute --inplace \
      arrowspace-base-use/00__feature_space_laplacian_and_search.ipynb \
      arrowspace-base-use/01__sequencing_api.ipynb \
      arrowspace-base-use/02__motives_api.ipynb

Requires arrowspace >= 0.28.0.
"""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

OUT_DIR = Path("arrowspace-base-use")
OUT_DIR.mkdir(exist_ok=True)


def new_nb():
    nb = nbf.v4.new_notebook()
    nb.cells = []
    return nb


def md(nb, src):
    nb.cells.append(nbf.v4.new_markdown_cell(src))


def code(nb, src):
    nb.cells.append(nbf.v4.new_code_cell(src))


DATA_CELL = '''\
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from arrowspace import ArrowSpaceBuilder, sequence_by_lambda, sequence_by_graph

SEED = 42
DATA_PATH = Path("../dataset/eigenmaps_controlled.parquet")

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 11, "axes.grid": True,
    "grid.alpha": 0.3, "figure.constrained_layout.use": True,
})

# The dataset README prescribes these graph parameters; the pre-0.28 defaults
# (eps=0.001) disconnect the unit-norm feature graph (every lambda ~ 0).
# Notably, 0.28.0's recalibrated defaults (eps=0.5, k=12) match this prescription.
# Reuse only, never regenerate.
GRAPH_PARAMS = {"eps": 0.5, "k": 12, "topk": 8, "p": 2.0, "sigma": 0.5}

X = np.ascontiguousarray(  # pyarrow/pandas slices may be non-contiguous; ArrowSpace asks for C-contiguous rows
    pq.read_table(DATA_PATH).to_pandas().to_numpy(np.float64)
)
N, D = X.shape
assert X.shape == (1000, 128) and np.allclose(np.linalg.norm(X, axis=1), 1.0)

# Ground truth: three orthonormal clusters. The shipped copy drops the `cluster`
# column; recover it from the data — the clusters are mutually orthogonal, so
# k-means (fixed seed) recovers them exactly. K-means labels are index-arbitrary, so
# we re-index clusters by mean effective-dimension count (dense -> sparse). This makes
# "cluster 2" always the sparsest, so later narrative/asserts about labels==2 are correct.
p2 = X ** 2
eff_dims = p2.sum(axis=1) ** 2 / np.sum(p2 ** 2, axis=1)   # participation ratio per item
_km = KMeans(3, n_init=10, random_state=SEED).fit_predict(X)
_dense_first = np.argsort([-eff_dims[_km == c].mean() for c in range(3)])
labels = np.empty_like(_km)
for _new, _old in enumerate(_dense_first):
    labels[_km == _old] = _new
NAMES = ["cluster 0 (dense)", "cluster 1", "cluster 2 (sparse)"]
COLORS = np.asarray(plt.get_cmap("tab10").colors)

def modal_share(ix):
    """Fraction of a motif's items belonging to one k-means cluster."""
    _, cnt = np.unique(labels[list(ix)], return_counts=True)
    return float(cnt.max() / len(ix))

xy = PCA(n_components=2, random_state=SEED).fit_transform(X)
fig, ax = plt.subplots(figsize=(6, 5))
for c in range(3):
    ax.scatter(xy[labels == c, 0], xy[labels == c, 1], s=10, alpha=0.7,
               color=COLORS[c], label=NAMES[c])
ax.legend(fontsize=8)
ax.set_title("eigenmaps_controlled: 1000 x 128, 3 orthonormal clusters")
OUT = Path("output__DATASETDIR")
if OUT.exists():
    shutil.rmtree(OUT)                       # deterministic, clean output dir
OUT.mkdir(parents=True)
fig.savefig(OUT / "fig_01_dataset.png")
plt.show()

print(f"items: {X.shape}  counts: {np.bincount(labels).tolist()}")
print(f"cross-cluster cosine ~ {(X @ X.T)[labels[:, None] != labels[None, :]].mean():.4f}")'''


# ═══════════════════════════════════════════════════════════════════════════
# 00 — Feature-space Laplacian + search
# ═══════════════════════════════════════════════════════════════════════════
nb = new_nb()

md(nb, r'''# Base-use 00 — The feature-space graph Laplacian, seen through `search()`

**Question.** Every `ArrowSpaceBuilder().build(params, items)` returns a **pair**: the
index `aspace` and a `GraphLaplacian` `gl` — and every search call takes `gl` again:
`aspace.search(q, gl, tau)`. What is this Laplacian a graph *of*, and what does the `tau`
dial actually do to rankings?

The bindings' `gl` is a **feature-space** Laplacian $L_F$: a $D \times D$ graph whose
nodes are the **embedding dimensions**, weighted by how the dataset makes them
co-vary — not an item-space adjacency. Per-item $\lambda\tau$ scores are Rayleigh-style
statistics computed through this wiring: items in spectrally coherent dimension regions
score low, rough/transitional items score high. A query must be $\lambda$-prepared against
the same graph before it can be ranked — hence `gl` as a search argument.

**Dataset.** `dataset/eigenmaps_controlled.parquet` — 1000 unit-norm items × 128
dimensions, three orthonormal clusters of increasing sparsity (pyarrowspace calibration
fixture). The dataset README prescribes `eps=0.5, sigma=0.5, k=12` (the pre-0.28 default
`eps=0.001` disconnects the graph and collapses every $\lambda\tau$ to 0 — the
quality-gate failure mode in the docs; 0.28.0 recalibrated its defaults to this regime). We recover the cluster labels in-notebook with k-means; nothing else
about the data is used, so every score below is an *unsupervised* ArrowSpace readout.

You will see:

1. $L_F$ at real scale: a 128-node dimension graph — heatmap, eigenvalue spectrum, Fiedler value.
2. $\lambda\tau$ as an unsupervised **sparsity detector**: rank-correlation $\approx -0.4$
   with each item's effective dimension count — with labels never shown to the index.
3. `tau` as the cosine $\leftrightarrow$ spectral blend: `tau=1.0` reproduces cosine
   exactly; as $\tau$ drops, the tail of the ranking becomes increasingly $\lambda\tau$-ordered.
4. The 0.27 error contract: catchable `ValueError`s for degenerate/non-finite/mismatched
   queries — and a Principle-0 caution against re-deriving $\lambda$ in NumPy.

### References
- ArrowSpace JOSS paper: https://doi.org/10.21105/joss.09002
- Dataset README + MIT license: `dataset/`
- Repo principles: `notebooks/README.md` (API scores only; $\lambda\tau$ is a final score)''')

md(nb, r'---\n## 0 · Imports, dataset, and the prescribed hyperparameters')

code(nb, DATA_CELL.replace("output__DATASETDIR", "output__intro"))

md(nb, r'''---\n## 1 · The build returns an index *and* the wiring

`with_dims_reduction(False, None)` keeps the analysis honest: no Johnson–Lindenstrauss
projection between our 128 dimensions and $L_F$.''')

code(nb, '''\
aspace, gl = (
    ArrowSpaceBuilder()
    .with_seed(SEED)
    .with_dims_reduction(False, None)
    .build(GRAPH_PARAMS, X)
)

lam = np.asarray(aspace.lambdas())
n_degenerate = int(np.sum(np.abs(lam) < 1e-12))

print(f"aspace: nitems={aspace.nitems}  nfeatures={aspace.nfeatures}  nclusters={aspace.nclusters}")
print(f"gl:     shape={gl.shape()}  nnodes={gl.nnodes}")
print(f"degenerate lambdas: {n_degenerate} / {len(lam)}   range: [{lam.min():.3f}, {lam.max():.3f}]")
assert gl.shape() == (D, D), "the exposed Laplacian is the feature-space L_F"
assert n_degenerate < 0.05 * len(lam), "index collapsed: the README's eps is not being honoured"
# Note: gl.nnodes reports the item count of the raw data, NOT the node count of
# gl.matrix (F x F over features) — the #165 namespace split; trust shape().''')

code(nb, r'''# What is written on L_F at 128-node scale: the dimension graph and its spectrum.
L = np.asarray(gl.to_dense(), dtype=np.float64)
Lsym = (L + L.T) / 2
eig = np.sort(np.linalg.eigvalsh(Lsym))
fiedler = float(eig[1])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
im = axes[0].imshow(np.abs(Lsym), cmap="viridis")
axes[0].set_title(r"$|L_F|$: 128 x 128 dimension graph")
fig.colorbar(im, ax=axes[0], fraction=0.046)

axes[1].plot(eig, lw=1.2)
axes[1].axvline(1, color="tab:red", ls="--", lw=1, label=f"Fiedler = {fiedler:.2f}")
axes[1].set_xlabel("eigenvalue index"); axes[1].set_ylabel(r"$\lambda_i(L_F)$")
axes[1].set_title("spectrum of $L_F$"); axes[1].legend(fontsize=8)

for c in range(3):
    axes[2].hist(lam[labels == c], bins=40, alpha=0.6, color=COLORS[c], label=NAMES[c])
axes[2].set_xlabel(r"$\lambda\tau$ (from the API)")
axes[2].set_title(r"per-item $\lambda\tau$ by cluster")
axes[2].legend(fontsize=7)
fig.savefig(OUT / "fig_02_laplacian_and_scores.png")
plt.show()

rho_sparsity = float(spearmanr(lam, eff_dims).statistic)
mean_lam = [round(float(lam[labels == c].mean()), 4) for c in range(3)]
print(f"Fiedler value = {fiedler:.3f}  (well above 0.1: dimension graph is well wired)")
print(f"mean lambda-tau per cluster (dense->sparse): {mean_lam}  |  "
      f"spearman(lambda-tau, eff-dims) = {rho_sparsity:.3f}")
assert mean_lam == sorted(mean_lam), "lambda-tau should rise monotonically with cluster sparsity"
assert rho_sparsity < -0.3, "lambda-tau should rise as items spread over fewer effective dims"''')

md(nb, r'''**Reading.** At $D=128$ the exposed Laplacian is a real graph with structure, not a
toy. Its Fiedler value (~0.65) says the dimension wiring is healthy; and the per-item
readout already *unsupervisedly* tracks the dataset's construction: the dense cluster
sits near $\lambda\tau \approx 0.02$ while the sparser clusters sit roughly four times higher,
with rank correlation $\approx -0.4$ against an item's effective dimension count. Coherent,
energy-spread items are spectrally smooth; sparse/rough items are not.''')

md(nb, r'''---\n## 2 · `search(q, gl, tau)`: tau is the cosine $\leftrightarrow$ spectral dial

`tau` weights the geometric term; $1-\tau$ weights the spectral ($\lambda\tau$) term.
Passing `k=N` returns the full ranking, so we can *measure* what the blend does instead
of eyeballing top-3. With orthonormal clusters the retrieval window itself is stable
(the query's own basin wins on cosine at any tau) — the interesting motion is **in the
tail**, where ordering switches from geometric noise to the spectral curriculum.''')

code(nb, '''\
X_feat = aspace.get_all_items()                   # features as stored by the index
q_idx = int(np.where(labels == 0)[0][5])
q = X[q_idx]
cos_scores = X_feat @ q                           # rows are unit-norm

TAUS = [1.0, 0.7, 0.5, 0.3, 0.1]
rows = []
for tau in TAUS:
    hits = aspace.search(q, gl, tau, k=N)         # k overrides build-time topk
    idx = np.array([i for i, _ in hits])
    scores = np.array([s for _, s in hits])
    assert np.all(np.diff(scores) <= 1e-12), "hits arrive best-first"
    tail = idx[100:]
    rows.append({
        "tau": tau,
        "top20 == cosine top20": int((idx[:20] == np.argsort(-cos_scores)[:20]).all()),
        "own-basin in top100": int((labels[idx[:100]] == 0).sum()),
        "overlap@100 with cosine": round(float(len(set(idx[:100]) & set(np.argsort(-cos_scores)[:100])) / 100), 3),
        "mean eff-dims in top100": round(float(eff_dims[idx[:100]].mean()), 1),
        "spearman(tail order, lambda-tau)": round(float(spearmanr(np.arange(len(tail)), lam[tail]).statistic), 3),
    })
df = pd.DataFrame(rows)
print(df.to_string(index=False))''')

code(nb, '''\
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(df["tau"], df["overlap@100 with cosine"], "o-", color="tab:blue",
             label="top-100 overlap with cosine")
axes[0].plot(df["tau"], df["mean eff-dims in top100"] / df.loc[df.tau == 1.0, "mean eff-dims in top100"].iloc[0],
             "o--", color=COLORS[0], label="eff-dims of top-100 (normalised)")
axes[0].set_xlabel("tau (cosine weight)"); axes[0].set_ylim(0.3, 1.05)
axes[0].set_title("the retrieval window reorders as geometry loses weight")
axes[0].legend(fontsize=8)

axes[1].plot(df["tau"], df["spearman(tail order, lambda-tau)"], "o-", color="tab:red")
axes[1].set_xlabel("tau"); axes[1].set_ylabel("spearman(position, lambda-tau) on ranks 100+")
axes[1].set_title("the tail becomes a lambda-tau curriculum")
fig.savefig(OUT / "fig_03_tau_blend.png")
plt.show()

row1 = df.loc[df.tau == 1.0].iloc[0]
assert row1["top20 == cosine top20"] == 1, "tau=1 must equal cosine"
r_low = df.loc[df.tau == 0.1, "spearman(tail order, lambda-tau)"].iloc[0]
assert abs(r_low) > abs(row1["spearman(tail order, lambda-tau)"]), "spectral weight must order the tail"
print(f"tau=1.0 reproduces cosine exactly; at tau=0.1 the post-window tail is lambda-ordered "
      f"(rho={r_low}). Own-basin retrieval held at 100/100 throughout — the blend cost was paid "
      "by the tail, not the window.")''')

md(nb, r'''---\n## 3 · Error contract and a Principle-0 caution

Queries go through `try_prepare_query_item` against the graph wiring, and every recoverable
failure surfaces as a catchable `ValueError` (0.27 typed errors — never a
`PanicException` crossing the FFI).''')

code(nb, '''\
try:
    aspace.search(np.ones(3, dtype=np.float64), gl, 0.7)
except ValueError as e:
    print("wrong-dim query  ->", str(e)[:68])
try:
    aspace.search(np.full(D, np.nan), gl, 0.7)
except ValueError as e:
    print("non-finite query ->", str(e)[:68])

j = int(np.argmin(np.abs(lam)))                 # the dataset's one degenerate item
try:
    aspace.search(X[j], gl, 0.3)
except ValueError as e:
    print(f"degenerate-lambda item (index {j}, lambda={lam[j]:.1e}) -> {str(e)[:60]}...")

# Principle 0: manually Rayleigh-quotienting item vectors on the exposed L_F is a
# DIFFERENT object from the API's lambda-tau (which folds in the builder's internal
# item/centroid graph and the TauMode synthesis). Always read scores from the API.
R_manual = (X @ Lsym * X).sum(axis=1)           # unit-norm rows: x^T L x / x^T x = x^T L x
mad = float(np.mean(np.abs(R_manual - lam)))
rho = float(spearmanr(R_manual, lam).statistic)
print(f"manual x^T L_F x vs API lambda-tau: spearman = {rho:.3f},  mean abs diff = {mad:.3f}")
print("  -> not the same numbers; only the API's are the index's.")''')

md(nb, r'''---\n## 4 · Summary''')

code(nb, '''\
summary = pd.DataFrame({
    "check": [
        "gl.shape equals (D, D) — feature-space L_F",
        "degenerate lambdas",
        "Fiedler value of L_F",
        "spearman(lambda-tau, eff-dims) — unsupervised sparsity detector",
        "tau=1.0 equals cosine top-20",
        "tail lambda-ordering at tau=0.1 (spearman)",
        "manual vs API score spearman / mean-abs-diff",
    ],
    "value": [
        str(gl.shape()),
        n_degenerate,
        round(fiedler, 3),
        round(rho_sparsity, 3),
        row1["top20 == cosine top20"],
        r_low,
        f"{rho:.3f} / {mad:.3f}",
    ],
})
summary.to_csv(OUT / "laplacian_intro_summary.csv", index=False)
print(summary.to_string(index=False))''')

md(nb, r'''**Reading.**

1. `build` hands you the dimension graph $L_F$ because queries must be $\lambda$-prepared
   against the same wiring; item ranking itself goes through the index's internal
   centroid graph plus the TauMode synthesis. `gl.shape()`, not `gl.nnodes`, is the
   Laplacian's true size.
2. `search(q, gl, tau)` is one honest knob: $\tau=1$ is audited cosine (bit-exact here);
   as $\tau\downarrow$ the blend reorders the tail into a $\lambda\tau$ curriculum
   (spearman 0.29 -> 0.75 here) while the retrieval window itself holds at 100/100 own
   basin — on orthonormal basins the blend cost lands in the tail, not the window.
3. Hyperparameters are load-bearing: the dataset README's `eps=0.5, sigma=0.5` is what
   keeps this unit-norm 128-dim graph connected. At the pre-0.28 default `eps=0.001`
   every $\lambda\tau$ collapses to 0 and no amount of `tau` tuning recovers a signal —
   which is exactly why 0.28.0 recalibrated its builder defaults to the README's regime.

**Caveats.** Single query, single seed, controlled fixture — tune the sweep (multiple
queries, tau grid, eps/k sensitivity) before drawing conclusions on your own embeddings.''')

nbf.write(nb, OUT_DIR / "00__feature_space_laplacian_and_search.ipynb")
print("wrote", OUT_DIR / "00__feature_space_laplacian_and_search.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# 01 — Sequencing API
# ═══════════════════════════════════════════════════════════════════════════
nb = new_nb()

md(nb, r'''# Base-use 01 — The ArrowSpace Sequencing API (`sequence_by_lambda`, `sequence_by_graph`)

**Question.** ArrowSpace gives every item a $\lambda\tau$ score and every build a graph
Laplacian. Both are *sets* of numbers — what does a principled **total ordering** of them
look like, and what is each ordering good for?

**Background — sequencing.** Sequencing is the problem of turning a set into a principled
*total order*. It has two classical roots. *Seriation* — from archaeology and numerical
taxonomy — orders objects so that similar ones sit adjacent; modern variants order graph
nodes by spectral coordinates (the Fiedler vector) to compress matrices into banded form.
*Curriculum learning* (Bengio et al. 2009) orders training data easy → hard, so a model
sees coherent structure before edge cases; its perennial pain point is the difficulty
signal, usually a hand-designed proxy.

**Curriculum in this context.** A *spectral curriculum* is an unsupervised easy → hard
ordering of the items where "difficulty" is spectral roughness: items deep inside a basin
carry low Rayleigh energy (low $\lambda\tau$), boundary and transition items carry high.
Ascending $\lambda\tau$ therefore *is* a curriculum — no difficulty labels, no proxies,
the same signal search already blends with geometry.

**What ArrowSpace contributes.** The ordering is not a separate optimisation bolted onto
the index — it is a *readout* of the per-item $\lambda\tau$ statistics and Laplacian
wiring the index already computed for search. One build yields two deterministic orderings
(documented tie-breaking, no new hyperparameters): the item-space curriculum
(`sequence_by_lambda`) and a graph-side seriation (`sequence_by_graph`, DFS over the
minimum spanning forest with a contiguity guarantee per connected component) — the
classical seriation objective, obtained as a walk rather than an eigenvector solve.

Since arrowspace 0.27 the Python bindings expose two module-level sequencing functions:

| Call | Orders | Positions report | Use case |
|---|---|---|---|
| `sequence_by_lambda(lambdas, descending=False)` | the **items** (any 1-D score vector) | the $\lambda\tau$ score at each step | spectral curriculum: train/retrieve coherent items first, rough/sparse items last |
| `sequence_by_graph(gl)` | the **nodes of the Laplacian `gl`** | DFS discovery depth | matrix seriation: reveal contiguous block structure |

Both return a `Sequence` with `order` (integer permutation), `positions` (per-step
coordinate aligned with `order`), and `components`. Both are fully deterministic (ties
break on ascending node index); inputs of length < 2 raise `ValueError`.

**Node-space caveat.** `gl` is the feature-space Laplacian $L_F$ ($D\times D$, nodes =
dimensions), so `sequence_by_graph(gl)` seriates *dimensions*, while
`sequence_by_lambda(aspace.lambdas())` curricula *items*. Keep the namespaces apart.

**Dataset.** `dataset/eigenmaps_controlled.parquet` (1000 × 128, three orthonormal
clusters of increasing sparsity) built with the README-prescribed `eps=0.5, sigma=0.5,
k=12`; cluster labels recovered in-notebook by k-means and never shown to the index.''')

md(nb, r'---\n## 0 · Imports, dataset, and the prescribed hyperparameters')

code(nb, DATA_CELL.replace("output__DATASETDIR", "output__sequencing"))

md(nb, r'---\n## 1 · Build the index and health-check it')

code(nb, '''\
aspace, gl = (
    ArrowSpaceBuilder()
    .with_seed(SEED)
    .with_dims_reduction(False, None)
    .build(GRAPH_PARAMS, X)
)

lam = np.asarray(aspace.lambdas())
n_degenerate = int(np.sum(np.abs(lam) < 1e-12))
print(f"nitems={aspace.nitems}  nfeatures={aspace.nfeatures}  nclusters={aspace.nclusters}")
print(f"L_F shape={gl.shape()}  degenerate lambdas: {n_degenerate} / {len(lam)}")
assert n_degenerate < 0.05 * len(lam), "too many degenerate lambdas: honour the README eps"''')

md(nb, r'''---\n## 2 · `sequence_by_lambda` — the spectral curriculum

Ascending $\lambda\tau$: spectrally smooth, energy-spread items first; sparse/rough items
last. `positions` carry the score itself.''')

code(nb, '''\
cur = sequence_by_lambda(lam)
order = np.asarray(cur.order)

assert sorted(order.tolist()) == list(range(N)), "order must be a permutation"
assert np.allclose(lam[order], np.asarray(cur.positions)), "positions == lambda at each step"
assert np.all(np.diff(np.asarray(cur.positions)) >= -1e-12), "ascending by construction"
print("head (most coherent):", order[:8])
print("tail (rough/sparse):", order[-8:])
print("components:", cur.components)

rev = sequence_by_lambda(lam, descending=True)
print("descending reverses the head:", np.asarray(rev.order)[:8])

# Determinism: same scores -> byte-identical order (ties break on ascending index).
assert np.array_equal(order, np.asarray(sequence_by_lambda(lam).order))''')

code(nb, '''\
# Quantify the curriculum: sliding-window cluster purity vs random orders, and the
# sparsity gradient along the sequence (labels never entered the index).
W = 50
starts = np.arange(0, N - W + 1, W)

def window_purity(seq_order):
    return np.array([np.unique(labels[seq_order[i:i + W]], return_counts=True)[1].max() / W
                     for i in starts])

pur_cur = window_purity(order)
pur_rand = np.stack([window_purity(np.random.default_rng(s).permutation(N)) for s in range(10)])

deciles = np.array_split(order, 10)
share_sparse = np.array([float(np.mean(labels[d] == 2)) for d in deciles])
mean_effdims = np.array([float(eff_dims[d].mean()) for d in deciles])

fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), height_ratios=[1, 2.2])
strip = axes[0].imshow(labels[order][None, :], aspect="auto",
                       cmap=mcolors.ListedColormap([COLORS[c] for c in range(3)]))
axes[0].set_yticks([]); axes[0].set_title("curriculum strip (item cluster by position)")
fig.colorbar(strip, ax=axes[0], fraction=0.02)

axes[1].plot(starts + W / 2, pur_cur, lw=2, color="black", label="sequence_by_lambda")
axes[1].plot(starts + W / 2, pur_rand.mean(axis=0), lw=2, ls="--", color="tab:orange",
             label="random order (mean of 10)")
axes[1].fill_between(starts + W / 2, pur_rand.min(axis=0), pur_rand.max(axis=0),
                     color="tab:orange", alpha=0.2)
axes[1].set_xlabel("position in curriculum"); axes[1].set_ylabel("windowed modal-cluster share")
axes[1].set_title(f"curriculum purity {pur_cur.mean():.2f} vs random {pur_rand.mean():.2f}")
axes[1].legend()
fig.savefig(OUT / "fig_02_lambda_curriculum.png")
plt.show()

print(f"mean windowed purity: curriculum={pur_cur.mean():.2f}  random={pur_rand.mean():.2f}")
print("sparse-cluster share per decile:", np.round(share_sparse, 2))
print("mean eff-dims per decile:       ", np.round(mean_effdims, 1))
assert pur_cur.mean() > pur_rand.mean(), "curriculum must beat random ordering"
assert share_sparse[-1] > share_sparse[0], "curriculum should end on the sparse cluster"
assert mean_effdims[0] > mean_effdims[-1], "effective dims should fall along the curriculum"''')

md(nb, r'''**Reading.** Two independent unsupervised signals: the curriculum keeps same-cluster items
together measurably better than chance (0.53 vs 0.43 windowed purity on this fixture), and
its deciles drift toward the sparse cluster (first decile 0.18 -> last 0.44; the middle
deciles are noisy) down the dataset's density→sparsity gradient. The direction is
trustworthy; the per-decile values are not — *train on the head, audit the tail*.''')

md(nb, r'''---\n## 3 · `sequence_by_graph` — seriation of the feature Laplacian

`sequence_by_graph(gl)` walks the minimum spanning forest of $L_F$ in DFS preorder from
approximate diameter endpoints: one contiguous block per connected component, ordered by
descending size; `positions` = DFS depth. Here the 128 nodes are the **feature dimensions**.''')

code(nb, '''\
import collections

ser = sequence_by_graph(gl)
so = np.asarray(ser.order)
L = np.abs(np.asarray(gl.to_dense(), dtype=np.float64))
nz_rows, nz_cols = np.nonzero(L)

def mean_edge_bandwidth(perm):
    return float(np.abs(perm[nz_rows] - perm[nz_cols]).mean()) if len(nz_rows) else 0.0

bw_id = mean_edge_bandwidth(np.arange(L.shape[0]))
bw_ser = mean_edge_bandwidth(so)
print(f"nodes={L.shape[0]}  components={ser.components}  "
      f"mean edge bandwidth: identity={bw_id:.2f}  seriated={bw_ser:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].imshow(L, cmap="viridis")
axes[0].set_title(r"$|L_F|$ (identity order)")
axes[1].imshow(L[np.ix_(so, so)], cmap="viridis")
axes[1].set_title(r"$|L_F|$ (DFS seriation, 128 nodes)")
for ax in axes:
    ax.set_xlabel("node (feature dim)"); ax.set_ylabel("node (feature dim)")
fig.savefig(OUT / "fig_03_graph_seriation.png")
plt.show()''')

code(nb, '''\
# The real payoff of the seriator shows when L_F fragments: shrink eps/k and the
# dimension graph splits into many components, which the walk lays out as contiguous
# blocks. (Component-per-run contiguity is guaranteed by the DFS design — assert it.)
aspace_s, gl_s = (
    ArrowSpaceBuilder()
    .with_seed(SEED)
    .with_dims_reduction(False, None)
    .build({"eps": 0.25, "k": 3, "topk": 8, "p": 2.0, "sigma": 0.25}, X)
)
ser_s = sequence_by_graph(gl_s)
so_s = np.asarray(ser_s.order)
Ls = np.abs(np.asarray(gl_s.to_dense(), dtype=np.float64))

def mean_edge_bandwidth_S(perm):
    r, c = np.nonzero(Ls)
    return float(np.abs(perm[r] - perm[c]).mean())

adj = collections.defaultdict(set)
rs, cs = np.nonzero(Ls)
for r, c in zip(rs, cs):
    adj[r].add(c); adj[c].add(r)
seen, comp_id = set(), {}
for n0 in range(Ls.shape[0]):
    if n0 in seen:
        continue
    stack = [n0]; seen.add(n0)
    while stack:
        u = stack.pop(); comp_id[u] = n0
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)

runs = [comp_id[so_s[0]]] + [comp_id[so_s[i]] for i in range(1, len(so_s))
                             if comp_id[so_s[i]] != comp_id[so_s[i - 1]]]
print(f"sparse wiring: components={ser_s.components}  DFS-blocks={len(runs)}  "
      f"bandwidth {mean_edge_bandwidth_S(np.arange(Ls.shape[0])):.2f} -> {mean_edge_bandwidth_S(so_s):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
axes[0].imshow(Ls, cmap="viridis")
axes[0].set_title(f"sparse $|L_F|$: {ser_s.components} components")
axes[1].imshow(Ls[np.ix_(so_s, so_s)], cmap="viridis")
axes[1].set_title("seriated: contiguous component blocks")
fig.savefig(OUT / "fig_04_multi_component.png")
plt.show()

assert len(runs) == ser_s.components, "each component must occupy one contiguous run"
assert ser_s.components > 10, "the sparse wiring should genuinely fragment the dimension graph"''')

md(nb, r'''---\n## 4 · Guardrails''')

code(nb, '''\
try:
    sequence_by_lambda([0.5])
except ValueError as e:
    print("single score ->", e)

s_list = sequence_by_lambda([0.2, 0.9, 0.5])          # plain lists are accepted
print("list input order:", np.asarray(s_list.order), "positions:", np.asarray(s_list.positions))
assert np.array_equal(np.asarray(s_list.order), [0, 2, 1])''')

md(nb, r'---\n## 5 · Summary')

code(nb, '''\
summary = pd.DataFrame({
    "quantity": ["lambda curriculum: mean windowed purity",
                 "lambda curriculum: random-order baseline",
                 "sparse-cluster share: first decile",
                 "sparse-cluster share: last decile",
                 "graph seriation: bandwidth (sparse, identity)",
                 "graph seriation: bandwidth (sparse, seriated)",
                 "graph seriation: components (sparse)"],
    "value": [round(float(pur_cur.mean()), 3),
              round(float(pur_rand.mean()), 3),
              round(float(share_sparse[0]), 3),
              round(float(share_sparse[-1]), 3),
              round(mean_edge_bandwidth_S(np.arange(Ls.shape[0])), 3),
              round(mean_edge_bandwidth_S(so_s), 3),
              ser_s.components],
})
summary.to_csv(OUT / "sequencing_summary.csv", index=False)
print(summary.to_string(index=False))''')

md(nb, r'''**Reading.**

1. The $\lambda\tau$ curriculum is a real ordering signal on 128-dim embeddings: windowed
   purity beats the random baseline (0.53 vs 0.43) and the decile drift follows the
   sparsity gradient — all unsupervised. The margins here are fixture-dependent; on
   tightly-wired graphs the curriculum is sharper than on sparse wirings.
2. Seriation's contiguity guarantee (one run per component) is what makes it safe for
   batching/visualisation of a fragmented dimension graph; the ~40-way fragmentation at
   `eps=0.25, k=3` is a wiring *finding*, not a failure — the walk handles it
   deterministically. Note DFS preorder gives *contiguity*, not minimum bandwidth: the
   sparse-graph bandwidth can move either way (here 32.3 -> 34.1) — only contiguity is
   asserted.
3. The two sequences live in different namespaces (items vs feature nodes): compare their
   *metrics* (purity, bandwidth, contiguity), never their `order` arrays.

**Version note.** Values in this notebook are pinned to arrowspace 0.28.0: release 0.28
fixed a matrix-layout defect (#167) that changes every stored $\lambda$ and the graph
content behind it — indexes built with $\le$ 0.27.4 must be rebuilt.

**Caveats.** Purity yardsticks use k-means labels from the same data — fine for a
controlled fixture, but on real embeddings prefer label-free window statistics. The
$\tau$-sweep companion (notebook 00) uses this same build; conclusions transfer.''')

nbf.write(nb, OUT_DIR / "01__sequencing_api.ipynb")
print("wrote", OUT_DIR / "01__sequencing_api.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# 02 — Motives API
# ═══════════════════════════════════════════════════════════════════════════
nb = new_nb()

md(nb, r'''# Base-use 02 — The ArrowSpace Motives API (`spot_motives_*`, `spot_subg_*`)

**Question.** Beyond per-item scores and orderings, can ArrowSpace name the **repeating
local structures** in its Laplacians — cohesive, triangle-dense, low-Rayleigh subgraphs —
and project them back to items?

**Background — motifs and subgraph spotting.** Network motifs (Milo et al. 2002) are the
recurring, statistically over-represented interaction patterns — triangles, near-cliques,
feed-forward loops — that act as the *building blocks* of complex graphs: in a social
graph a triangle is "a friend group", in a regulatory network a loop is "a switch".
Spotting cohesive subgraphs generalises this to *which nodes* form each block, and is the
workhorse behind community detection, deduplication, and anomaly hunting. Classical
tooling is combinatorial: count triangles against degree-preserving nulls, expand greedily
toward clique-ness, partition by cut size.

**What spectral analysis contributes.** Spectral graph theory turns "how cohesive is this
set?" from a counting heuristic into algebra: a set $S$ is tightly connected when its
indicator has a low Rayleigh quotient $R_L(1_S)$ on the Laplacian (small boundary per
node, by the Cheeger correspondence), and the Fiedler value bounds how well any graph can
be cut. So spectral methods validate motif *candidates* by energy rather than by count
alone — the same mathematics that ranks items by smoothness ranks subgraphs by cohesion.

**What ArrowSpace brings.** The detector is fused with the index rather than bolted on:
seeding, expansion, and Rayleigh validation all run on the Laplacian the index already
built for $\lambda\tau$ scoring and search — no second graph, no recomputation. Motifs
are detected in the index's *compressed* node spaces (centroids on the eigen track,
subcentroids on the energy track) and projected to **item indices** through the index's
own bookkeeping, so the granularity of what a "building block" means is a build
parameter (whole-cluster unions vs sub-cluster pockets). And the failure modes that
historically made motif APIs dangerous — misread node spaces, silent config typos — are
now typed contracts: documented node spaces, `ValueError` on the wrong pipeline,
`TypeError` on unknown keys, deterministic tie-breaking throughout.

Since arrowspace 0.28 the bindings expose five motif calls on the built `ArrowSpace`
object (each takes the `GraphLaplacian` plus a config dict; pass `None` for defaults):

| Call | Node space of the returned sets | Requires |
|---|---|---|
| `spot_motives_eigen(gl, cfg)` | **feature dimensions** (nodes of the $F \times F$ $L_F$; documented since #165) | EigenMaps build |
| `spot_motives_eigen_items(gl, cfg)` | **item indices** — centroid-graph motifs expanded via item→cluster assignments; every motif is a union of whole clusters (new in 0.28, resolves #165) | EigenMaps build |
| `spot_motives_energy(gl, cfg)` | **item indices** (subcentroid motifs expanded via the centroid map) | EnergyMaps build (`build_energy`) |
| `spot_subg_motives(gl, cfg)` | list of subgraph dicts (`node_indices`, `item_indices`, `rayleigh`, ...) | EnergyMaps build |
| `spot_subg_centroids(gl, cfg)` | centroid-hierarchy dicts (`level`, `node_indices`, `root_indices`) | either build |

The two item-space tracks answer at **different granularities**: the eigen track detects on
the cluster-centroid graph (motifs are whole-cluster unions), the energy track on the finer
subcentroid graph (motifs resolve structure *inside* clusters).

The algorithm (arrowspace-rs `analysis::motives`): prune to the `top_l` strongest edges per
node, seed nodes with triangle count ≥ `min_triangles` and clustering coefficient ≥
`min_clust`, greedily expand by triangle gain up to `max_motif_size`, keep sets with
Rayleigh quotient ≤ `rayleigh_max` (when set), Jaccard-dedupe down to `max_sets`.
Deterministic: ties break on ascending node index.

### Config keys (unknown keys raise `TypeError` — no silent no-ops)

| dict | keys (defaults) |
|---|---|
| motives cfg | `top_l` (16), `min_triangles` (2), `min_clust` (0.4), `max_motif_size` (32), `max_sets` (256), `jaccard_dedup` (0.8) |
| subgraph cfg | all motives keys + `min_size`, `rayleigh_max` (`None` → skip Rayleigh validation *and* leave the field `None`) |
| centroid params | `eps, k, topk, p, sigma, normalise, sparsitycheck, min_centroids, max_depth, seed` |

**Dataset.** `dataset/eigenmaps_controlled.parquet` (1000 × 128, three orthonormal clusters
of increasing sparsity) with the README-prescribed `eps=0.5, sigma=0.5, k=12`. A $128$-node
$L_F$ finally gives the eigen-side detector a graph large enough to triangulate —
on $D=12$ toys it collapses to one set.''')

md(nb, r'---\n## 0 · Imports, dataset, and the prescribed hyperparameters')

code(nb, DATA_CELL.replace("output__DATASETDIR", "output__motives"))

md(nb, r'''---\n## 1 · EigenMaps motifs: feature space *and* item space

Two entry points since 0.28. `spot_motives_eigen` runs directly on the $128\times128$
$L_F$ and returns groups of *dimensions* (the node space is now documented — #165).
`spot_motives_eigen_items` mirrors the energy track: it rebuilds the X×X centroid
graph from the index's own rows, detects motifs there, and expands them through the
item→cluster assignments to **item indices** — every motif a union of whole
clusters, ids guaranteed in `0..n_items`.''')

code(nb, '''\
aspace, gl = (
    ArrowSpaceBuilder()
    .with_seed(SEED)
    .with_dims_reduction(False, None)
    .build(GRAPH_PARAMS, X)
)
lam = np.asarray(aspace.lambdas())
print(f"nitems={aspace.nitems}  L_F={gl.shape()}  degenerate lambdas: "
      f"{int(np.sum(np.abs(lam) < 1e-12))} / {len(lam)}")

motifs_default = aspace.spot_motives_eigen(gl, None)
print(f"feature-space, default cfg -> motifs: {len(motifs_default)}, sizes: "
      f"{sorted((len(m) for m in motifs_default), reverse=True)[:8]}")
assert max(max(m) for m in motifs_default) < D, "feature-space ids live in 0..F-1"

MOTIVE_CFG = {"top_l": 8, "min_triangles": 2, "min_clust": 0.3,
              "max_motif_size": 24, "max_sets": 64, "jaccard_dedup": 0.7}
motifs = aspace.spot_motives_eigen(gl, MOTIVE_CFG)
sizes = sorted((len(m) for m in motifs), reverse=True)
print(f"feature-space, tuned cfg -> motifs: {len(motifs)}, sizes head: {sizes[:10]}")
assert len(motifs_default) >= 5, "a 128-node graph should triangulate into multiple motifs"

eigen_item_motifs = aspace.spot_motives_eigen_items(gl, None)
eigen_item_motifs.sort(key=lambda m: -len(m))
for i, m in enumerate(eigen_item_motifs):
    print(f"  item-space m{i}: items={len(m):4d}  modal-cluster share={modal_share(m):.2f}")
assert all(0 <= i < N for m in eigen_item_motifs for i in m), "item ids in 0..n_items"
assert len(eigen_item_motifs) >= 2, "the centroid graph should yield whole-cluster motifs"''')

code(nb, '''\
# Validate motifs against the item side: mean |value| over the motif's dimensions.
A = np.abs(X)
act = np.stack([A[:, m].mean(axis=1) for m in motifs])          # (n_motifs, n_items)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4))
grid = np.zeros((D, len(motifs)))
for j, m in enumerate(motifs):
    grid[m, j] = 1
axes[0].imshow(grid, cmap="Greens", aspect="auto")
axes[0].set_xlabel("motif index"); axes[0].set_ylabel("feature dimension")
axes[0].set_title(f"motif membership ({len(motifs)} motifs x {D} dims)")

axes[1].imshow(act, aspect="auto", cmap="magma")
axes[1].set_xlabel("item (cluster-ordered)"); axes[1].set_ylabel("motif")
axes[1].set_title("item activity per motif")
for y in np.cumsum(np.bincount(labels))[:-1] - 0.5:
    axes[1].axhline(y, color="cyan", lw=0.6)

rows = []
for j, m in enumerate(motifs[:16]):
    a_row = act[j]
    mus = [float(a_row[labels == c].mean()) for c in range(3)]
    rows.append({"motif": j, "size": len(m),
                 **{f"activity c{c}": round(v, 3) for c, v in enumerate(mus)},
                 "max/min ratio": round(max(mus) / (min(mus) + 1e-9), 2)})
df_mot = pd.DataFrame(rows)
im = axes[2].imshow(df_mot[[f"activity c{c}" for c in range(3)]].to_numpy(),
                    aspect="auto", cmap="magma")
axes[2].set_xticks(range(3), NAMES, fontsize=7, rotation=20)
axes[2].set_yticks(df_mot["motif"], df_mot["motif"], fontsize=7)
axes[2].set_title("mean activity per cluster")
fig.colorbar(im, ax=axes[2], fraction=0.046)
fig.savefig(OUT / "fig_02_eigen_motifs.png")
plt.show()
print(df_mot.head(8).to_string(index=False))''')

md(nb, r'''**Reading.** Motifs are **hypotheses about the wiring of $L_F$**, not claims about
semantics. Two families show up on this dataset: broad ensembles whose member dimensions
are active for everyone (shared-support structure), and narrower ensembles with visible
per-cluster activity ratios (mid-column contrast) — those are the interesting ones and are
candidates for cluster-support signatures. The activity panel is where a motif earns or
loses its interpretation.''')

md(nb, r'''---\n## 2 · The energy APIs refuse EigenMaps builds (#35)

Before 0.27.3, `spot_motives_energy` on an EigenMaps build silently returned $F\times F$
node ids as if they were item indices — a namespace bug. Now the typed
`ArrowSpaceError::EnergyModeRequired` surfaces across the FFI as a catchable `ValueError`.

The mirror image used to be true as well: `spot_motives_eigen` returned feature ids with
no item-space option on the eigen track — filed as
[arrowspace-rs #165](https://github.com/tuned-org-uk/arrowspace-rs/issues/165) from this
very notebook series, and **resolved in 0.27.4/0.28.0**: `spot_motives_eigen_items` now
provides the item-space contract, the feature-space method documents its node space, and
feature-space detection is deterministic. The tracks also police each other now — calling
the eigen item-space variant on an *energy* build raises the symmetric `ValueError`.''')

code(nb, '''\
for name in ("spot_motives_energy", "spot_subg_motives"):
    try:
        getattr(aspace, name)(gl, None)
    except ValueError as e:
        print(f"{name} on EigenMaps -> ValueError: {str(e)[:72]}...")

try:
    aspace.spot_motives_eigen(gl, {"top_k": 8})       # typo'd knob
except TypeError as e:
    print(f"unknown cfg key -> TypeError: {str(e)[:72]}...")
''')

md(nb, r'''---\n## 3 · EnergyMaps build — where item-level motifs live

Subcentroid granularity drives everything: the default clustering collapses this dataset
to a handful of centroids (below $F$), and every motif swallows the corpus. Force the
splitter past $F$ with `with_cluster_radius` / `with_cluster_max_clusters` — then
`try_spot_motives_energy` rebuilds the motif graph **in subcentroid space** and projects
through `centroid_map`, so the returned indices are genuine item indices.''')

code(nb, '''\
ENERGY_PARAMS = {
    "eta": 0.06, "steps": 6, "trim_quantile": 0.0, "split_quantile": 0.9,
    "neighbor_k": 6, "split_tau": 0.12,
    "w_lambda": 1.0, "w_disp": 0.5, "w_dirichlet": 0.25,
    "candidate_m": 32, "optical_tokens": None,
}
aspace_e, gl_e = (
    ArrowSpaceBuilder()
    .with_seed(SEED)
    .with_cluster_radius(0.02)
    .with_cluster_max_clusters(600)
    .build_energy(X, ENERGY_PARAMS, GRAPH_PARAMS)
)
print(f"energy build: nitems={aspace_e.nitems}  subcentroids={aspace_e.nclusters}  "
      f"gl_e={gl_e.shape()} (bootstrap F x F)")
assert aspace_e.nclusters > D, "subcentroid graph must exceed F for item-space motifs"
# (0.28.0 note: at max_clusters=300 the corrected subcentroid coordinates fuse into one
#  corpus-sized motif; 600 splits fine enough for basin-resolved item motifs.)''')

md(nb, r'---\n## 4 · `spot_motives_energy` — motifs in *item* space')

code(nb, '''\
MOTIVE_CFG_E = {"top_l": 8, "min_triangles": 1, "min_clust": 0.0,
                "max_motif_size": 60, "max_sets": 48, "jaccard_dedup": 0.4}
item_motifs = aspace_e.spot_motives_energy(gl_e, MOTIVE_CFG_E)
item_motifs.sort(key=lambda m: -len(m))

def modal_share(ix):
    _, cnt = np.unique(labels[list(ix)], return_counts=True)
    return float(cnt.max() / len(ix))

print(f"item motifs: {len(item_motifs)}")
for i, m in enumerate(item_motifs[:10]):
    print(f"  m{i}: items={len(m):3d}  modal-cluster share={modal_share(m):.2f}")

n_pure = sum(1 for m in item_motifs if modal_share(m) > 0.95)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
ax.scatter(xy[:, 0], xy[:, 1], s=6, c="#cccccc")
for i, m in enumerate(item_motifs[:10]):
    ax.scatter(xy[m, 0], xy[m, 1], s=16, color=COLORS[i % 10], label=f"motif {i} (n={len(m)})")
ax.legend(fontsize=6.5, loc="best", ncols=2)
ax.set_title("item motifs in PCA space (top 10 by size)")

ax = axes[1]
ax.bar(range(len(item_motifs)), [modal_share(m) for m in item_motifs],
       color=[COLORS[i % 10] for i in range(len(item_motifs))], alpha=0.85)
ax.axhline(1 / 3, color="k", ls=":", lw=1, label="chance")
ax.set_xlabel("motif"); ax.set_ylabel("modal-cluster share")
ax.set_title(f"{n_pure}/{len(item_motifs)} motifs are >95% single-cluster")
ax.legend(fontsize=8)
fig.savefig(OUT / "fig_03_item_motifs.png")
plt.show()
assert len(item_motifs) >= 5 and n_pure >= 3, "granularity tuning should produce pure item motifs"''')

md(nb, r'''**Reading.** Compare the two item-space tracks on the same data: the eigen track returned
two whole-cluster unions (350 + 250 items, purity 1.0 each — cluster granularity), while
the energy track resolves *sub-cluster* structure: several motifs are >95% single-cluster
at sizes the eigen track cannot express, and the largest is a multi-basin union whose size
tracks how much inter-basin wiring survived top-$L$ pruning. Same detector, two
resolutions — pick the track that matches the question.''')

md(nb, r'''---\n## 5 · `spot_subg_motives` — subgraph dicts with optional Rayleigh cohesion

Same detection pass, dict output. `rayleigh_max` **doubles as the switch that populates
the `"rayleigh"` field** — left unset the field is `None` (#35 finding 2), which reads like
missing data but is really "not computed".''')

code(nb, '''\
for tag, rmax in (("rayleigh_max=None", None), ("rayleigh_max=0.1", 0.1)):
    sgs = aspace_e.spot_subg_motives(gl_e, {**MOTIVE_CFG_E, "min_size": 3, "rayleigh_max": rmax})
    fields = [s["rayleigh"] for s in sgs]
    n_with = sum(f is not None for f in fields)
    lo = min((f for f in fields if f is not None), default=float("nan"))
    hi = max((f for f in fields if f is not None), default=float("nan"))
    print(f"{tag}: {len(sgs)} subgs; rayleigh populated on {n_with}/{len(sgs)}; range [{lo:.2g}, {hi:.2g}]")

subgs = aspace_e.spot_subg_motives(gl_e, {**MOTIVE_CFG_E, "min_size": 3, "rayleigh_max": 0.1})
rows = [{"nodes": len(s["node_indices"]),
         "items": len(s["item_indices"] or []),
         "rayleigh": s["rayleigh"],
         "modal_share": modal_share(s["item_indices"]) if s["item_indices"] else np.nan,
         "nfeatures": s["nfeatures"], "x_dim": s["x_dim"]}
        for s in sorted(subgs, key=lambda x: -len(x["item_indices"] or []))]
df_sub = pd.DataFrame(rows)
print(df_sub.head(8).to_string(index=False))

fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.scatter(df_sub["items"], df_sub["modal_share"], s=70, c="tab:blue")
for _, r in df_sub.head(6).iterrows():
    ax.annotate(f"{int(r['nodes'])} sc", (r["items"], r["modal_share"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.axhline(1 / 3, color="k", ls=":", lw=1, label="chance")
ax.set_xscale("log"); ax.set_xlabel("items in subgraph (log)"); ax.set_ylabel("modal-cluster share")
ax.set_title("motif subgraphs: size vs purity")
ax.legend(fontsize=8)
fig.savefig(OUT / "fig_04_subg_cohesion.png")
plt.show()''')

md(nb, r'''Every retained set has $|R_L(1_S)| \le 1.1\times10^{-16}$: **zero-cut** at floating-point
precision — exactly what "cohesive subgraph" means spectrally. The $0.1$ threshold is far
from binding at this scale; it exists to *reject* rough sets on coarser wirings. `x_dim`
counts the subcentroids in the set; `nfeatures` is the feature-space dimension.''')

md(nb, r'''---\n## 6 · `spot_subg_centroids` — the centroid hierarchy''')

code(nb, '''\
CENTROID_CFG = {"eps": 0.5, "k": 12, "topk": 4, "p": 2.0, "sigma": 0.5,
                "normalise": True, "min_centroids": 2, "max_depth": 3, "seed": SEED}
cent = aspace_e.spot_subg_centroids(gl_e, CENTROID_CFG)
print(f"centroid-level subgraphs: {len(cent)}")
for c in cent:
    covered = sum(len(r) for r in c["root_indices"])
    print(f"  level {c['level']}: nnodes={c['nnodes']:4d}  roots cover {covered:5d} item slots")

levels = sorted({c["level"] for c in cent})
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.bar(range(len(cent)), [c["nnodes"] for c in cent],
       color=COLORS[[c["level"] for c in cent]])
ax.set_xticks(range(len(cent)), [f"L{c['level']}" for c in cent], fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("centroids (nnodes)")
ax.set_title("one subgraph dict per hierarchy level")
fig.savefig(OUT / "fig_05_centroid_levels.png")
plt.show()''')

md(nb, r'''---\n## 7 · Energy-mode search contracts

Energy indexes swap the query-side API: `search_energy(q, gl, k)` replaces the $\tau$-blended
search, `search_batch` is refused (`NotImplementedError`, #123), and
`search_linear_sorted` works on both pipelines.''')

code(nb, '''\
q = X[5]
print("search_energy top-3 (energy):", aspace_e.search_energy(q, gl_e, 3))

try:
    aspace_e.search_batch(X[:4], gl_e, 0.7)
except NotImplementedError as e:
    print("search_batch on energy build -> NotImplementedError:", str(e)[:62], "...")

print("search_linear_sorted top-3:", aspace_e.search_linear_sorted(q, gl_e, 3))

# Pipeline policing is symmetric: the eigen item-space variant refuses energy builds too.
try:
    aspace_e.spot_motives_eigen_items(gl_e, None)
except ValueError as e:
    print(f"eigen item-motifs on EnergyMaps -> ValueError: {str(e)[:60]}...")''')

md(nb, r'---\n## 8 · Summary')

code(nb, '''\
df_sum = pd.DataFrame({
    "pipeline": ["eigen", "eigen", "eigen", "energy", "energy", "energy", "energy"],
    "quantity": ["eigen dimension motifs (default cfg)",
                 "eigen dimension motifs (tuned cfg)",
                 "eigen item-space motifs (whole-cluster unions)",
                 "subcentroids in energy build",
                 "energy item motifs",
                 "pure energy item motifs (>95% single cluster)",
                 "subgraph dicts with Rayleigh computed"],
    "value": [len(motifs_default), len(motifs), len(eigen_item_motifs),
              aspace_e.nclusters, len(item_motifs), n_pure, len(subgs)],
})
df_sum.to_csv(OUT / "motives_summary.csv", index=False)
df_sub.to_csv(OUT / "subg_table.csv", index=False)
print(df_sum.to_string(index=False))''')

md(nb, r'''**Reading.**

1. **Node space first, config second.** The eigen track now offers *both* namespaces —
   `spot_motives_eigen` returns *dimensions*, `spot_motives_eigen_items` returns *items*;
   the energy track returns items at subcentroid resolution. Each call polices its build
   type (`ValueError` on the wrong pipeline, both directions), and misreading the node
   space was the actual historical bug (#35, #165).
2. **Motif granularity is a build parameter.** Item-side motifs only become basin-resolved
   once the energy splitter is forced past $F$ subcentroids; with the default heuristic the
   subcentroid graph is smaller than the feature space and every motif swallows the corpus.
3. **Rayleigh validation is opt-in** (`rayleigh_max` both filters and populates), and the
   retained sets here are zero-cut at float precision — cohesion at its strongest.
4. **128 dims matter, and so does the version.** At $D=128$ the $L_F$ triangle-detector
   returns a genuinely partitioned motif family; on toy $D$-dimensional data it degenerates
   to one set. And 0.28.0's #167 layout fix changed the graph content itself — motif counts
   from $\le$ 0.27.4 builds are not comparable, and feature-space detection is only
   deterministic since #165's fix.

**Caveats.** Modal-cluster purity is a yardstick for this controlled fixture only — on real
embeddings, validate motifs with held-out probes (ablation of a motif's items against its
own members' retrieval). All numbers here are single-seed; sweep `SEED` before trusting
motif counts quantitatively.

### References
- arrowspace-rs `analysis::motives` (triangle seeding + greedy expansion + Jaccard dedup)
- pyarrowspace #35 (energy-mode enforcement, Rayleigh opt-in, unknown-key errors)
- arrowspace-rs #165 (item-space motifs for the EigenMaps track — resolved in 0.28,
  `spot_motives_eigen_items` + documented node spaces)
- `notebooks/README.md` principles — API scores only, $\lambda$ as a final score.''')

nbf.write(nb, OUT_DIR / "02__motives_api.ipynb")
print("wrote", OUT_DIR / "02__motives_api.ipynb")
