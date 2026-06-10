# ArrowSpace Local Minima Experiments

This folder contains notebooks used to probe how ArrowSpace spectral indexing behaves on synthetic manifolds and real-world embeddings. The goal is to quantify **how well ArrowSpace λ-scores expose semantic basins and boundaries**, and how they should be correctly combined (or not combined) with vanilla geometric algorithms.

The design principles below apply to all notebooks, with notebook `01__arrowspace_local_minima.ipynb` acting as the canonical reference implementation.

---

## 0. Always use the ArrowSpace API (pyarrowspace)

These experiments are intended to validate **ArrowSpace itself**, not a reimplementation of its behaviour in NumPy.

All λ-scores used for evaluation **must come from the ArrowSpace/pyarrowspace API**, e.g. `aspace.search(...)`, `aspace.lambda_scores(...)`, or similar calls, as demonstrated in the [`pyarrowspace` tests](https://github.com/tuned-org-uk/pyarrowspace/tree/main/tests).[web:36]

**Principle 0.** Do not reconstruct λ, Laplacians, or eigen-decompositions manually when measuring ArrowSpace performance. Instead:

- Build or load an `ArrowSpace` index from embeddings.
- Obtain λ-scores directly from the ArrowSpace API.
- Use NumPy/Pandas/Plotly only for **analysis and visualisation** of those scores, not for re-implementing ArrowSpace internals.

Synthetic experiments (such as `01__arrowspace_local_minima.ipynb`) should therefore:

1. Generate embeddings and labels.
2. Construct an ArrowSpace index via Python `arrowspace` (github.com/tuned-org-uk/pyarrowspace).
3. Query λ-scores via the same public API that production code would use.
4. Only then compute basins/boundaries and quality metrics.

The main objective is to measure the effectiveness of arrowspace and its $$\lambda$$ score as an improved metrics over cosine and other geometric methods. While cosine remains the baseline, the main objective is to establish $$\lambda$$ as the actual semantic search while geometric methods are considered geometric search. 

---

## 1. Treat λ as a final score, not a feature

ArrowSpace search exposes a λ-score per item

$$
\lambda_w(x) = w \cdot \text{geom}(x) + (1-w) \cdot \text{spec}(x)\, , \quad w \in [0,1]
$$

- `geom(x)` is the geometric / low-frequency component (smooth on the feature graph).
- `spec(x)` is the spectral / high-frequency component (boundary, transition, anomaly signal).[file:11][file:7]

**Principle 1.** For any given `w`, $$\lambda_w$$ is a *final* similarity / energy score and should be **compared directly** with vanilla metrics (cosine search, KDE, diffusion maps, basin hopping, etc.), not embedded again into a new linear combination with another geometric score.[file:6][cite:52]

Corollary:
- When we want to see "how ArrowSpace search behaves" for a given `w`, we evaluate $$\lambda_w$$ directly against baselines, using common quality metrics (cluster purity, MRR-Top0, NDCG, etc.).[file:2][file:4]

---

## 2. Separate geometric and spectral components explicitly

In synthetic settings (e.g. notebook 01), we may approximate the λ decomposition (if needed for analysis only) by eigendecomposing the feature-space Laplacian

$$ L = \Phi \, \Lambda \, \Phi^\top $$

and splitting eigenmodes into:

- **Geometric subspace**: low eigenvalues (smooth modes) → geometric component $$R_{\text{geom}}$$.
- **Spectral subspace**: high eigenvalues (rough modes) → spectral component $$R_{\text{spec}}$$.

We then define three per-item energies for analysis:

- `R_geom`: geometric-only Rayleigh energy ($$w = 1.0$$ analogue).
- `R_spec`: spectral-only Rayleigh energy ($$w = 0.0$$ analogue).
- `lambda_full`: full ArrowSpace λ (blended, $$w \approx 0.5$$ analogue).[cite:52]

**Principle 2.** All experiments must:

1. Compute `lambda_full` via the ArrowSpace API once as the main ArrowSpace score.
2. Compute `R_geom` and `R_spec` only as **diagnostic/analysis views** if we need to reason about geometry vs spectrum.
3. Clearly label which quantity is being used (λ full vs geometric-only vs spectral-only) in tables and plots.[file:6][file:7]

---

## 3. Only spectral components may augment vanilla algorithms

Vanilla algorithms in these notebooks operate in item-space only:

- KDE: density in PCA space, gradient ascent modes, and anti-modes.
- Diffusion Maps: Markov diffusion basins and distances to diffusion centroid.
- Basin Hopping: local minima of $$-\log \text{KDE}$$ in 2D.

They are all **geometric** methods. ArrowSpace λ already contains a geometric term. If λ is added directly to a vanilla score, geometry is counted twice.[cite:52]

**Principle 3.** When combining ArrowSpace with a vanilla algorithm, we must:

- Use **only the spectral component** $$R_{\text{spec}}$$ as the ArrowSpace term.
- Keep the vanilla score as the sole geometric term.

Concretely, for a vanilla score $$v(x)$$:

$$
\text{aug}(x) = \alpha \, v(x) + (1-\alpha) \, R_{\text{spec}}(x)\, , \quad \alpha \in [0,1]
$$

This implements a clean **spectral augmentation**:

- $$\alpha = 1$$: pure vanilla geometric algorithm.
- $$\alpha = 0$$: pure spectral ArrowSpace signal on top of the same graph.

We never use `lambda_full` inside this formula; we reserve it for direct comparisons.

---

## 4. Boundary and basin evaluation

The main question these notebooks answer is: *Can ArrowSpace correctly highlight semantic basins and boundaries in embedding space?* On synthetic data, cluster labels act as ground truth for basins; on real embeddings, we use topology-aware metrics such as MRR-Top0 and tail/head behaviour.[file:6][file:2]

In synthetic experiments:

- Basins are items in dense regions or diffusion attractors.
- Boundaries are low-density anti-modes or high-λ items on the feature manifold.

**Principle 4.** Each experiment must quantify, at minimum:

1. **Cluster purity** of the minima set (fraction of items in the dominant cluster).
2. **Mean full λ** (`lambda_full`) inside the minima set (low = on-manifold basin, high = off-manifold / boundary).
3. **Jaccard overlap** between ArrowSpace minima and vanilla minima.

This holds for:

- ArrowSpace alone (bottom $$k\%$$ of `lambda_full`).
- Each vanilla method alone.
- Each vanilla + spectral(ArrowSpace) variant.[cite:52]

Boundary analysis is interpreted as:

- If boundaries are correctly highlighted, **vanilla + spectral(AS)** should:
  - Preserve vanilla boundary-level scores where $$R_{\text{spec}}$$ is small.
  - Sharpen basins by lowering `lambda_full` in the selected set.

---

## 5. α sweeps as a spectral dial

To understand the interaction between vanilla geometry and ArrowSpace spectrum, notebooks sweep

$$ \alpha \in [0,1] $$

in the spectral-only augmentation formula and track:

- Cluster purity vs α.
- Mean full λ vs α.

**Principle 5.** α sweeps are always interpreted as:

- α → 1: "turn off" ArrowSpace and keep only vanilla.
- α → 0: "turn off" vanilla and keep only ArrowSpace spectral component.

We look for:

- Purity peaks at intermediate α → **optimal spectral correction**.
- Monotonic increase of mean λ as α → 1 → **spectral signal is not redundant**.

Results are visualised with line plots per method (KDE, DiffMaps, BasinHop) and grouped bar charts across methods.[cite:52]

---

## 6. Independence checks between signals

Any claim that ArrowSpace adds information must be backed by independence checks:

- Scatter plots of `R_spec` vs KDE score (`1 − density`).
- Scatter plots of `R_spec` vs diffusion distance.
- Pearson correlation values annotated on the plots.[cite:52]

**Principle 6.** If the correlation between `R_spec` and a vanilla score is near zero (or at most weak), then spectral augmentation is justified:

- The spectral component is **not** a disguised copy of the vanilla metric.
- Blending them cannot be replaced by a rescaling of the vanilla score.

---

## 7. Reproducibility and wiring invariants

The notebooks follow ArrowSpace wiring invariants from the main design documents:[file:11][file:13][file:15]

- Feature graph L is built once per experiment from high-dimensional embeddings (or imported from an ArrowSpace index).
- k-NN wiring in feature-space is symmetric and uses cosine similarity, matching ArrowSpace defaults.
- All Rayleigh energies are normalised to $$[0,1]$$ before comparisons.
- Random seeds are fixed for synthetic data and stochastic algorithms.

**Principle 7.** Any change to graph wiring (k, similarity function, normalisation) or to the λ decomposition must be:

- Explicitly described in the notebook markdown.
- Reflected in the experiment metadata and chart captions.

This keeps ArrowSpace experiments comparable across notebooks and over time.

---

## How to extend these experiments

When adding a new notebook or metric, align with the principles above:

1. Decide whether you are doing **direct λ comparison** or **spectral-only augmentation**.
2. Obtain all λ-scores from the `pyarrowspace` API (do not roll your own λ in NumPy).
3. If augmenting, extract `R_spec` explicitly (or an equivalent spectral-only score from ArrowSpace) and *never* add `lambda_full` on top of a geometric baseline.
4. Always report cluster purity, mean full λ, and Jaccard overlaps for minima sets.
5. Include at least one α sweep and one independence scatter plot per new method.

Following these rules keeps ArrowSpace analysis consistent with the underlying definition of λ search and with the graph wiring semantics laid out in the ArrowSpace and MRR-Top0 materials.[file:6][file:7][file:2]