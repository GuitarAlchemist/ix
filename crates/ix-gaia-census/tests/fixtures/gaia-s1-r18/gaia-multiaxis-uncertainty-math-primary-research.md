# Gaia multi-axis uncertainty: mathematical foundations

## Verdict

Gaia should represent uncertainty as a **typed product of partially ordered axes**, not as one confidence scalar. Keep evidence state, predictive uncertainty, decision sensitivity, provenance, and temporal validity distinct. Derive task-specific summaries only at decision boundaries.

## Recommended core

For a claim `c`, use a record conceptually like

`U(c) = (support, refutation, probability-set, aleatoric, epistemic, provenance, valid-time, decision-impact)`.

- Give each component its own order and combine records with the componentwise product order. This preserves incomparability: one claim may have stronger evidence but worse freshness. Joins/meets are legal only where the component domains supply them. A scalar rank is a policy projection, not the underlying mathematics.
- Model `support` and `refutation` as independent coordinates. Ginsberg's bilattice separates a truth order from a knowledge order and explicitly accommodates incomplete and inconsistent information; this is the right semantic precedent for “unknown” versus “both supported and contradicted” ([Ginsberg 1988](https://doi.org/10.1111/j.1467-8640.1988.tb00280.x)). Use a small finite bilattice for logical/evidential status, not for numeric probability.
- Represent probability uncertainty by a **credal set** (or coherent lower/upper expectations), not an arbitrary interval per event. Walley shows that coherent lower previsions / sets of probability measures are more general and decision-relevant than isolated lower/upper probabilities, belief functions, or possibility measures ([Walley 2000](https://doi.org/10.1016/S0888-613X(00)00031-1)). Store constraints or extreme distributions when tractable; expose lower/upper expectations for actions.
- Keep aleatoric uncertainty (irreducible observation/outcome noise) separate from epistemic uncertainty (model uncertainty potentially reducible with data). Kendall and Gal operationalize this distinction and jointly model both ([Kendall & Gal 2017](https://arxiv.org/abs/1703.04977)). Do not assume every domain permits a uniquely identifiable decomposition; label the estimator and assumptions.
- Make decisions with expected utility when a defensible distribution exists; under a credal set, report lower/upper expected utility and optionally choose **minimax regret** for ambiguity-sensitive, reversible choices. Regret compares an action with the statewise best action; its minimax origin is Savage's statistical decision formulation ([Savage 1951](https://doi.org/10.1080/01621459.1951.10500768)). Always display the worst-case witness distribution/state: robust does not mean probable.
- Prioritize investigation by **expected value of information (EVI) minus acquisition cost**, computed against the actual downstream decision. Information value depends jointly on probabilities and consequences and is non-additive even for independent unknowns ([Howard 1966](https://doi.org/10.1109/TSSC.1966.300074)). This is preferable to “research the least-confident axis first.”

## Evidence, calibration, time, and graphs

- Attach source identity, derivation, responsible agent, and activity to every evidence item and aggregation. W3C PROV's `Entity`/`Activity`/`Agent`, `wasDerivedFrom`, `used`, and attribution relations are a suitable interoperable minimum ([PROV-O Recommendation](https://www.w3.org/TR/prov-o/)). Dependence is then explicit: evidence derived from the same source must not be counted as independent corroboration.
- Give assertions transaction time plus valid-time interval (or instant), temporal precision, and an explicit expiry/revalidation policy. W3C OWL-Time provides instants, intervals, ordering/topological relations, reference systems, and precision ([OWL-Time](https://www.w3.org/TR/owl-time/)). Expiry should change epistemic status/freshness, not retroactively erase provenance.
- Propagate numeric uncertainty only through an explicit probabilistic dependency graph. Bayesian-network arcs encode direct dependencies and conditional probabilities; Pearl's propagation computes beliefs consistent with probability axioms on the supported graph class ([Pearl 1986](https://doi.org/10.1016/0004-3702(86)90072-X)). Preserve correlation/common-cause identifiers. For loops or unknown dependence, use bounds, sampling, or sensitivity analysis and label approximation; never multiply marginal confidences blindly.
- Evaluate probabilistic outputs with strictly proper scores, which uniquely incentivize reporting the believed distribution. Use log score or Brier score for categorical forecasts and CRPS/interval score where appropriate; also show reliability/calibration curves by axis, source class, and horizon ([Gneiting & Raftery 2007](https://doi.org/10.1198/016214506000001437)). Accuracy alone is insufficient.
- Use conformal prediction only for repeated predictive tasks with a stated exchangeability/i.i.d.-style assumption and held-out calibration data. It supplies marginal finite-sample coverage sets, not epistemic truth or causal confidence; the guarantee and assumption are explicit in Shafer and Vovk's original tutorial ([Shafer & Vovk 2008](https://jmlr.org/papers/volume9/shafer08a/shafer08a.pdf)). Monitor coverage under drift and stratify when conditional performance matters.

## Rejected as Gaia defaults

- **Single confidence number:** destroys partial-order incomparability, hides conflict, and cannot distinguish reducible uncertainty, noise, staleness, or decision impact.
- **Unqualified Dempster–Shafer normalization:** Dempster's construction yields lower/upper probabilities and a combination rule ([Dempster 1967](https://doi.org/10.1214/aoms/1177698950)), but normalized combination divides by `1-K`; it is undefined at total conflict (`K=1`) and can amplify tiny surviving intersections when conflict is near one. Gaia should retain conflict mass and provenance, refuse automatic fusion above a policy threshold, and use credal/sensitivity analysis unless source independence is justified.
- **Fuzzy membership as probability:** degree of category membership is not frequency, belief, or evidential support; mixing them makes calibration meaningless.
- **Naive averages, votes, or products of confidence:** unjustified under shared provenance, correlated errors, heterogeneous semantics, or stale evidence.
- **Minimax alone:** often excessively conservative and insensitive to likelihood. Use it as a displayed ambiguity policy, not as universal aggregation.
- **Conformal sets as universal confidence:** coverage is distributional and marginal under assumptions; it does not validate one-off research claims or guarantee conditional coverage.
- **Time-decay multiplication:** a universal exponential discount invents probabilistic meaning. Prefer explicit validity intervals and domain-specific hazard/revalidation models.

## Multi-axis geometry for semantic navigation

Treat the full typed uncertainty/evidence state as authoritative and any 2-D/3-D embedding as a lossy view.

- **Linear baseline:** PCA finds the affine subspace minimizing squared orthogonal reconstruction error (the closest-fit construction originates with [Pearson 1901](https://doi.org/10.1080/14786440109462720)). Use it when axes are numeric, commensurately scaled, and a global linear summary is meaningful. Report explained variance and reconstruction error; never feed nominal/provenance fields into Euclidean PCA by arbitrary coding.
- **Local linearization/manifolds:** a smooth manifold is approximately its tangent space only locally. LLE preserves local reconstruction weights while producing one global coordinate system ([Roweis & Saul 2000](https://doi.org/10.1126/science.290.5500.2323)); Isomap estimates manifold geodesics through a neighborhood graph before embedding ([Tenenbaum, de Silva & Langford 2000](https://doi.org/10.1126/science.290.5500.2319)). Use only after checking neighborhood stability across `k`, bootstrap samples, and metric choices; sparse sampling, branching, boundaries, shortcuts, or mixed discrete/continuous strata violate the smooth-manifold picture.
- **Nonlinear neighbor maps:** t-SNE optimizes local similarity preservation and is explicitly a visualization technique ([van der Maaten & Hinton 2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html)); UMAP constructs a weighted neighborhood object under manifold assumptions and optimizes a low-dimensional representation ([McInnes, Healy & Melville 2018](https://arxiv.org/abs/1802.03426)). Cluster spacing, area, direction, and apparent empty regions in either map are not calibrated semantic distances or probabilities. Compare seeds and hyperparameters.
- **Discrete graph structure:** when semantic relations are inherently edges (derivation, contradiction, dependency, temporal succession), retain the graph. Laplacian Eigenmaps use the sampled graph Laplacian to preserve locality and connect embedding with clustering ([Belkin & Niyogi 2001](https://papers.nips.cc/paper_files/paper/2001/file/f106b7f99d2cb30c3db1c3cc0fde9ccb-Paper.pdf)). An embedding must not replace edge types, direction, provenance, or partial-order relations; show those as overlays or linked detail.
- **Distortion audit:** publish neighborhood **trustworthiness** (displayed neighbors that are genuine) and **continuity** (genuine neighbors retained), which necessarily trade off in nonlinear visualization ([Venna & Kaski 2006](https://doi.org/10.1016/j.neunet.2006.05.014)). Also report global stress/distance correlation where global distance is claimed, graph-component/topology changes, and uncertainty bands from resampling. “Topology preservation” must name the invariant tested (components, neighbor graph, cycles), not rely on visual resemblance.
- **Out-of-sample points:** prefer an explicit fitted linear/parametric map. For spectral/manifold coordinates, use a declared extension such as Nyström/geometric harmonics, whose validity depends on scale and proximity to the sampled manifold ([Coifman & Lafon 2006](https://doi.org/10.1016/j.acha.2005.07.005)). Flag extrapolation and distribution drift; do not silently refit a map whose moving coordinates imply false semantic change.

**Gaia recommendation:** use PCA as the reproducible diagnostic baseline; retain a typed k-nearest-neighbor/evidence graph as the navigation substrate; optionally offer UMAP or t-SNE as a seeded exploratory lens with visible distortion metrics. Selecting a point must always reveal original axis values, provenance, neighbors in the source metric, valid time, and projection version. Never use projected distance, cluster membership, or visual separation alone to fuse evidence, rank truth, approve an action, or calculate confidence.

## Minimal implementation rules

1. Define axis types, direction of order, units, and `unknown` separately from extrema.
2. Require provenance and valid-time on every evidence edge; track dependence groups.
3. Aggregate only like-typed quantities under declared assumptions; otherwise preserve a set/bounds and conflict.
4. Make every scalar dashboard score a named, versioned decision policy with visible component values.
5. Log forecasts before outcomes; audit proper scores, calibration, interval coverage, and decision regret by cohort and time horizon.
6. Trigger research when net EVI is positive; trigger human review for high impact, wide robust-decision disagreement, provenance dependence, or high conflict.

GAIA_MULTIAXIS_UNCERTAINTY_MATH_PRIMARY_RESEARCH_COMPLETE
