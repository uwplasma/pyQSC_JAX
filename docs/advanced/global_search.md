# Global-search semantics

The workflow is global-to-local, not a proof-producing global optimizer:

1. deterministic branch-specific and Halton seeds;
2. batched coarse evaluation;
3. damped local least-squares refinement;
4. normalized basin clustering;
5. hard criteria and conditioning filters;
6. independent resolution verification.

Only a verified zero reaches the known global lower bound of the nonnegative
mean-free \(B_{20}\) residual. Nonzero output is `best_found` within the
reported finite budget. The implementation never turns the best sampled
point into a global claim.

See the [full search derivation](../theory/global-search.md) for status values,
bounded variables, high-mode safeguards, and continuation across Fourier
stages.
