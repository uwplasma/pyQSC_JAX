# Field tensors

Regular transverse coordinates \(q_1=r\cos\vartheta\) and
\(q_2=r\sin\vartheta\) remove the polar-coordinate singularity on axis.
Applying the inverse-coordinate chain rule to the complete r2 expansion gives

\[
B_i,\qquad D_{ij}=\partial_jB_i,\qquad
H_{ijk}=\partial_j\partial_kB_i.
\]

Vacuum fields satisfy symmetric trace-free identities; finite-current total
fields retain derivative-index symmetry and Ampère-law antisymmetry in the
gradient. The result reports divergence, derivative asymmetry, and the
gradient of divergence.

See [total on-axis field jet](field-jet.md) for the full regular-map
derivation, periodic cylindrical connection terms, tensor order, scale
lengths, and singular-radius construction. Implementation:
`pyqsc_jax.field.total_field_jet`. Validation:
`tests/physics/test_field_jet.py`.
