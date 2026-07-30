# Limitations

The current released computational core is first order only. Although the
legacy constructor accepts `order`, `B2c`, and `p2`, those arguments do not yet
activate second-order physics.

Known limitations under active refactor include:

- exactly five Newton updates without a convergence report;
- only stellarator-symmetric `rc`/`zs` axes;
- a confirmed normal/binormal unpacking defect in the mutable `dofs` setter;
- no complete second- or third-order solution;
- no plasma/external field-jet separation;
- no branch-aware target-transform inverse solve.

Frenet coordinates are invalid when the magnetic-axis curvature vanishes. The
new geometry API will report that validity condition explicitly.

Near-axis results are asymptotic in distance from the axis. “Surface-free”
plasma–coil separation does not mean radius-free: a formal minor radius or an
equivalent flux/current normalization is required.
