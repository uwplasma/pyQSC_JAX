# Literature cases

The validation matrix is tied to primary literature:

- Garren--Boozer existence and magnetic-field-strength papers for the
  near-axis framework
  ([DOI 10.1063/1.859916](https://doi.org/10.1063/1.859916),
  [DOI 10.1063/1.859915](https://doi.org/10.1063/1.859915));
- Landreman--Sengupta direct construction and high-order equations
  ([arXiv:1809.10233](https://arxiv.org/abs/1809.10233),
  [arXiv:1908.10253](https://arxiv.org/abs/1908.10253));
- Landreman's near-axis figures of merit
  ([arXiv:2012.00865](https://arxiv.org/abs/2012.00865));
- Rodríguez, Sengupta, and Bhattacharjee for magnetic shear
  ([DOI 10.1063/5.0076583](https://doi.org/10.1063/5.0076583));
- Curvo, Ferreira, and Jorge for the configurable screening profile
  ([DOI 10.1017/S002237782400165X](https://doi.org/10.1017/S002237782400165X));
- Giuliani et al. for single-stage near-axis coil optimization
  ([arXiv:2010.02033](https://arxiv.org/abs/2010.02033)).

The [physics traceability table](../development/physics-traceability.md)
connects every block to code and independent tests. Numerical constants are
never copied from prose without a test or local, checksummed reference.

## Public stellarator-database cases

The README design gallery is traceable to the
[University of Wisconsin stellarator database](https://stellarator.physics.wisc.edu/)
and its [interactive application](https://stellarator.physics.wisc.edu/app).
The app reports transform magnitude, while the JSON downloads retain the
signed convention. The lead QA case uses the requested
\(\lvert\iota\rvert\ge0.3\) gate; the three QH/optimization cases use
\(\lvert\iota\rvert\ge0.4\).

| bundled name | public source | role |
| --- | --- | --- |
| `database_qa_139524` | [ID 139524](https://stellarator.physics.wisc.edu/app/plot/139524) | one-period QA showcase with helicity zero and \(\lvert\iota\rvert>0.3\) |
| `database_example_3` | [ID 3](https://stellarator.physics.wisc.edu/app/plot/3) | documented API/download example |
| `database_low_b20_57409` | [ID 57409](https://stellarator.physics.wisc.edu/app/plot/57409) | low-\(B_{20}\) optimization seed |
| `b20_optimized_good` | derived from [ID 57409](https://stellarator.physics.wisc.edu/app/plot/57409) | constrained eight-mode refinement |
| `database_large_singularity_107579` | [ID 107579](https://stellarator.physics.wisc.edu/app/plot/107579) | large-singular-radius example |

Every case is solved again rather than trusting displayed database metrics.
Regression tests require all four displayed cases to pass the full
configurable Curvo profile at the stated transform gate. ID 139524 must also
retain helicity zero, one field period, exactly zero \(I_2\), finite pressure,
and RMS axis torsion above \(1\ \mathrm{m}^{-1}\). The derived ID-57409
refinement retains the source ID and URL in `ReferenceConfiguration`
metadata.
