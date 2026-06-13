# Claims

- Mechanism claim under test: mechanism-level sim-to-real contracts should improve transfer by measuring contact onset, slip, force slope, energy dissipation, and actuator lag before accepting a simulated plan.
- Real-evidence result: the v4 MuJoCo benchmark falsifies this implementation. On combined reality gap, `mechanism_contract_mpc` reaches 0.500 success, below `residual_adaptation_mpc` at 0.683 and `conformal_transfer_filter` at 0.567.
- Safety result: the contract method has 0.000 false accepts on combined gap only because it accepts almost no transfers (0.017 accept rate).
- Ablation result: `scalar_residual_only` reaches 0.650 success, beating the full contract method.
- Scope claim: results support archiving this specific mechanism, not deployment.
- Unsupported claim explicitly avoided: no claim of SOTA robot performance.
