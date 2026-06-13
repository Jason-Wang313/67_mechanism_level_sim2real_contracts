# Paper 67 Terminal Evidence

Decision: `KILL_ARCHIVE`

## Real-Evidence Rebuild
The v4 rebuild replaces the synthetic scaffold with a MuJoCo source-to-target transfer benchmark. Each episode executes a source-predicted probe, a target probe, contract-diagnostic comparisons, and a final target rollout.

Run command:

```powershell
python src\run_experiment.py
```

Generated evidence:
- 3,780 main MuJoCo episode rows.
- 420 ablation rows.
- 960 stress-sweep rows.
- 5 seeds, 12 main episodes per seed, 7 target splits, 9 main methods.
- CSVs: raw rollouts, metrics, seed metrics, pairwise comparisons, contract diagnostics, ablations, stress sweep, negative cases.
- Figures: success by split, false accepts, ablation success, stress sweep.

## Combined Reality-Gap Results

| Method | Success | CI95 | Error | Energy | False Accept | Accept |
|---|---:|---:|---:|---:|---:|---:|
| `random_push` | 0.100 | 0.077 | 0.157 | 0.433 | 0.900 | 1.000 |
| `source_sim_mpc` | 0.517 | 0.128 | 0.094 | 0.532 | 0.483 | 1.000 |
| `domain_randomized_mpc` | 0.500 | 0.128 | 0.100 | 0.531 | 0.500 | 1.000 |
| `ensemble_uncertainty_mpc` | 0.450 | 0.127 | 0.102 | 0.514 | 0.550 | 1.000 |
| `system_id_probe_mpc` | 0.500 | 0.128 | 0.091 | 0.565 | 0.500 | 1.000 |
| `residual_adaptation_mpc` | 0.683 | 0.119 | 0.071 | 0.531 | 0.317 | 1.000 |
| `conformal_transfer_filter` | 0.567 | 0.126 | 0.093 | 0.520 | 0.217 | 0.433 |
| `mechanism_contract_mpc` | 0.500 | 0.128 | 0.087 | 0.539 | 0.000 | 0.017 |
| `oracle_target_mpc` | 1.000 | 0.000 | 0.006 | 0.604 | 0.000 | 1.000 |

Pairwise combined-gap comparisons show `mechanism_contract_mpc` is below residual adaptation by 0.183 success and below conformal filtering by 0.067 success.

## Ablation Results

| Ablation | Success | CI95 | False Accept | Accept |
|---|---:|---:|---:|---:|
| `full_mechanism_contract_mpc` | 0.483 | 0.128 | 0.033 | 0.033 |
| `no_actuator_lag_contract` | 0.567 | 0.126 | 0.033 | 0.067 |
| `no_contact_onset_contract` | 0.567 | 0.126 | 0.233 | 0.417 |
| `no_energy_contract` | 0.550 | 0.127 | 0.033 | 0.067 |
| `no_force_slope_contract` | 0.433 | 0.126 | 0.050 | 0.083 |
| `no_slip_contract` | 0.600 | 0.125 | 0.017 | 0.083 |
| `scalar_residual_only` | 0.650 | 0.122 | 0.350 | 0.967 |

## Terminal Rationale
The central claim requires explicit mechanism contracts to improve transfer over scalar residuals, residual adaptation, conformal filtering, and domain randomization. They do not. The full contract method is too conservative, accepting almost no transfers, and scalar residuals achieve better success. The honest action is `KILL_ARCHIVE`.
