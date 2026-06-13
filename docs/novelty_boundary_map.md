# Novelty Boundary Map

## Crowded Territory
- Bigger data/model scaling.
- New benchmark only.
- Generic active learning or uncertainty.
- Combining a planner with a learned policy without a new state/action object.

## Claimed Boundary
Mechanism level sim2real contracts keeps action-critical alternatives explicit until a physical observation collapses them.

## What Would Falsify The Claim
If observed-only baselines match the adverse-mode coverage and closed-loop success of the proposed branch-aware mechanism, the paper should be revised or killed.

## v4 Falsification
The real MuJoCo rebuild falsifies the current claim. On combined reality gap, `mechanism_contract_mpc` reaches 0.500 success, while `residual_adaptation_mpc` reaches 0.683 and `scalar_residual_only` reaches 0.650. The full contract method's low false-accept rate is mostly caused by a 0.017 accept rate, not robust transfer.
