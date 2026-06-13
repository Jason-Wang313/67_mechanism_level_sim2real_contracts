# Paper 67 Rebuild Plan: Mechanism-Level Sim2Real Contracts

## Terminal Objective
Rebuild `67_mechanism_level_sim2real_contracts` into a real evidence package. The paper may be submission-ready only if mechanism-level transfer contracts predict and improve sim-to-real manipulation transfer beyond strong domain-randomization, uncertainty, system-identification, residual-adaptation, and conformal-risk baselines. If the contract mechanism is matched by those baselines, archive it.

## Central Claim Under Test
Observation-level matching and generic uncertainty are too coarse for contact-rich sim-to-real transfer. A planner should transfer only when measurable mechanism contracts hold, such as contact onset, slip ratio, force-displacement slope, energy dissipation, and actuator lag. When contracts fail, the planner should adapt, choose a conservative action, or reject transfer.

## High-Fidelity Benchmark
- Build a lightweight MuJoCo planar pushing benchmark with a source simulator and target "real" variants.
- Source simulator: nominal object mass, friction, actuator response, contact stiffness, and sensor noise.
- Target variants:
  - nominal target
  - high-friction target
  - low-friction target
  - mass/compliance shift
  - actuator-lag shift
  - contact-sensor bias
  - combined reality gap
- Each episode executes a diagnostic probe and a final push in MuJoCo. The source model predicts mechanism features; the target rollout supplies the actual features and success outcome.

## Methods And Baselines
- `random_push`: lower bound.
- `source_sim_mpc`: transfers the source-simulator plan without adaptation.
- `domain_randomized_mpc`: robustly plans over randomized friction/mass/lag ranges.
- `ensemble_uncertainty_mpc`: chooses actions with low predicted variance over model samples.
- `system_id_probe_mpc`: uses the diagnostic probe to fit friction/mass proxies before planning.
- `residual_adaptation_mpc`: corrects predicted displacement with recent probe residuals.
- `conformal_transfer_filter`: rejects or shortens actions when probe residuals exceed calibration thresholds.
- `mechanism_contract_mpc`: proposed method; evaluates explicit mechanism contracts and plans only through actions whose predicted contract violations are acceptable.
- `oracle_target_mpc`: upper bound with access to target physical parameters.

## Required Experiments
- Main benchmark: at least 5 seeds, 10-12 episodes per seed, 7 target splits, and all main methods.
- Contract-prediction analysis:
  - accepted-transfer success
  - false-accept rate
  - false-reject rate
  - violation detection AUROC or threshold proxy
  - correlation between contract violation score and final error
- Ablations:
  - no contact-onset contract
  - no slip contract
  - no force-slope contract
  - no energy contract
  - no actuator-lag contract
  - scalar residual only
- Stress sweep over friction gap, actuator lag, and compliance shift.
- Negative cases: unmodeled object geometry, adversarial probe compliance, and semantic target mismatch.

## Submission-Readiness Gate
To be ICLR-main ready, the proposed method must:
- beat every non-oracle baseline on combined reality gap and at least four target splits
- reduce false accepts versus domain randomization, ensemble uncertainty, and conformal filtering
- show that each contract family matters through ablations
- avoid gaining safety by simply rejecting too many transfers or using excessive energy
- provide honest limitations and hostile prior-work framing

## Terminal Decision Rules
- `SUBMISSION_READY_CANDIDATE`: only if the contract mechanism clears all empirical gates and the paper can support a strong empirical claim.
- `STRONG_REVISE`: if contracts help but lack hardware/public benchmark breadth or manual related-work depth.
- `KILL_ARCHIVE`: if domain randomization, system identification, conformal filtering, residual adaptation, or scalar-residual ablations match the proposed contracts.

## Resource Discipline
Keep RAM light with compact MuJoCo models, small candidate action sets, compact CSVs, and at most four workers. Do not reduce rigor: retain seeds, target splits, baselines, ablations, stress tests, uncertainty, and terminal-failure analysis.

## Deliverables
- Rewritten `src/run_experiment.py` with real MuJoCo source-target transfer rollouts.
- Updated requirements, README, child status, claims, gate, readiness, audit, and terminal evidence docs.
- CSV results, pairwise comparisons, contract diagnostics, figures, stress sweep, and negative cases.
- Rewritten paper and compiled `C:/Users/wangz/Downloads/67.pdf` only.
- Public GitHub repo pushed with final commit.
- Root status/report files updated before Paper 68 starts.
