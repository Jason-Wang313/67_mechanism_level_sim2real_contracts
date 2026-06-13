# Final Audit

1. Chosen thesis: Mechanism-Level Sim2Real Contracts explores `Transfer by preserving causal mechanisms rather than matching observation statistics.` for sim-to-real transfer for robot manipulation.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4 real MuJoCo rebuild.
4. Reason: real MuJoCo source-target transfer evidence falsifies the mechanism. On combined reality gap, mechanism contracts reach 0.500 success, residual adaptation reaches 0.683, and scalar residual ablation reaches 0.650; the contract planner reduces false accepts by accepting almost no transfers.
5. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`.
6. Reproducibility: `python src\run_experiment.py` reproduces the MuJoCo benchmark, CSVs, contract diagnostics, figures, ablations, pairwise stats, stress sweep, and negative cases.
7. Claim-validity status: main-conference claims killed by direct empirical evidence; archive retained as a negative result.
8. Exact Downloads PDF path: `C:/Users/wangz/Downloads/67.pdf`
9. GitHub URL: https://github.com/Jason-Wang313/67_mechanism_level_sim2real_contracts
10. Confirmation: no visible Desktop copy was requested or made.
