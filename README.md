# 67 Mechanism-Level Sim2Real Contracts

Submission-hardening version: v4 real MuJoCo rebuild

Terminal decision: KILL_ARCHIVE for ICLR main conference.

The repository is retained as an archive of a falsified sim-to-real mechanism. The v4 rebuild replaces the synthetic probability scaffold with a MuJoCo source-to-target contact-manipulation benchmark containing friction, mass/compliance, actuator-lag, sensor-bias, and combined reality-gap shifts.

The proposed mechanism-level contract planner does not survive the ICLR-main gate. On combined reality gap it reaches 0.500 success, while residual adaptation reaches 0.683 and conformal filtering reaches 0.567. It reduces false accepts only by accepting almost no transfers (0.017 accept rate), and scalar residual ablations beat the full contract method.

## Reproduce Real Evidence

```powershell
python src\run_experiment.py
```

The run writes raw MuJoCo transfer rollouts, contract diagnostics, seed metrics, pairwise comparisons, ablations, stress sweeps, negative cases, and figures into `results/` and `figures/`.

## Rebuild Archive PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/67.pdf`
