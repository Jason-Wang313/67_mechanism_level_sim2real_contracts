# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: v4 adds a real MuJoCo source-to-target transfer benchmark, but the evidence is negative. The mechanism contract planner is matched or beaten by residual adaptation, conformal filtering, and scalar residual ablations. Its false-accept reduction is achieved by rejecting almost all transfers rather than by enabling robust transfer.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: invent and test a substantially different action-conditioned contract mechanism that improves transfer success and false-accept rate without trivial rejection, validated on hardware or public high-fidelity sim-to-real benchmarks.
