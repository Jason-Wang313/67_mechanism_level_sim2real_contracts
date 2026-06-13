        # Hostile Reviewer Response

        Paper: 67 Mechanism-Level Sim2Real Contracts

        ## Strongest Technical Threats
        - Robust Visual Sim-to-Real Transfer for Robotic Manipulation (2023)
- In-Hand Manipulation of Articulated Tools with Dexterous Robot Hands with Sim-to-Real Transfer (2025)
- Bi-Touch: Bimanual Tactile Manipulation with Sim-to-Real Deep Reinforcement Learning (2023)
- Zero-shot sim-to-real transfer of reinforcement learning framework for robotics manipulation with demonstration and force feedback (2023)
- SymBridge: Bilateral-Symmetry-Aware Vision-Language-Action Learning with Head-Mounted Display Teleoperation for Sim2Real Transfer in Bimanual Manipulation (n.d.)
- ExoGS: A 4D Real-to-Sim-to-Real Framework for Scalable Manipulation Data Collection (2026)
- Sim-to-Real Causal Transfer: A Metric Learning Approach to Causally-Aware Interaction Representations (2025)
- EMMA: Generalizing Real-World Robot Manipulation via Generative Visual Transfer (2025)

        ## ICLR Main Response
        A hostile ICLR reviewer would be correct to reject this as a main-conference submission. The v4 rebuild now contains a real MuJoCo source-to-target transfer benchmark, but explicit mechanism contracts lose to residual adaptation and scalar residual ablations on the combined reality gap.

        ## Honest Action
        The paper is marked `KILL_ARCHIVE`. This avoids converting a falsified mechanism into an overstated main-conference claim.

        ## What Would Be Needed To Revive
        - A substantially different action-conditioned contract mechanism that improves both transfer success and false-accept control.
        - Real robot or public high-fidelity benchmark experiments.
        - Manual full-paper related-work audit.
        - Evidence that the core mechanism is learned and useful under deployment shift.
