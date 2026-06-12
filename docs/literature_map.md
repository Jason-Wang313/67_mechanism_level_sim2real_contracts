        # Literature Map

        Paper: 67 mechanism_level_sim2real_contracts

        Field box: sim-to-real transfer for robot manipulation

        Thesis: Mechanism-Level Sim2Real Contracts turns the seed bet into a mechanism: Transfer by preserving causal mechanisms rather than matching observation statistics.

        ## Landscape Sweep Summary
        The selector ranked records from the shared 500,000-record pool. The closest visible clusters are:
        - In-Hand Manipulation of Articulated Tools with Dexterous Robot Hands with Sim-to-Real Transfer (2025)
- Robust Visual Sim-to-Real Transfer for Robotic Manipulation (2023)
- Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors (2019)
- Bi-Touch: Bimanual Tactile Manipulation with Sim-to-Real Deep Reinforcement Learning (2023)
- Sim-to-Real Causal Transfer: A Metric Learning Approach to Causally-Aware Interaction Representations (2025)
- Improving Sim-to-Real Transfer in Vision-Based Robot Navigation via Instance-Level GAN-Based Data Augmentation (2025)
- Sim-to-Real Transfer for Visual Reinforcement Learning of Deformable Object Manipulation for Robot-Assisted Surgery (2023)
- DexSim2Real: Foundation Model-Guided Sim-to-Real Transfer for Generalizable Dexterous Manipulation (2026)
- Task-Level Control and Poincaré Map-Based Sim-to-Real Transfer for Effective Command Following of Quadrupedal Trot Gait (2023)
- Zero-shot sim-to-real transfer of reinforcement learning framework for robotics manipulation with demonstration and force feedback (2023)
- On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer (2024)
- A Survey on Sim-to-Real Transfer Methods for Robotic Manipulation (2024)

        ## Hidden Assumptions
        - The executed trajectory is a sufficient training target.
- Unobserved physical alternatives can be averaged into uncertainty.
- Task labels capture the mechanism that caused failure.
- A planner only needs nominal feasibility.
- Embodiment-specific contact effects are nuisance variation.

        ## Boundary
        The project avoids weak moves such as bigger models, generic uncertainty, or a benchmark-only contribution. It centers a mechanism-level change: Mechanism level sim2real contracts keeps action-critical alternatives explicit until a physical observation collapses them.
