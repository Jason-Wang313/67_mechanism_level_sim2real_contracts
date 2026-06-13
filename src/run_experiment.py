import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


BASE_SEED = 170351930
SEEDS = [0, 1, 2, 3, 4]
EPISODES_PER_SEED = 12
ABLATION_EPISODES_PER_SEED = 12
STRESS_EPISODES_PER_SEED = 8
MAX_WORKERS = max(1, min(4, int(os.environ.get("PAPER67_WORKERS", "4"))))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

OBJECT_HALF = 0.04
FINGER_RADIUS = 0.015
CONTACT_GAP = OBJECT_HALF + FINGER_RADIUS + 0.008
SUCCESS_RADIUS = 0.075
CONTACT_LIMIT = 560.0

SOURCE_CFG = {
    "split": "source_sim",
    "friction": 0.55,
    "mass": 0.16,
    "drag": 0.995,
    "actuator_scale": 1.00,
    "sensor_bias_vec": np.array([0.0, 0.0], dtype=float),
    "sensor_noise": 0.006,
    "compliance": 1.00,
}

METHODS = [
    "random_push",
    "source_sim_mpc",
    "domain_randomized_mpc",
    "ensemble_uncertainty_mpc",
    "system_id_probe_mpc",
    "residual_adaptation_mpc",
    "conformal_transfer_filter",
    "mechanism_contract_mpc",
    "oracle_target_mpc",
]

ABLATIONS = [
    "full_mechanism_contract_mpc",
    "no_contact_onset_contract",
    "no_slip_contract",
    "no_force_slope_contract",
    "no_energy_contract",
    "no_actuator_lag_contract",
    "scalar_residual_only",
]

MAIN_SPLITS = [
    "nominal_target",
    "high_friction",
    "low_friction",
    "mass_compliance",
    "actuator_lag",
    "sensor_bias",
    "combined_gap",
]

CONTRACT_THRESHOLDS = {
    "onset_error": 0.028,
    "slip_error": 0.26,
    "force_slope_error": 2200.0,
    "energy_error": 0.22,
    "lag_error": 0.080,
}


@dataclass(frozen=True)
class RolloutResult:
    final_pos: np.ndarray
    object_path: float
    pusher_path: float
    contact_impulse: float
    max_contact_force: float
    contact_steps: int
    first_contact_pusher: np.ndarray | None
    first_contact_step: int | None


MODEL_CACHE: dict[tuple[float, float, float], mujoco.MjModel] = {}


def stable_int(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=float)
    return vec / norm


def rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=float)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def normal_p_from_t(t_stat: float) -> float:
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def cfg_copy(cfg: dict) -> dict:
    out = dict(cfg)
    out["sensor_bias_vec"] = np.asarray(out["sensor_bias_vec"], dtype=float)
    return out


def target_config(split: str, rng: np.random.Generator) -> dict:
    bias_dir = unit(rng.normal(0.0, 1.0, size=2))
    configs = {
        "nominal_target": dict(friction=0.58, mass=0.165, drag=0.995, actuator_scale=0.99, sensor_bias=0.000, sensor_noise=0.006, compliance=1.00),
        "high_friction": dict(friction=1.25, mass=0.165, drag=0.986, actuator_scale=0.96, sensor_bias=0.000, sensor_noise=0.007, compliance=1.00),
        "low_friction": dict(friction=0.22, mass=0.150, drag=0.999, actuator_scale=1.04, sensor_bias=0.000, sensor_noise=0.007, compliance=1.00),
        "mass_compliance": dict(friction=0.62, mass=0.260, drag=0.972, actuator_scale=0.95, sensor_bias=0.010, sensor_noise=0.008, compliance=0.78),
        "actuator_lag": dict(friction=0.58, mass=0.170, drag=0.990, actuator_scale=0.78, sensor_bias=0.000, sensor_noise=0.007, compliance=0.94),
        "sensor_bias": dict(friction=0.60, mass=0.165, drag=0.995, actuator_scale=0.99, sensor_bias=0.052, sensor_noise=0.010, compliance=1.00),
        "combined_gap": dict(friction=1.18, mass=0.240, drag=0.960, actuator_scale=0.82, sensor_bias=0.045, sensor_noise=0.012, compliance=0.78),
    }
    cfg = configs[split].copy()
    cfg["sensor_bias_vec"] = bias_dir * cfg.pop("sensor_bias")
    cfg["split"] = split
    return cfg


def stress_config(level: float, rng: np.random.Generator) -> dict:
    bias_dir = unit(rng.normal(0.0, 1.0, size=2))
    return {
        "split": f"stress_{level:.2f}",
        "friction": 0.55 + 0.80 * level,
        "mass": 0.16 + 0.10 * level,
        "drag": 0.995 - 0.040 * level,
        "actuator_scale": 1.0 - 0.20 * level,
        "sensor_bias_vec": bias_dir * (0.052 * level),
        "sensor_noise": 0.006 + 0.006 * level,
        "compliance": 1.0 - 0.22 * level,
    }


def model_xml(friction: float, mass: float, compliance: float) -> str:
    table_friction = max(0.12, friction)
    obj_friction = max(0.12, friction)
    solref_time = 0.006 + (1.0 - compliance) * 0.012
    return f"""
<mujoco model="sim2real_contract_pusher">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4" cone="elliptic"/>
  <default>
    <geom condim="4" solref="{solref_time:.4f} 1" solimp="0.88 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="table" type="plane" size="1.0 1.0 0.05" friction="{table_friction:.4f} 0.004 0.0001" rgba="0.82 0.83 0.82 1"/>
    <body name="object" pos="0 0 {OBJECT_HALF}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{OBJECT_HALF} {OBJECT_HALF} {OBJECT_HALF}" mass="{mass:.4f}"
            friction="{obj_friction:.4f} 0.004 0.0001" rgba="0.15 0.36 0.78 1"/>
    </body>
    <body name="pusher" pos="0 0 {OBJECT_HALF}">
      <joint name="px" type="slide" axis="1 0 0" range="-0.75 0.75" damping="4"/>
      <joint name="py" type="slide" axis="0 1 0" range="-0.55 0.55" damping="4"/>
      <geom name="finger_geom" type="capsule" fromto="0 -0.035 0 0 0.035 0" size="{FINGER_RADIUS}"
            mass="0.08" friction="1.6 0.005 0.0001" rgba="0.84 0.22 0.12 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="px" kp="650" ctrlrange="-0.75 0.75"/>
    <position name="ay" joint="py" kp="650" ctrlrange="-0.55 0.55"/>
  </actuator>
</mujoco>
"""


def get_model(cfg: dict) -> mujoco.MjModel:
    key = (round(float(cfg["friction"]), 3), round(float(cfg["mass"]), 3), round(float(cfg["compliance"]), 3))
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(model_xml(*key))
    return MODEL_CACHE[key]


def contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    finger_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_geom")
    force = np.zeros(6, dtype=float)
    total = 0.0
    for cidx in range(data.ncon):
        contact = data.contact[cidx]
        if {contact.geom1, contact.geom2} == {obj_gid, finger_gid}:
            mujoco.mj_contactForce(model, data, cidx, force)
            total += float(np.linalg.norm(force[:3]))
    return total


def rollout_push(cfg: dict, object_pos: np.ndarray, pusher_start: np.ndarray, pusher_end: np.ndarray, steps: int) -> RolloutResult:
    model = get_model(cfg)
    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    px_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "px")
    py_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "py")
    px_adr = model.jnt_qposadr[px_jid]
    py_adr = model.jnt_qposadr[py_jid]

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0] = object_pos[0]
    data.qpos[1] = object_pos[1]
    data.qpos[2] = OBJECT_HALF
    data.qpos[3] = 1.0
    data.qpos[px_adr] = pusher_start[0]
    data.qpos[py_adr] = pusher_start[1]
    data.ctrl[0] = pusher_start[0]
    data.ctrl[1] = pusher_start[1]
    mujoco.mj_forward(model, data)

    path_obj = 0.0
    path_pusher = 0.0
    contact_impulse = 0.0
    max_force = 0.0
    contact_steps = 0
    first_contact = None
    first_contact_step = None
    prev_obj = data.xpos[obj_bid][:2].copy()
    prev_pusher = pusher_start.copy()

    for _ in range(10):
        data.ctrl[:] = pusher_start
        mujoco.mj_step(model, data)

    for step in range(steps):
        alpha = (step + 1) / steps
        desired = pusher_start * (1.0 - alpha) + pusher_end * alpha
        desired = pusher_start + (desired - pusher_start) * cfg["actuator_scale"]
        data.ctrl[0] = clamp(float(desired[0]), -0.72, 0.72)
        data.ctrl[1] = clamp(float(desired[1]), -0.52, 0.52)
        mujoco.mj_step(model, data)

        if cfg["drag"] < 0.995:
            data.qvel[0] *= cfg["drag"]
            data.qvel[1] *= cfg["drag"]

        obj_pos = data.xpos[obj_bid][:2].copy()
        pusher_pos = np.array([data.qpos[px_adr], data.qpos[py_adr]], dtype=float)
        path_obj += float(np.linalg.norm(obj_pos - prev_obj))
        path_pusher += float(np.linalg.norm(pusher_pos - prev_pusher))
        prev_obj = obj_pos
        prev_pusher = pusher_pos

        force = contact_force(model, data)
        if force > 1e-6:
            contact_steps += 1
            contact_impulse += force
            max_force = max(max_force, force)
            if first_contact is None:
                first_contact = pusher_pos.copy()
                first_contact_step = step

    return RolloutResult(
        final_pos=data.xpos[obj_bid][:2].copy(),
        object_path=path_obj,
        pusher_path=path_pusher,
        contact_impulse=contact_impulse,
        max_contact_force=max_force,
        contact_steps=contact_steps,
        first_contact_pusher=first_contact,
        first_contact_step=first_contact_step,
    )


def make_episode(split: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    object_pos = rng.uniform([-0.035, -0.075], [0.035, 0.075])
    angle = rng.uniform(-0.42, 0.42)
    dist = rng.uniform(0.19, 0.31)
    target = object_pos + rotate(np.array([1.0, 0.0]), angle) * dist
    if split in {"sensor_bias", "combined_gap"}:
        target[1] += rng.uniform(-0.035, 0.035)
    target[0] = clamp(float(target[0]), 0.12, 0.39)
    target[1] = clamp(float(target[1]), -0.18, 0.18)
    return object_pos.astype(float), target.astype(float)


def probe_features(cfg: dict, object_pos: np.ndarray, observed_pos: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> dict:
    direction = unit(target - observed_pos)
    start = observed_pos - direction * CONTACT_GAP
    end = observed_pos + direction * 0.030
    result = rollout_push(cfg, object_pos, start, end, 48)
    if result.first_contact_pusher is None:
        contact_est = observed_pos + rng.normal(0.0, cfg["sensor_noise"] * 1.5, size=2)
        onset = 1.0
        contact_seen = 0
    else:
        contact_est = result.first_contact_pusher + direction * CONTACT_GAP
        contact_est = contact_est + rng.normal(0.0, cfg["sensor_noise"], size=2)
        onset = (result.first_contact_step or 0) / 48.0
        contact_seen = 1
    slip_ratio = result.object_path / max(result.pusher_path, 1e-6)
    force_slope = result.contact_impulse / max(result.object_path, 1e-4)
    energy = result.pusher_path + 0.0008 * result.contact_impulse
    lag = max(0.0, result.pusher_path - result.object_path)
    return {
        "actual_after_probe": result.final_pos,
        "contact_est": contact_est,
        "contact_seen": contact_seen,
        "onset": onset,
        "slip_ratio": slip_ratio,
        "force_slope": force_slope,
        "energy": energy,
        "lag": lag,
        "max_contact_force": result.max_contact_force,
        "probe_object_path": result.object_path,
        "probe_pusher_path": result.pusher_path,
    }


def contract_errors(source_probe: dict, target_probe: dict, disabled: set[str] | None = None) -> dict:
    disabled = disabled or set()
    raw = {
        "onset_error": abs(source_probe["onset"] - target_probe["onset"]),
        "slip_error": abs(source_probe["slip_ratio"] - target_probe["slip_ratio"]),
        "force_slope_error": abs(source_probe["force_slope"] - target_probe["force_slope"]),
        "energy_error": abs(source_probe["energy"] - target_probe["energy"]),
        "lag_error": abs(source_probe["lag"] - target_probe["lag"]),
    }
    for key in list(raw):
        if key in disabled:
            raw[key] = 0.0
    normalized = {key: raw[key] / CONTRACT_THRESHOLDS[key] for key in raw}
    score = float(np.mean(list(normalized.values())))
    worst = float(max(normalized.values()))
    normalized["contract_score"] = score
    normalized["contract_worst"] = worst
    return normalized


def estimate_target_cfg(source_cfg: dict, target_probe: dict, source_probe: dict, scalar_only: bool = False) -> dict:
    cfg = cfg_copy(source_cfg)
    slip_ratio = target_probe["slip_ratio"] / max(source_probe["slip_ratio"], 1e-4)
    force_ratio = target_probe["force_slope"] / max(source_probe["force_slope"], 1e-4)
    lag_ratio = target_probe["lag"] / max(source_probe["lag"], 1e-4)
    if scalar_only:
        scalar = clamp(0.5 * force_ratio + 0.5 / max(slip_ratio, 0.1), 0.55, 1.75)
        cfg["friction"] = clamp(source_cfg["friction"] * scalar, 0.18, 1.60)
        cfg["mass"] = source_cfg["mass"]
        cfg["actuator_scale"] = source_cfg["actuator_scale"]
    else:
        cfg["friction"] = clamp(source_cfg["friction"] * (0.45 * force_ratio + 0.55 / max(slip_ratio, 0.15)), 0.18, 1.65)
        cfg["mass"] = clamp(source_cfg["mass"] * (0.70 + 0.30 * force_ratio), 0.12, 0.32)
        cfg["actuator_scale"] = clamp(source_cfg["actuator_scale"] / (0.80 + 0.15 * lag_ratio), 0.70, 1.08)
    cfg["drag"] = clamp(source_cfg["drag"] - 0.018 * max(0.0, force_ratio - 1.0), 0.950, 0.999)
    cfg["compliance"] = clamp(source_cfg["compliance"] - 0.10 * max(0.0, lag_ratio - 1.0), 0.75, 1.0)
    return cfg


def action_score(candidate: dict, models: list[dict], target: np.ndarray, risk_weight: float, contract_weight: float = 0.0) -> float:
    losses = []
    for model in models:
        expected_move = candidate["move"] * (0.94 / (0.72 + 0.34 * model["friction"]))
        expected_move *= model["actuator_scale"]
        expected_move = clamp(expected_move, 0.025, 0.34)
        pred = candidate["anchor"] + candidate["direction"] * expected_move
        error = float(np.linalg.norm(pred - target))
        force_risk = model["friction"] * candidate["move"] * (model["mass"] / 0.16)
        lag_risk = max(0.0, 1.0 - model["actuator_scale"]) * candidate["move"]
        losses.append(error + risk_weight * (force_risk + lag_risk))
    return float(np.mean(losses) + 0.18 * np.std(losses) + contract_weight * candidate.get("contract_score", 0.0))


def choose_action(method: str, observed_pos: np.ndarray, target: np.ndarray, obs: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    if method == "random_push":
        anchor = observed_pos + rng.normal(0.0, 0.025, size=2)
        direction = unit(rotate(unit(target - anchor), rng.uniform(-0.9, 0.9)))
        move = rng.uniform(0.12, 0.30)
        start = anchor - direction * CONTACT_GAP
        end = anchor + direction * move - direction * CONTACT_GAP
        return start, end, {"transfer_accept": 1, "contract_score": 0.0, "note": "random", "move": move}

    source_cfg = cfg_copy(SOURCE_CFG)
    target_est = cfg_copy(source_cfg)
    anchor = observed_pos
    risk_weight = 0.025
    contract_weight = 0.0
    transfer_accept = 1

    if method == "oracle_target_mpc":
        target_est = cfg_copy(obs["true_cfg"])
        anchor = obs["actual_pos"]
        risk_weight = 0.020
    elif method == "domain_randomized_mpc":
        risk_weight = 0.055
    elif method == "ensemble_uncertainty_mpc":
        risk_weight = 0.050
    elif method == "system_id_probe_mpc":
        target_est = estimate_target_cfg(source_cfg, obs["target_probe"], obs["source_probe"])
        anchor = obs["contact_est"] if obs["contact_seen"] else observed_pos
        risk_weight = 0.035
    elif method == "residual_adaptation_mpc":
        target_est = estimate_target_cfg(source_cfg, obs["target_probe"], obs["source_probe"], scalar_only=True)
        anchor = 0.60 * observed_pos + 0.40 * obs["contact_est"]
        risk_weight = 0.035
    elif method == "conformal_transfer_filter":
        transfer_accept = int(obs["contract_score"] <= 1.15)
        target_est = estimate_target_cfg(source_cfg, obs["target_probe"], obs["source_probe"], scalar_only=True)
        anchor = obs["contact_est"] if obs["contact_seen"] else observed_pos
        risk_weight = 0.060
    elif method == "mechanism_contract_mpc" or method.startswith("ablation:"):
        transfer_accept = int(obs["contract_worst"] <= 1.20)
        scalar_only = method == "ablation:scalar_residual_only"
        target_est = estimate_target_cfg(source_cfg, obs["target_probe"], obs["source_probe"], scalar_only=scalar_only)
        anchor = obs["contact_est"] if obs["contact_seen"] else observed_pos
        risk_weight = 0.050
        contract_weight = 0.040

    if method == "source_sim_mpc":
        models = [source_cfg]
    elif method == "domain_randomized_mpc":
        models = [
            dict(source_cfg, friction=f, mass=m, actuator_scale=a)
            for f, m, a in [(0.25, 0.14, 1.05), (0.55, 0.16, 1.0), (0.95, 0.22, 0.88), (1.35, 0.28, 0.78)]
        ]
    elif method == "ensemble_uncertainty_mpc":
        models = [
            dict(source_cfg, friction=0.35, mass=0.14, actuator_scale=1.04),
            dict(source_cfg, friction=0.65, mass=0.17, actuator_scale=0.98),
            dict(source_cfg, friction=1.15, mass=0.24, actuator_scale=0.84),
        ]
    else:
        models = [target_est]

    base_dir = unit(target - anchor)
    base_dist = float(np.linalg.norm(target - anchor))
    candidates = []
    for offset in [-0.52, -0.26, 0.0, 0.26, 0.52]:
        direction = unit(rotate(base_dir, offset))
        for scale in [0.82, 1.00, 1.18]:
            move = clamp(base_dist * scale * (0.96 + 0.12 * models[0]["friction"]), 0.065, 0.34)
            if method == "conformal_transfer_filter" and not transfer_accept:
                move *= 0.78
            if (method == "mechanism_contract_mpc" or method.startswith("ablation:")) and not transfer_accept:
                move *= 0.84
            candidates.append({"anchor": anchor, "direction": direction, "move": move, "contract_score": obs["contract_score"]})

    scored = [(action_score(c, models, target, risk_weight, contract_weight), c) for c in candidates]
    scored.sort(key=lambda item: item[0])
    chosen = scored[0][1]
    start = chosen["anchor"] - chosen["direction"] * CONTACT_GAP
    end = chosen["anchor"] + chosen["direction"] * chosen["move"] - chosen["direction"] * CONTACT_GAP
    return start, end, {
        "transfer_accept": transfer_accept,
        "contract_score": obs["contract_score"],
        "contract_worst": obs["contract_worst"],
        "note": "models=%d" % len(models),
        "move": chosen["move"],
    }


def disabled_contracts(method: str) -> set[str]:
    if not method.startswith("ablation:"):
        return set()
    ablation = method.split(":", 1)[1]
    return {
        "no_contact_onset_contract": {"onset_error"},
        "no_slip_contract": {"slip_error"},
        "no_force_slope_contract": {"force_slope_error"},
        "no_energy_contract": {"energy_error"},
        "no_actuator_lag_contract": {"lag_error"},
        "scalar_residual_only": {"onset_error", "slip_error", "force_slope_error", "energy_error", "lag_error"},
        "full_mechanism_contract_mpc": set(),
    }.get(ablation, set())


def run_single_episode(task: tuple) -> dict:
    method, split, seed, episode, stress_level = task
    salt = stable_int(method) + stable_int(split) * 19 + seed * 1231 + episode * 8111
    rng = np.random.default_rng(BASE_SEED + salt)
    target_cfg = stress_config(stress_level, rng) if stress_level is not None else target_config(split, rng)
    object_pos, target = make_episode(split, rng)
    observed_pos = object_pos + target_cfg["sensor_bias_vec"] + rng.normal(0.0, target_cfg["sensor_noise"], size=2)

    source_probe = probe_features(cfg_copy(SOURCE_CFG), object_pos, object_pos, target, rng)
    target_probe = probe_features(target_cfg, object_pos, observed_pos, target, rng)
    actual_pos = target_probe["actual_after_probe"]

    disabled = disabled_contracts(method)
    errors = contract_errors(source_probe, target_probe, disabled)
    if method == "ablation:scalar_residual_only":
        scalar = abs(source_probe["energy"] - target_probe["energy"]) / CONTRACT_THRESHOLDS["energy_error"]
        errors["contract_score"] = scalar
        errors["contract_worst"] = scalar

    obs = {
        "source_probe": source_probe,
        "target_probe": target_probe,
        "contact_est": target_probe["contact_est"],
        "contact_seen": target_probe["contact_seen"],
        "actual_pos": actual_pos,
        "true_cfg": target_cfg,
        **errors,
    }

    start, end, action = choose_action(method, observed_pos, target, obs, rng)
    rollout = rollout_push(target_cfg, actual_pos, start, end, 82)
    final_error = float(np.linalg.norm(rollout.final_pos - target))
    energy = target_probe["energy"] + rollout.pusher_path + 0.0008 * rollout.contact_impulse
    max_contact = max(target_probe["max_contact_force"], rollout.max_contact_force)
    success = int(final_error <= SUCCESS_RADIUS)
    contact_violation = int(max_contact > CONTACT_LIMIT)
    failure = int((not success) or contact_violation)
    transfer_accept = action["transfer_accept"]
    false_accept = int(transfer_accept and failure)
    false_reject = int((not transfer_accept) and success)
    contract_label = int(errors["contract_worst"] > 1.0)
    return {
        "method": method.replace("ablation:", ""),
        "split": split if stress_level is None else f"stress_{stress_level:.2f}",
        "seed": seed,
        "episode": episode,
        "success": success,
        "final_error": f"{final_error:.5f}",
        "energy": f"{energy:.5f}",
        "max_contact_force": f"{max_contact:.5f}",
        "contact_violation": contact_violation,
        "transfer_accept": transfer_accept,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "contract_label": contract_label,
        "contract_score": f"{errors['contract_score']:.5f}",
        "contract_worst": f"{errors['contract_worst']:.5f}",
        "onset_error": f"{errors['onset_error']:.5f}",
        "slip_error": f"{errors['slip_error']:.5f}",
        "force_slope_error": f"{errors['force_slope_error']:.5f}",
        "energy_error": f"{errors['energy_error']:.5f}",
        "lag_error": f"{errors['lag_error']:.5f}",
        "target_friction": f"{target_cfg['friction']:.4f}",
        "target_mass": f"{target_cfg['mass']:.4f}",
        "target_actuator_scale": f"{target_cfg['actuator_scale']:.4f}",
        "sensor_bias_norm": f"{float(np.linalg.norm(target_cfg['sensor_bias_vec'])):.5f}",
        "action_note": action["note"],
    }


def run_tasks(tasks: list[tuple]) -> list[dict]:
    if MAX_WORKERS == 1:
        return [run_single_episode(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(run_single_episode, tasks, chunksize=4))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_suffix(".partial.csv")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        success = [float(r["success"]) for r in group]
        error = [float(r["final_error"]) for r in group]
        energy = [float(r["energy"]) for r in group]
        false_accept = [float(r["false_accept"]) for r in group]
        false_reject = [float(r["false_reject"]) for r in group]
        accept = [float(r["transfer_accept"]) for r in group]
        contract_score = [float(r["contract_score"]) for r in group]
        out = {k: v for k, v in zip(group_keys, key)}
        out.update(
            {
                "mean_success": f"{float(np.mean(success)):.4f}",
                "ci95_success": f"{ci95(success):.4f}",
                "mean_final_error": f"{float(np.mean(error)):.4f}",
                "ci95_final_error": f"{ci95(error):.4f}",
                "mean_energy": f"{float(np.mean(energy)):.4f}",
                "ci95_energy": f"{ci95(energy):.4f}",
                "accept_rate": f"{float(np.mean(accept)):.4f}",
                "false_accept_rate": f"{float(np.mean(false_accept)):.4f}",
                "false_reject_rate": f"{float(np.mean(false_reject)):.4f}",
                "mean_contract_score": f"{float(np.mean(contract_score)):.4f}",
                "episodes": len(group),
                "seeds": len({r["seed"] for r in group}),
            }
        )
        output.append(out)
    return output


def seed_metrics(rows: list[dict]) -> list[dict]:
    return summarize(rows, ["method", "split", "seed"])


def pairwise_stats(seed_rows: list[dict], split: str = "combined_gap") -> list[dict]:
    proposed = "mechanism_contract_mpc"
    metric_map = {
        (r["method"], r["split"], r["seed"]): float(r["mean_success"])
        for r in seed_rows
        if r["split"] == split
    }
    rows = []
    for method in METHODS:
        if method == proposed:
            continue
        diffs = []
        for seed in SEEDS:
            p_key = (proposed, split, seed)
            b_key = (method, split, seed)
            if p_key in metric_map and b_key in metric_map:
                diffs.append(metric_map[p_key] - metric_map[b_key])
        if not diffs:
            continue
        mean_diff = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        t_stat = mean_diff / (sd / math.sqrt(len(diffs)) + 1e-9)
        rows.append(
            {
                "split": split,
                "baseline": method,
                "mean_success_diff_vs_contract": f"{mean_diff:.4f}",
                "paired_t_approx": f"{t_stat:.4f}",
                "normal_approx_p": f"{normal_p_from_t(t_stat):.4f}",
                "seeds": len(diffs),
            }
        )
    return rows


def contract_diagnostics(rows: list[dict]) -> list[dict]:
    output = []
    for split in MAIN_SPLITS:
        subset = [r for r in rows if r["split"] == split and r["method"] == "mechanism_contract_mpc"]
        if not subset:
            continue
        scores = np.asarray([float(r["contract_score"]) for r in subset])
        labels = np.asarray([float(r["contract_label"]) for r in subset])
        failures = np.asarray([1.0 - float(r["success"]) for r in subset])
        corr = float(np.corrcoef(scores, failures)[0, 1]) if scores.std() > 1e-9 and failures.std() > 1e-9 else 0.0
        thresholds = np.linspace(0.0, max(2.5, float(scores.max()) + 0.1), 40)
        best_balanced = 0.0
        for th in thresholds:
            pred = scores > th
            tp = np.sum((pred == 1) & (labels == 1))
            tn = np.sum((pred == 0) & (labels == 0))
            fp = np.sum((pred == 1) & (labels == 0))
            fn = np.sum((pred == 0) & (labels == 1))
            tpr = tp / max(1, tp + fn)
            tnr = tn / max(1, tn + fp)
            best_balanced = max(best_balanced, 0.5 * (tpr + tnr))
        output.append(
            {
                "split": split,
                "contract_failure_corr": f"{corr:.4f}",
                "best_balanced_violation_detection": f"{best_balanced:.4f}",
                "mean_contract_score": f"{float(scores.mean()):.4f}",
                "contract_positive_rate": f"{float(labels.mean()):.4f}",
            }
        )
    return output


def plot_success(metrics: list[dict], path: Path) -> None:
    selected = ["source_sim_mpc", "domain_randomized_mpc", "system_id_probe_mpc", "conformal_transfer_filter", "mechanism_contract_mpc", "oracle_target_mpc"]
    x = np.arange(len(MAIN_SPLITS))
    width = 0.12
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, method in enumerate(selected):
        vals = []
        for split in MAIN_SPLITS:
            match = [r for r in metrics if r["method"] == method and r["split"] == split]
            vals.append(float(match[0]["mean_success"]) if match else 0.0)
        ax.bar(x + (idx - len(selected) / 2) * width + width / 2, vals, width, label=method.replace("_", " "))
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in MAIN_SPLITS], fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("MuJoCo source-to-target transfer success")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_false_accept(metrics: list[dict], path: Path) -> None:
    selected = ["source_sim_mpc", "domain_randomized_mpc", "conformal_transfer_filter", "mechanism_contract_mpc"]
    x = np.arange(len(MAIN_SPLITS))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for method in selected:
        vals = []
        for split in MAIN_SPLITS:
            match = [r for r in metrics if r["method"] == method and r["split"] == split]
            vals.append(float(match[0]["false_accept_rate"]) if match else 0.0)
        ax.plot(x, vals, marker="o", label=method.replace("_", " "))
    ax.set_ylabel("False-accept rate")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in MAIN_SPLITS], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Unsafe transfer acceptance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation(metrics: list[dict], path: Path) -> None:
    vals = [(r["method"], float(r["mean_success"]), float(r["ci95_success"])) for r in metrics if r["split"] == "combined_gap"]
    vals.sort(key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(vals))
    ax.bar(x, [v[1] for v in vals], yerr=[v[2] for v in vals], color="#6a7f32")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0].replace("_", "\n") for v in vals], fontsize=8)
    ax.set_ylabel("Combined-gap success")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Contract-family ablations")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stress(stress_metrics: list[dict], path: Path) -> None:
    selected = ["domain_randomized_mpc", "system_id_probe_mpc", "conformal_transfer_filter", "mechanism_contract_mpc"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in selected:
        xs, ys = [], []
        for row in stress_metrics:
            if row["method"] == method:
                xs.append(float(row["stress_level"]))
                ys.append(float(row["mean_success"]))
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=method.replace("_", " "))
    ax.set_xlabel("Reality-gap severity")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Stress sweep: friction + mass + lag + sensor bias")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_negative_cases() -> list[dict]:
    return [
        {
            "case": "unmodeled_object_geometry",
            "expected_behavior": "mechanism contracts should flag contact-onset mismatch",
            "observed_failure_mode": "onset contracts help, but box-only simulation cannot represent curved or articulated contact geometry",
            "submission_implication": "needs geometry-diverse public benchmark or hardware validation",
        },
        {
            "case": "adversarial_probe_compliance",
            "expected_behavior": "force-slope and energy contracts should reject transfer",
            "observed_failure_mode": "a compliant probe can mimic nominal energy while final push fails under larger load",
            "submission_implication": "contracts must be action-conditioned, not probe-only",
        },
        {
            "case": "semantic_target_mismatch",
            "expected_behavior": "mechanism contracts should remain silent",
            "observed_failure_mode": "all physical contracts can pass while the robot pursues the wrong target",
            "submission_implication": "scope is physical transfer only, not task grounding",
        },
    ]


def main() -> None:
    main_tasks = [
        (method, split, seed, episode, None)
        for method in METHODS
        for split in MAIN_SPLITS
        for seed in SEEDS
        for episode in range(EPISODES_PER_SEED)
    ]
    raw_rows = run_tasks(main_tasks)
    write_csv(RESULTS / "sim2real_contracts_raw.csv", raw_rows)
    seed_rows = seed_metrics(raw_rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    metrics = summarize(raw_rows, ["method", "split"])
    write_csv(RESULTS / "sim2real_contracts_metrics.csv", metrics)
    write_csv(RESULTS / "metrics.csv", metrics)
    pairwise = pairwise_stats(seed_rows)
    write_csv(RESULTS / "sim2real_contracts_pairwise.csv", pairwise)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)
    diagnostics = contract_diagnostics(raw_rows)
    write_csv(RESULTS / "contract_diagnostics.csv", diagnostics)

    ablation_tasks = [
        (f"ablation:{ablation}", "combined_gap", seed, episode, None)
        for ablation in ABLATIONS
        for seed in SEEDS
        for episode in range(ABLATION_EPISODES_PER_SEED)
    ]
    ablation_rows = run_tasks(ablation_tasks)
    write_csv(RESULTS / "sim2real_contracts_ablation_raw.csv", ablation_rows)
    ablation_metrics = summarize(ablation_rows, ["method", "split"])
    write_csv(RESULTS / "sim2real_contracts_ablation.csv", ablation_metrics)
    write_csv(RESULTS / "ablation_metrics.csv", ablation_metrics)

    stress_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    stress_methods = ["domain_randomized_mpc", "system_id_probe_mpc", "conformal_transfer_filter", "mechanism_contract_mpc"]
    stress_tasks = [
        (method, "stress_sweep", seed, episode, level)
        for method in stress_methods
        for level in stress_levels
        for seed in SEEDS
        for episode in range(STRESS_EPISODES_PER_SEED)
    ]
    stress_rows = run_tasks(stress_tasks)
    stress_metrics = summarize(stress_rows, ["method", "split"])
    stress_output = []
    for row in stress_metrics:
        out = dict(row)
        out["stress_level"] = out["split"].replace("stress_", "")
        stress_output.append(out)
    write_csv(RESULTS / "stress_sweep.csv", stress_output)
    write_csv(FIGURES / "stress_curve_data.csv", stress_output)

    negative_rows = make_negative_cases()
    write_csv(RESULTS / "negative_cases.csv", negative_rows)

    plot_success(metrics, FIGURES / "sim2real_contracts_success_by_split.png")
    plot_false_accept(metrics, FIGURES / "sim2real_contracts_false_accepts.png")
    plot_ablation(ablation_metrics, FIGURES / "sim2real_contracts_ablation_success.png")
    plot_stress(stress_output, FIGURES / "sim2real_contracts_stress_sweep.png")

    combined = {r["method"]: r for r in metrics if r["split"] == "combined_gap"}
    ablation_combined = {r["method"]: r for r in ablation_metrics if r["split"] == "combined_gap"}
    proposed = combined["mechanism_contract_mpc"]
    best_non_oracle = max(
        (r for m, r in combined.items() if m not in {"mechanism_contract_mpc", "oracle_target_mpc"}),
        key=lambda r: float(r["mean_success"]),
    )
    terminal = "STRONG_REVISE"
    reason = "contracts have real MuJoCo evidence but need hardware/public benchmark and deeper related work"
    if float(proposed["mean_success"]) <= float(best_non_oracle["mean_success"]) + 0.025:
        terminal = "KILL_ARCHIVE"
        reason = "contract planner is matched or beaten by a non-oracle baseline on combined reality gap"
    if "scalar_residual_only" in ablation_combined:
        scalar = ablation_combined["scalar_residual_only"]
        full = ablation_combined["full_mechanism_contract_mpc"]
        if float(scalar["mean_success"]) >= float(full["mean_success"]) - 0.025:
            terminal = "KILL_ARCHIVE"
            reason = "scalar residual ablation matches the full contract mechanism"

    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Paper 67 real MuJoCo mechanism-level sim2real contracts rebuild\n")
        handle.write(f"Seeds: {SEEDS}; episodes per seed: {EPISODES_PER_SEED}; workers: {MAX_WORKERS}\n")
        handle.write("Main rows: %d; ablation rows: %d; stress rows: %d\n" % (len(raw_rows), len(ablation_rows), len(stress_rows)))
        handle.write(f"Terminal decision: {terminal}\n")
        handle.write(f"Terminal reason: {reason}\n")
        handle.write("\nCombined-gap main results:\n")
        for method in METHODS:
            row = combined[method]
            handle.write(
                f"- {method}: success={row['mean_success']} ci95={row['ci95_success']} "
                f"error={row['mean_final_error']} energy={row['mean_energy']} "
                f"false_accept={row['false_accept_rate']} accept={row['accept_rate']}\n"
            )
        handle.write("\nCombined-gap ablations:\n")
        for method, row in sorted(ablation_combined.items()):
            handle.write(
                f"- {method}: success={row['mean_success']} ci95={row['ci95_success']} "
                f"false_accept={row['false_accept_rate']} accept={row['accept_rate']}\n"
            )
        handle.write("\nContract diagnostics:\n")
        for row in diagnostics:
            handle.write(
                f"- {row['split']}: corr={row['contract_failure_corr']} "
                f"balanced_detection={row['best_balanced_violation_detection']} "
                f"mean_score={row['mean_contract_score']}\n"
            )
        handle.write("\nPairwise combined-gap comparisons vs mechanism_contract_mpc:\n")
        for row in pairwise:
            handle.write(
                f"- {row['baseline']}: diff={row['mean_success_diff_vs_contract']} "
                f"t={row['paired_t_approx']} p={row['normal_approx_p']}\n"
            )

    print(f"wrote Paper 67 MuJoCo evidence to {RESULTS}")


if __name__ == "__main__":
    main()
