# SyncSBC: Decentralized Swarm Behavior Prediction for Synchronized Autonomous Control

This folder contains the core scripts behind **SyncSBC**: a two-stage decentralized pipeline that lets each robot infer swarm behavior from local sensing and then synchronize decisions with neighbors.

SyncSBC targets communication-constrained swarm robotics with no centralized coordinator.

## What is in this folder

This is a lightweight code snapshot (not a full standalone ROS 2 package) with:

- Agent-level behavior prediction node (Stage 0 in code, classifier inference)
- Stage-1 internal belief formation methods
- Stage-2 synchronization methods
- Message definitions used by the pipeline
- Isaac Sim helper scripts and controller test scripts

## SyncSBC pipeline (paper -> code mapping)

### 1) Agent-level prediction (SBC + smoothing)

- Paper concept: each robot predicts global behavior from local observation history, then smooths probabilities over time.
- Code: `agent_classifier/predictor.py`
- Output topic per robot: `/prediction/<robot_name>`
- Message: `hero_common_only_msg/Predictions.msg` (`x, y, z`)

### 2) Stage-1: Internal belief formation

These map to the methods compared in the paper:

- Averaging: `stage1/averaging.py`
- Entropy-based fusion: `stage1/entropy.py`
- Neighbor variance update: `stage1/neighborv.py`
- Sample-and-hold baseline (Pure SH): `stage1/Puresh.py`

Each node consumes prediction streams and local TF-based neighbor geometry, then publishes per-robot binary belief consensus.

### 3) Stage-2: Synchronization

These correspond to the synchronized methods highlighted in the paper:

- **ESB** = Event-triggered + Sample-and-Hold + Boolean Gossip: `stage2/esb.py`
- **ESI** = Event-triggered + Sample-and-Hold + Integrated Belief Synchronizer: `stage2/esi.py`

Both are decentralized and neighborhood-limited (radius-based), using `/tf` for proximity and custom messages for state sharing.

## Repository structure

```text
SyncSBC/
├── agent_classifier/
│   └── predictor.py
├── stage1/
│   ├── averaging.py
│   ├── entropy.py
│   ├── neighborv.py
│   └── Puresh.py
├── stage2/
│   ├── esb.py
│   └── esi.py
├── hero_common_only_msg/
│   ├── Predictions.msg
│   ├── Boolconsensus.msg
│   └── Piggybacksdtec.msg
├── behavior_hero/
│   └── constants.py
├── hero_plus_robot/
│   └── urdf + meshes
└── tests/
    ├── multi_robot_controller.py
    ├── initializer.py
    ├── isaacsimconnector.sh
    ├── known_controllers.py
    └── robot_vs_radius.py
```

## Paper-aligned notes

From the manuscript:

- SyncSBC is a **dual-stage decentralized framework**.
- TCN is reported as the best-performing model among tested architectures for this setup.
- ESB and ESI reduce synchronization delay versus agreement-only methods.
- ESI generally lowers message count; ESB often achieves lower delay with higher communication.
- Demonstrated applications include **autonomous behavior switching** and **anomaly detection** on HeRo+ robots.

## Runtime assumptions and dependencies

This code assumes a ROS 2 environment (Humble-style APIs), including:

- `rclpy`, `tf2_msgs`, `geometry_msgs`, `std_msgs`
- Custom message package equivalent to the definitions in `hero_common_only_msg/`
- TensorFlow (for `agent_classifier/predictor.py`)
- `numpy`, `matplotlib`

It also assumes integration into a larger ROS 2 workspace/package layout where launch files and model artifacts (for example `TCN.keras`) are available.

## Minimal usage pattern (within your ROS 2 workspace)

1. Build and source your ROS 2 workspace containing these nodes and messages.
2. Start simulation/hardware publishers (`/tf`, `/<robot>/tof`, etc.).
3. Run predictor node(s) to publish `/prediction/<robot>`.
4. Run one Stage-1 node (`Puresh.py`, `averaging.py`, `entropy.py`, or `neighborv.py`).
5. Run one Stage-2 node (`esb.py` or `esi.py`) for synchronized belief alignment.

Common parameters in consensus nodes:

- `robot_names` (string array)
- `radius` (neighbor communication radius)
- `behavior`, `run_id`, `plot_dir`
- Method-specific thresholds (`tau`, `rho`, `hyst`, `sigma`, etc.)

## Isaac Sim helpers

- `tests/initializer.py`: places robots in scene layouts.
- `tests/isaacsimconnector.sh`: sends Python scripts to Isaac Sim TCP listeners.
- `tests/multi_robot_controller.py`: multi-robot behavior controller driver using known controller sets.


For videos and additional material, the manuscript references:

- https://sites.google.com/view/sync-sbc/home
