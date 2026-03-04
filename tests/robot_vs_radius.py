#!/usr/bin/env python3
from behavior_hero.plot_script3 import main as auc_main
import shlex
import subprocess
import os
import time


# ---- Fixed hyperparameters 
TAU   = 0.616
RHO   = 5.76
HYST  = 5.823
SIGMA = 0.02211

EPS   = 0.012
MAR   = 0.015
HYS   = 19.63
EPS_H = 0.063


def run(
    bag: str,
    behavior: str,
    run_id: int,
    duration: int,
    *,
    radius: float,
    base_d: str,
    launch_file: str,
    # always explicitly provided 
    tau: float,
    rho: float,
    hysteresis: float,
    sigma: float,
    eps: float,
    mar: float,
    hys: float,
    eps_h: float,
):
    """
    Same structure as your script: bash + setsid + kill -INT/-TERM/-KILL.
    Uses the launch_file you pass (baseline / baseline5 / baseline16).
    ALWAYS passes all hyperparams per-run.
    """
    os.makedirs(base_d, exist_ok=True)

    bag_q        = shlex.quote(bag)
    behavior_q   = shlex.quote(behavior)
    run_id_q     = shlex.quote(str(run_id))
    base_d_q     = shlex.quote(str(base_d))
    launch_q     = shlex.quote(str(launch_file))

    cmd = f"""
    set +e
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash

    # Bag in its own process group
    setsid ros2 bag play {bag_q} > /dev/null 2>&1 &
    BAGPID=$!

    sleep 1

    # Launch in its own process group
    setsid ros2 launch behavior_hero {launch_q} \
      behavior:={behavior_q} run_id:={run_id_q} plot_dir:={base_d_q} \
      radius:={radius} \
      tau:={tau} rho:={rho} hyst:={hysteresis} sigma:={sigma} \
      eps:={eps} mar:={mar} hys:={hys} eps_h:={eps_h} \
      > /dev/null 2>&1 &
    LAUNCHPID=$!

    sleep {duration}

    # Graceful stop: SIGINT to process groups
    kill -INT -$LAUNCHPID 2>/dev/null
    kill -INT -$BAGPID    2>/dev/null

    # Wait a bit
    for i in $(seq 1 10); do
      if ! kill -0 $LAUNCHPID 2>/dev/null && ! kill -0 $BAGPID 2>/dev/null; then
        break
      fi
      sleep 0.5
    done

    # Escalate
    kill -TERM -$LAUNCHPID 2>/dev/null
    kill -TERM -$BAGPID    2>/dev/null
    sleep 1
    kill -KILL -$LAUNCHPID 2>/dev/null
    kill -KILL -$BAGPID    2>/dev/null

    exit 0
    """

    subprocess.run(["bash", "-lc", cmd], start_new_session=True)

