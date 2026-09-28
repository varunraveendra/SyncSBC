#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration
from rcl_interfaces.msg import ParameterDescriptor, ParameterType as PT
from functools import partial
from rclpy.clock import Clock , ClockType

from tf2_msgs.msg import TFMessage
from hero_common.msg import Predictions
from hero_common.msg import Piggybacksdtec
import numpy as np
from behavior_hero.constants import TF_TO_ROBOT
import pathlib, csv
import matplotlib
matplotlib.use("Agg")  # safe headless
import matplotlib.pyplot as plt
import hashlib
import signal
import threading
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.task import Future
import matplotlib.pyplot as plt
import math
from collections import defaultdict


# esi parameters

tau_off = 0.3
mu_on = 0.25
# rng = np.random.default_rng(42)

mu_off = 0.1
rho_off = 10.0
USE_REGION_QUIET = True

# ALPHA = 0.3  # for moving average
# BETA = 0.6   # for moving average



def _build_robot_colors(names, cmap='tab20'):
    cm = plt.get_cmap(cmap)  # 20 distinct colors
    colors = {}
    for rn in names:
        # stable index from name (avoid Python's randomized hash)
        idx = int(hashlib.md5(rn.encode()).hexdigest()[:8], 16)
        colors[rn] = cm(idx % cm.N)
    return colors



class ConsensusESI(Node):
    def __init__(self):
        super().__init__('consensus_esi')

        # --- parameters ---
        self.declare_parameter(
            'robot_names', ['hero_plus_01'],
            ParameterDescriptor(type=PT.PARAMETER_STRING_ARRAY)
        )
        self.declare_parameter('plot_dir', str(pathlib.Path.home() / 'ros_logs'))
        self.declare_parameter('behavior', 'C1')
        self.declare_parameter('basename', 'metrics_esi' + self.get_parameter('behavior').get_parameter_value().string_value)
        self.plot_dir = pathlib.Path(self.get_parameter('plot_dir').get_parameter_value().string_value)
        self.declare_parameter( "run_id", -1, ParameterDescriptor(type=PT.PARAMETER_INTEGER) )
        self.declare_parameter(
            "tau", 0.1,
            ParameterDescriptor(type=PT.PARAMETER_DOUBLE)
        )

        self.declare_parameter(
            "rho", 0.0,
            ParameterDescriptor(type=PT.PARAMETER_DOUBLE)
        )

        self.declare_parameter(
            "hyst", 0.0,
            ParameterDescriptor(type=PT.PARAMETER_DOUBLE)
        )

        self.declare_parameter(
            "sigma", 0.0,
            ParameterDescriptor(type=PT.PARAMETER_DOUBLE)
        )
        self.declare_parameter(
            "use_region_quiet_gate", USE_REGION_QUIET,
            ParameterDescriptor(type=PT.PARAMETER_BOOL)
        )
        self.declare_parameter( "radius", 0.01, ParameterDescriptor(type=PT.PARAMETER_DOUBLE) )
        self.radius = self.get_parameter("radius").value

        # Read parameters
        self.run_id = self.get_parameter("run_id").value
        self.tau    = self.get_parameter("tau").value
        self.rho    = self.get_parameter("rho").value
        self.hys   = int(self.get_parameter("hyst").value)
        self.sigma  = self.get_parameter("sigma").value
        self.use_region_quiet_gate = bool(self.get_parameter("use_region_quiet_gate").value)
        self.basename = str(self.get_parameter('run_id').value)+ "_" + self.get_parameter('basename').get_parameter_value().string_value
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.robot_names = [str(x) for x in self.get_parameter('robot_names').value]
        self.robot_colors = _build_robot_colors(self.robot_names)  # store for reuse


        qos = QoSProfile(depth=10)
        self.consensus_pub = {}   
        qos_sensor = QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.tf_subs   =  self.create_subscription(TFMessage, '/tf', self.tf_callback, qos_sensor)   
        # rclpy.get_default_context().on_shutdown(self.cleanup)
        self.pred_subs   = []
        self.positions = {}
        self.seq = {}
        self.z = {}
         

        # ----------------------------
        # One-time lazy init (so you don't have to edit __init__ right now)
        # Recommended: move these to __init__ instead.
        # ----------------------------
        if not hasattr(self, "quiet_since"):
            self.quiet_since = defaultdict(lambda: None)        # per-robot Time or None
        if not hasattr(self, "stable_since"):
            self.stable_since = defaultdict(lambda: None)       # per-robot Time or None
        if not hasattr(self, "prev_final"):
            self.prev_final = defaultdict(lambda: (math.nan, math.nan, math.nan))  # per-robot (x,y,z)
        if not hasattr(self, "swarm_quiet_key"):
            self.swarm_quiet_key = defaultdict(lambda: None)   # per-robot aggregated swarm snapshot key
        if not hasattr(self, "swarm_changed_at"):
            self.swarm_changed_at = defaultdict(lambda: None)  # per-robot Time when snapshot last changed
        self.last_update = {}
        self.preds = {}
        self.hyst = {}
        self.latest_binary = {}
        self.previous_binary = {}
        self.latest_positive = {}
        self.clock= Clock(clock_type=ClockType.ROS_TIME)
        now_t = self.get_clock().now().to_msg()
        t = now_t.sec + now_t.nanosec * 1e-9
        self.start_time = t
        self.total_binaries= [[],[],[]]
        self.total_binaries_t= []
        self.last_changes_swarm = self.get_clock().now()
        self.influenced_count= 0
        self.talkto_neighbors= 0
        self.talkto_neighbors_l1= 0
        self.changed = {name: False for name in self.robot_names}

        self.create_timer(0.5, self.binary_logger)
        self.esi_subs   = []
        self.esi_pubs = {}
        self.esi_state_set = {}

        #book keeping for recently received neighbor states
        self.esi_state_rec = {}


        self.record = {name: [None, None, None] for name in self.robot_names}

        self.informed = {name: [False, False, False] for name in self.robot_names}

        self.esi_internal_state = {}

        for name in self.robot_names:
            pred_topic = f'/prediction/{name}'
            consensus_topic = f'/esiconsensuspb/{name}'
            self.positions[name] = []
            self.esi_state_set[name] = None
            self.esi_state_rec[name] = Piggybacksdtec()
            
            self.latest_positive[name] = None
            
            self.seq[name] = 0.0
            self.esi_internal_state[name] = [0,0,0.0,0.0]
            self.z[name] = [False, False, False]
            self.last_update[name] = self.get_clock().now()
            self.latest_binary[name] = [False, False, False]
            self.previous_binary[name] = [False, False, False]
            self.hyst[name] = {"ok_count_x":0, "ok_count_y":0, "ok_count_z":0, "done_x":False, "done_y":False, "done_z":False}
            temp = Predictions()
            self.preds[name] = temp
            self.consensus_pub[name] = self.create_publisher(Predictions, consensus_topic, qos)

            
            self.create_timer(0.4, partial(self.check_quiet, robot_name=name))
            self.create_timer(0.5, partial(self.esi_algorithm, robot_name=name))
            self.esi_pubs[name] = self.create_publisher(Piggybacksdtec, f'/esi/{name}', qos)
            self.pred_subs.append(
                self.create_subscription(Predictions, pred_topic, partial(self.pred_cb, robot_name=name), qos)
            )
            self.esi_subs.append(
                self.create_subscription(Piggybacksdtec, f'/esi/{name}', partial(self.esi_cb, robot_name=name), qos)
            )
            
        if not self.robot_names:
            self.get_logger().warn('No robot_names provided. Set parameter robot_names:=[...]')

    def binary_logger(self):
        sum_x = sum(1.0 if self.latest_binary[i][0] else 0.0 for i in self.robot_names)
        sum_y = sum(1.0 if self.latest_binary[i][1] else 0.0 for i in self.robot_names)
        sum_z = sum(1.0 if self.latest_binary[i][2] else 0.0 for i in self.robot_names)

        N = float(len(self.robot_names))
        self.total_binaries[0].append(sum_x / N)
        self.total_binaries[1].append(sum_y / N)
        self.total_binaries[2].append(sum_z / N)

        now_t = self.get_clock().now().to_msg()
        t = now_t.sec + now_t.nanosec * 1e-9
        
        self.total_binaries_t.append(t)
       

    def binary_state(self, con: list, preds: list, s_s: float, robot_name: str) -> bool:
        w = int(np.argmax(con))
  
        if con[w]>=self.tau and s_s<= self.rho:
            if w==0:
                self.hyst[robot_name]["ok_count_x"] += 1
                self.hyst[robot_name]["ok_count_y"] = 0
                self.hyst[robot_name]["ok_count_z"] = 0
            elif w==1:
                self.hyst[robot_name]["ok_count_y"] += 1
                self.hyst[robot_name]["ok_count_x"] = 0
                self.hyst[robot_name]["ok_count_z"] = 0
            elif w==2:
                self.hyst[robot_name]["ok_count_z"] += 1
                self.hyst[robot_name]["ok_count_x"] = 0
                self.hyst[robot_name]["ok_count_y"] = 0

        elif con[w]<tau_off or s_s>rho_off:
            if w==0:
                self.hyst[robot_name]["ok_count_x"] = max(0, self.hyst[robot_name]["ok_count_x"] - 1)
            elif w==1:
                self.hyst[robot_name]["ok_count_y"] = max(0, self.hyst[robot_name]["ok_count_y"] - 1)
            elif w==2:
                self.hyst[robot_name]["ok_count_z"] = max(0, self.hyst[robot_name]["ok_count_z"] - 1)

        self.hyst[robot_name]["done_x"] = self.hyst[robot_name]["ok_count_x"] >= self.hys
        self.hyst[robot_name]["done_y"] = self.hyst[robot_name]["ok_count_y"] >= self.hys
        self.hyst[robot_name]["done_z"] = self.hyst[robot_name]["ok_count_z"] >= self.hys
        
        return [self.hyst[robot_name]["done_x"], self.hyst[robot_name]["done_y"], self.hyst[robot_name]["done_z"]]


    def cleanup(self):
        self.get_logger().info("Cleaning up...")
        self.get_logger().info(f"Last changes esi: {self.last_changes_swarm}")
        self.get_logger().info(f"Influenced count esi: {self.influenced_count}")
        self.get_logger().info(f"Talked to neighbors esi: {self.talkto_neighbors}")
        self.get_logger().info(f"Talked to L1 neighbors esi: {self.talkto_neighbors_l1}")
        """Save two plots + CSVs on shutdown."""
        try:
            stamp = self.get_clock().now().to_msg()
            tag = f"{stamp.sec}_{stamp.nanosec:09d}"

            #=============== Latest Positive Timestamp per Robot ===============
            # Choose a reference: earliest positive seen, or use self.start_time if you store it.
            valid = []
            for rn in self.robot_names:
                ts = self.latest_positive.get(rn, None)  # can be None or int ns or rclpy Time
                if ts is None:
                    continue

                # Normalize to absolute nanoseconds
                if hasattr(ts, "nanoseconds"):         # rclpy.time.Time
                    abs_ns = ts.nanoseconds
                elif isinstance(ts, (int, np.integer)):  
                    abs_ns = int(ts)
                elif isinstance(ts, float):            
                    abs_ns = int(ts * 1e9)
                else:
                    continue

                valid.append((rn, abs_ns))

            if valid:
                t0_ns = min(ns for _, ns in valid)

                plt.figure(figsize=(11, 4.5))
                for rn, abs_ns in sorted(valid, key=lambda kv: kv[1]):
                    t_rel_s = (abs_ns - t0_ns) / 1e9
                    plt.axvline(x=t_rel_s, linestyle='--', linewidth=1.5, alpha=0.85,color=self.robot_colors.get(rn, None), label=rn)

                plt.xlabel('Time since first positive [s]')
                plt.ylabel('Latest positive (marker)')
                plt.title('Latest Positive Timestamp per Robot (vertical lines)')
                plt.legend(ncol=max(1, len(valid)//6))
                plt.tight_layout()
                png_path3 = self.plot_dir / f"{self.basename}_latest_positive_{tag}.png"
                plt.savefig(png_path3, dpi=130); plt.close()
                self.get_logger().info(f"Saved latest positive timestamp plot: {png_path3}")

                # CSV with both absolute (ns) and relative (s)
                csv_path3 = self.plot_dir / f"{self.basename}_latest_positive_{tag}.csv"
                with open(csv_path3, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['robot_name', 'latest_positive_abs_ns', 'latest_positive_rel_s', 'influenced_count', 'talkto_neighbors', 'talkto_neighbors_l1'])
                    for rn, abs_ns in sorted(valid, key=lambda kv: kv[1]):
                        w.writerow([rn, abs_ns, (abs_ns - t0_ns) / 1e9, self.influenced_count, self.talkto_neighbors, self.talkto_neighbors_l1])
                self.get_logger().info(f"Saved latest positive timestamp CSV: {csv_path3}")
            else:
                self.get_logger().warn("No latest_positive timestamps to plot.")

            
            #================ Total Binaries over Time ================
            
            if getattr(self, "total_binaries_t", None) and len(self.total_binaries) >= 3:
                # --- 0 plot ---
                plt.figure(figsize=(11, 4.5))
                plt.plot(self.total_binaries_t, self.total_binaries[0], linewidth=1.7)  # add drawstyle='steps-post' if you prefer
                plt.xlabel('Time [s]'); plt.ylabel('binary_0'); plt.title('Total Binaries (0) over time')
                plt.tight_layout()
                png_tb0 = self.plot_dir / f"{self.basename}_total_binaries_0_{tag}.png"
                plt.savefig(png_tb0, dpi=130); plt.close()
                self.get_logger().info(f"Saved total_binaries[0] plot: {png_tb0}")

                # --- 1 plot ---
                plt.figure(figsize=(11, 4.5))
                plt.plot(self.total_binaries_t, self.total_binaries[1], linewidth=1.7)
                plt.xlabel('Time [s]'); plt.ylabel('binary_1'); plt.title('Total Binaries (1) over time')
                plt.tight_layout()
                png_tb1 = self.plot_dir / f"{self.basename}_total_binaries_1_{tag}.png"
                plt.savefig(png_tb1, dpi=130); plt.close()
                self.get_logger().info(f"Saved total_binaries[1] plot: {png_tb1}")

                # --- 2 plot ---
                plt.figure(figsize=(11, 4.5))
                plt.plot(self.total_binaries_t, self.total_binaries[2], linewidth=1.7)
                plt.xlabel('Time [s]'); plt.ylabel('binary_2'); plt.title('Total Binaries (2) over time')
                plt.tight_layout()
                png_tb2 = self.plot_dir / f"{self.basename}_total_binaries_2_{tag}.png"
                plt.savefig(png_tb2, dpi=130); plt.close()
                self.get_logger().info(f"Saved total_binaries[2] plot: {png_tb2}")

                # --- CSV ---
                csv_tb = self.plot_dir / f"{self.basename}_total_binaries_{tag}.csv"
                with open(csv_tb, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['t_sec','binary_0','binary_1','binary_2'])
                    w.writerows(zip(self.total_binaries_t,
                                    self.total_binaries[0],
                                    self.total_binaries[1],
                                    self.total_binaries[2]))
                self.get_logger().info(f"Saved total_binaries CSV: {csv_tb}")
            else:
                self.get_logger().warn("No total_binaries recorded or missing components 0/1/2.")



        except Exception as e:
            self.get_logger().error(f"Cleanup/plot error: {e}")
   
    
    def check_quiet(self, robot_name: str):
        # ----------------------------
        # Compute neighbors / region
        # ----------------------------
        dist = {}
        for i in self.robot_names:
            if i == robot_name:
                continue
            try:
                dist[i] = (
                    (self.positions[robot_name][0] - self.positions[i][0]) ** 2
                    + (self.positions[robot_name][1] - self.positions[i][1]) ** 2
                ) ** 0.5
            except Exception:
                self.get_logger().warn(f"Position not found for hero_robot {i}")
                continue

        top = sorted(dist.items(), key=lambda x: x[1])

        # region tokens: neighbors + me, as 1-char tokens
        region_tokens = set()
        region_members = {robot_name}
        for k, v in top:
            if v <= self.radius and self.radius > 0:
                region_members.add(k)
                try:
                    rid_n = int(k.split("_")[-1])
                except Exception:
                    continue
                tok = self._rid_to_token(rid_n)   # '1'..'9' or 'd'..'j'
                if tok:
                    region_tokens.add(tok)

        rid = int(robot_name.split("_")[-1])
        my_tok = self._rid_to_token(rid)
        if my_tok:
            region_tokens.add(my_tok)

        now = self.get_clock().now()

        swarm = self.get_swarm_state(robot_name)

        # members strings live at [2]
        x_entry = swarm[0][2]
        y_entry = swarm[1][2]
        z_entry = swarm[2][2]

        # (optional but good) owners live at [0]
        x_owner = swarm[0][0]
        y_owner = swarm[1][0]
        z_owner = swarm[2][0]

        # region is a subset of entry digits
        def region_subset_of(entry: str) -> bool:
            if not entry or not region_tokens:
                return False
            entry_norm = self._norm_members(str(entry))     # ensures only 1-char tokens
            return region_tokens.issubset(set(entry_norm))

        # ----------------------------
        # Quiet gate
        # ----------------------------
        if self.use_region_quiet_gate:
            swarm_key = (
                (int(swarm[0][0]), int(swarm[0][1]), self._norm_members(str(swarm[0][2]))),
                (int(swarm[1][0]), int(swarm[1][1]), self._norm_members(str(swarm[1][2]))),
                (int(swarm[2][0]), int(swarm[2][1]), self._norm_members(str(swarm[2][2]))),
            )
            if self.swarm_quiet_key[robot_name] != swarm_key:
                self.swarm_quiet_key[robot_name] = swarm_key
                self.swarm_changed_at[robot_name] = now
                quiet = False
            else:
                changed_at = self.swarm_changed_at[robot_name]
                quiet = (
                    changed_at is not None
                    and (now - changed_at) > Duration(seconds=2.0)
                )
        else:
            quiet = (now - self.last_update[robot_name]) > Duration(seconds=2.0)

        in_any = (
            region_subset_of(x_entry)
            or region_subset_of(y_entry)
            or region_subset_of(z_entry)
        )

        if not quiet:
            # Not quiet yet → reset quiet/stability trackers
            self.quiet_since[robot_name] = None
            self.stable_since[robot_name] = None

            # keep prev_final in sync
            cur_x = float(rid) if region_subset_of(x_entry) else float(rid - 1)
            cur_y = float(rid) if region_subset_of(y_entry) else float(rid - 1)
            cur_z = float(rid) if region_subset_of(z_entry) else float(rid - 1)
            self.prev_final[robot_name] = (cur_x, cur_y, cur_z)
            return

        # ---------------------------
        # ----------------------------
        cur_x = float(rid) if region_subset_of(x_entry) else float(rid - 1)
        cur_y = float(rid) if region_subset_of(y_entry) else float(rid - 1)
        cur_z = float(rid) if region_subset_of(z_entry) else float(rid - 1)
        cur = (cur_x, cur_y, cur_z)

        # Entering quiet for first time
        if self.quiet_since[robot_name] is None:
            self.quiet_since[robot_name] = now
            self.stable_since[robot_name] = now 
            self.prev_final[robot_name] = cur
            conclude = False
        else:
            prev = self.prev_final[robot_name]
            if cur == prev:
                conclude = (now - self.stable_since[robot_name]) > Duration(seconds=1.0)
            else:
                self.prev_final[robot_name] = cur
                self.stable_since[robot_name] = now
                conclude = False

        # ----------------------------
        # Publish only when quiet + stable
        # Optional extra safety: require no one holds any token when concluding
        # ----------------------------
        no_token_holders = (x_owner == 0 and y_owner == 0 and z_owner == 0)

        if conclude and no_token_holders:
            # latest_positive should be the time when we conclude
            if in_any:
                if self.latest_positive[robot_name] is None:
                    now_t = self.get_clock().now().to_msg()
                    t = now_t.sec + now_t.nanosec * 1e-9
                    self.latest_positive[robot_name] = t - self.start_time
            else:
                self.latest_positive[robot_name] = None

            consensus_msg = Predictions()
            consensus_msg.x = cur_x
            consensus_msg.y = cur_y
            consensus_msg.z = cur_z
            self.consensus_pub[robot_name].publish(consensus_msg)

        
    def central_debugger(self):
        add = []
        for robot_name in self.robot_names:
            add.append(self.z[robot_name][1])
        # print("Current consensus states:", add)


    def pred_cb(self, msg: Predictions, robot_name: str):
        self.preds[robot_name] = msg

    def esi_cb(self, msg: Piggybacksdtec, robot_name: str):
        

        self.esi_state_rec[robot_name] = msg


    def _rid_to_token(self, rid: int) -> str:
        """1..9 -> '1'..'9', 10..16 -> 'd'..'j'"""
        if 1 <= rid <= 9:
            return str(rid)
        # 10->d, 11->e, 12->f, 13->g, 14->h, 15->i, 16->j
        if 10 <= rid <= 16:
            return chr(ord('d') + (rid - 10))
        return ""  # out of supported range

    def _token_rank(self, ch: str) -> int:
        """Total order for stable sorting: 1..9 then d..j."""
        if ch.isdigit():
            v = int(ch)
            if 1 <= v <= 9:
                return v
        if 'd' <= ch <= 'j':
            return 10 + (ord(ch) - ord('d'))
        return 999  # unknowns last (or you can drop them)

    def _norm_members(self, s: str) -> str:
        if not s:
            return ""
        allowed = []
        for ch in s:
            if ('1' <= ch <= '9') or ('d' <= ch <= 'j'):
                allowed.append(ch)
        # unique + deterministic order
        uniq = set(allowed)
        return "".join(sorted(uniq, key=self._token_rank))

    def _add_member(self, members: str, token: str) -> str:
        return self._norm_members(members + token)

    def _del_member(self, members: str, token: str) -> str:
        # safe even if token not present
        return self._norm_members(members.replace(token, ""))
    
    def _ensure_esi_lease_state(self):
        # per-robot local epochs for 3 lanes
        if not hasattr(self, "epoch_local") or self.epoch_local is None:
            self.epoch_local = defaultdict(lambda: [0, 0, 0])  # robot_name -> [ep_a, ep_b, ep_c]

    def _next_epoch(self, robot_name: str, k: int, observed_epoch: int) -> int:
        e = max(self.epoch_local[robot_name][k], int(observed_epoch)) + 1
        self.epoch_local[robot_name][k] = e
        return e

    def get_swarm_state(self, robot_name: str):
        """
        Returns per-lane state as:
            [(owner_a, epoch_a, members_a),
            (owner_b, epoch_b, members_b),
            (owner_c, epoch_c, members_c)]
        Aggregation rule (async-safe):
            pick the winning claim by max(epoch, owner) across neighbors.
        """
        # ---- neighbors ----
        dist = {}
        for i in self.robot_names:
            if i == robot_name:
                continue
            try:
                dist[i] = (
                    (self.positions[robot_name][0] - self.positions[i][0]) ** 2
                    + (self.positions[robot_name][1] - self.positions[i][1]) ** 2
                ) ** 0.5
            except Exception:
                self.get_logger().warn(f"Position not found for hero_robot {i}")
                continue

        top = sorted(dist.items(), key=lambda x: x[1])  
        neigh_names = [k for k, v in top if v <= self.radius and self.radius > 0]

        # Optionally include "my own last received" if you store it in esi_state_rec
      
        if robot_name in self.esi_state_rec:
            if robot_name not in neigh_names:
                neigh_names.append(robot_name)

        out = []
        suffixes = ["a", "b", "c"]

        for suf in suffixes:
            best_owner = 0
            best_epoch = 0

            # pass 1: find best (epoch, owner)
            for n in neigh_names:
                msg = self.esi_state_rec.get(n, None)
                if msg is None:
                    continue
                try:
                    owner = int(getattr(msg, f"own_{suf}"))
                    epoch = int(getattr(msg, f"ep_{suf}"))
                except Exception:
                    continue

                if (epoch > best_epoch) or (epoch == best_epoch and owner > best_owner):
                    best_epoch = epoch
                    best_owner = owner

            # pass 2: union members among all neighbors that match best (epoch, owner)
            members_set = set()
            for n in neigh_names:
                msg = self.esi_state_rec.get(n, None)
                if msg is None:
                    continue
                try:
                    owner = int(getattr(msg, f"own_{suf}"))
                    epoch = int(getattr(msg, f"ep_{suf}"))
                    members = self._norm_members(getattr(msg, f"int_{suf}"))
                except Exception:
                    continue

                if owner == best_owner and epoch == best_epoch:
                    members_set |= set(members)

            best_members = "".join(sorted(members_set, key=self._token_rank)) if members_set else ""
            if best_owner != 0:
                self.get_logger().info(
                    f"[SWARM] {robot_name} lane {suf}: picked (own={best_owner},ep={best_epoch}) members='{best_members}'"
                )
            out.append((best_owner, best_epoch, best_members))


        return out


    def esi_algorithm(self, robot_name: str):
        """
        Token/lease consensus over 3 lanes (a,b,c):
        - lane k chosen per tick (argmax preds)
        - if not confident in lane k: try to acquire token & remove self from members
        - if confident: add self to members, release token if you hold it
        - token claim uses (epoch, owner) to avoid simultaneous steal split-brain
        """
        self._ensure_esi_lease_state()

        # ---- init internal state (keep your original behavior) ----
        if (self.esi_internal_state[robot_name][0] == 0.0
            and self.esi_internal_state[robot_name][1] == 0.0
            and self.esi_internal_state[robot_name][2] == 0.0):
            self.esi_internal_state[robot_name] = [self.preds[robot_name].x,
                                                    self.preds[robot_name].y,
                                                    self.preds[robot_name].z]

        if self.esi_state_set[robot_name] is None:
            self.esi_state_set[robot_name] = Piggybacksdtec()
            self.esi_state_set[robot_name].x = self.preds[robot_name].x
            self.esi_state_set[robot_name].y = self.preds[robot_name].y
            self.esi_state_set[robot_name].z = self.preds[robot_name].z

        # ---- neighbor set for s_s ----
        dist = {}
        for i in self.robot_names:
            if i == robot_name:
                continue
            try:
                dist[i] = (
                    (self.positions[robot_name][0] - self.positions[i][0]) ** 2
                    + (self.positions[robot_name][1] - self.positions[i][1]) ** 2
                ) ** 0.5
            except Exception:
                self.get_logger().warn(f"Position not found for hero_robot {i}")
                continue

        top = sorted(dist.items(), key=lambda x: x[1])  
        topk = [k for k, v in top if v <= self.radius and self.radius > 0] 


        # ---- disagreement term ----
        s_s = 0.0
        for j in topk:
            if j not in self.esi_state_rec:
                continue
            dx = self.esi_state_set[robot_name].x - self.esi_state_rec[j].x
            dy = self.esi_state_set[robot_name].y - self.esi_state_rec[j].y
            dz = self.esi_state_set[robot_name].z - self.esi_state_rec[j].z
            s_s += dx * dx + dy * dy + dz * dz

        # ---- tracking error ----
        r0 = self.preds[robot_name].x - self.esi_state_set[robot_name].x
        r1 = self.preds[robot_name].y - self.esi_state_set[robot_name].y
        r2 = self.preds[robot_name].z - self.esi_state_set[robot_name].z
        r_s = float(r0 * r0 + r1 * r1 + r2 * r2)

        # ---- "help" gate (keep your spirit, simplified) ----
        swarm_state = self.get_swarm_state(robot_name)  # [(own, ep, members) x3]
        neigh_tokens = []
        for n in topk:
            try:
                rid_n = int(n.split("_")[-1])
            except Exception:
                continue
            tok = self._rid_to_token(rid_n)
            if tok:
                neigh_tokens.append(tok)

        help_flag = True
        for ntok in neigh_tokens:
            for kk in range(3):
                if ntok in swarm_state[kk][2]:
                    help_flag = False
                    break
            if not help_flag:
                break

        # ---- event-trigger condition ----
        
        rid = int(robot_name.split("_")[-1])
        my_tok = self._rid_to_token(rid)

        for k in range(3):
            if my_tok and (my_tok in swarm_state[k][2]):
                self.informed[robot_name][k] = True
                break
            self.informed[robot_name][k] = False

        informed_any = (self.informed[robot_name][0] or self.informed[robot_name][1] or self.informed[robot_name][2])
        holds_token = any(int(swarm_state[kk][0]) == rid for kk in range(3))
        if not ((r_s >= self.sigma * s_s) or help_flag or (not informed_any) or holds_token):
            return

        # ---- update local set to current preds ----
        self.esi_state_set[robot_name].x = self.preds[robot_name].x
        self.esi_state_set[robot_name].y = self.preds[robot_name].y
        self.esi_state_set[robot_name].z = self.preds[robot_name].z

        preds = [self.preds[robot_name].x, self.preds[robot_name].y, self.preds[robot_name].z]
        con = [self.esi_state_set[robot_name].x, self.esi_state_set[robot_name].y, self.esi_state_set[robot_name].z]
        # ===================== DEBUG: SD-ETC SNAPSHOT =====================
        try:
            preds = [
                self.preds[robot_name].x,
                self.preds[robot_name].y,
                self.preds[robot_name].z,
            ]
            argmax_k = max(range(3), key=lambda i: preds[i])

            swarm_dbg = []
            for k in range(3):
                owner, epoch, members = swarm_state[k]
                swarm_dbg.append(
                    f"k={k}(own={owner},ep={epoch},m='{members}')"
                )

        except Exception as e:
            self.get_logger().error(f"[esi-DEBUG] {robot_name} debug failed: {e}")
# =================================================================


        self.latest_binary[robot_name] = self.binary_state(con, preds, s_s, robot_name) # replace this with your favorite stage 1 consensus method

        
        k = next((kk for kk in range(3) if int(swarm_state[kk][0]) == rid),
                 max(range(3), key=lambda i: preds[i]))
        suf = ["a", "b", "c"][k]

        # normalize robot id string (avoid "06" vs "6")
        rid = int(robot_name.split("_")[-1])
        my_tok = self._rid_to_token(rid)   # '1'..'9' or 'd'..'j'

        owner, epoch, members = swarm_state[k]
        members = self._norm_members(members)

        my_conf = bool(self.latest_binary[robot_name][k])
        token_free = (int(owner) == 0)
        i_hold = (int(owner) == rid)
        can_edit = i_hold

        for kk in range(3):
            if self.record[robot_name][kk] is not None and int(swarm_state[kk][0]) != rid:
                self.record[robot_name][kk] = None

        # If your robot id is outside 1..16, you can just no-op membership edits:
        if not my_tok:
            return

        
        if i_hold:
            # Give simultaneous claims one timer cycle to settle before commit.
            if self.record[robot_name][k] == int(epoch):
                self.record[robot_name][k] = -int(epoch)
                return

            # Build message by copying current swarm state (all lanes)
            temp = Piggybacksdtec()
            temp.x = self.preds[robot_name].x
            temp.y = self.preds[robot_name].y
            temp.z = self.preds[robot_name].z

            for kk, ss in enumerate(["a", "b", "c"]):
                o, e, m = swarm_state[kk]
                setattr(temp, f"own_{ss}", int(o))
                setattr(temp, f"ep_{ss}", int(e))
                setattr(temp, f"inf_{ss}", 0)
                setattr(temp, f"int_{ss}", self._norm_members(m))
                setattr(temp, f"bint_{ss}", self._norm_members(m))

            new_members = self._add_member(members, my_tok) if my_conf else self._del_member(members, my_tok)
            new_epoch = self._next_epoch(robot_name, k, epoch)

            setattr(temp, f"own_{suf}", 0)             # release ownership
            setattr(temp, f"ep_{suf}", new_epoch)     # commit a newer version
            setattr(temp, f"int_{suf}", new_members)
            setattr(temp, f"bint_{suf}", new_members)
            self.record[robot_name][k] = None

            self.get_logger().info(
                f"[esi] {robot_name} releases lane k={k} -> members '{members}' -> '{new_members}'"
            )

            self.esi_state_rec[robot_name] = temp
            self.esi_pubs[robot_name].publish(temp)
            self.talkto_neighbors_l1= self.talkto_neighbors_l1 + 1 if self.latest_positive[robot_name] is None else self.talkto_neighbors_l1
           
            self.last_update[robot_name] = self.get_clock().now()
            return


        # ---- Build message by copying current swarm state (all lanes) ----
        temp = Piggybacksdtec()

        # include your real-valued prediction fields if you use them
        temp.x = self.preds[robot_name].x
        temp.y = self.preds[robot_name].y
        temp.z = self.preds[robot_name].z

        # copy all lanes from current aggregate (so we only change one lane)
        for kk, ss in enumerate(["a", "b", "c"]):
            o, e, m = swarm_state[kk]
            setattr(temp, f"own_{ss}", int(o))
            setattr(temp, f"ep_{ss}", int(e))
            setattr(temp, f"inf_{ss}", 0)  # keep/ignore old field if still in msg
            setattr(temp, f"int_{ss}", self._norm_members(m))
            setattr(temp, f"bint_{ss}", self._norm_members(m))

        # Claim a free token before changing membership.
        if token_free and ((my_tok in members) != my_conf):
            new_epoch = self._next_epoch(robot_name, k, epoch)
            setattr(temp, f"own_{suf}", rid)
            setattr(temp, f"ep_{suf}", new_epoch)
            self.record[robot_name][k] = new_epoch
            self.esi_state_rec[robot_name] = temp
            self.esi_pubs[robot_name].publish(temp)
            self.talkto_neighbors_l1= self.talkto_neighbors_l1 + 1 if self.latest_positive[robot_name] is None else self.talkto_neighbors_l1
            self.last_update[robot_name] = self.get_clock().now()
            return

        changed = False

        if not my_conf:
            
            if can_edit:
                if my_tok in members:
                    new_members = self._del_member(members, my_tok)
                    if new_members != members:
                        members = new_members
                        changed = True

                        # keep owner/epoch as-is (do NOT steal)
                        setattr(temp, f"own_{suf}", int(owner))
                        setattr(temp, f"ep_{suf}", int(epoch))
                        setattr(temp, f"int_{suf}", members)
                        setattr(temp, f"bint_{suf}", members)

        else:
            # Confident: add self; if you hold token, release it.
            if can_edit:
                new_members = self._add_member(members, my_tok)
                if new_members != members:
                    members = new_members
                    changed = True

                

                setattr(temp, f"own_{suf}", int(owner))
                setattr(temp, f"ep_{suf}", int(epoch))
                setattr(temp, f"int_{suf}", members)
                setattr(temp, f"bint_{suf}", members)

        # ---- publish only if we actually changed something ----
        if changed:
            self.last_update[robot_name] = self.get_clock().now()

        self.esi_pubs[robot_name].publish(temp)
        self.talkto_neighbors_l1= self.talkto_neighbors_l1 + 1 if self.latest_positive[robot_name] is None else self.talkto_neighbors_l1
                        
                        

    def tf_callback(self, msg: TFMessage):
        for i in msg.transforms:
            child_name = i.child_frame_id
            c = i.transform.translation
            self.positions[TF_TO_ROBOT[child_name]] = [c.x,c.y]
    

def main():
    rclpy.init()
    node = ConsensusESI()  # your node

    # --- graceful signal handling ---
    stopping = threading.Event()
    signal.signal(signal.SIGINT,  lambda *_: stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())

    exec = SingleThreadedExecutor()
    exec.add_node(node)

    try:
        # spin in small increments so we can react to signals quickly
        while not stopping.is_set():
            exec.spin_once(timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        # ---- run cleanup exactly once, and WAIT for it (bounded) ----
        try:
            # Prefer an async cleanup that returns a Future you fulfill
            # when all timers/threads/files/subprocesses are closed.
            if hasattr(node, "async_cleanup"):
                fut = node.async_cleanup()  # type: Future
                # keep executor alive so callbacks/joins can run
                exec.spin_until_future_complete(fut, timeout_sec=5.0)
            else:
                # Fallback: sync cleanup() 
                node.cleanup()
        except Exception as e:
            node.get_logger().warn(f"cleanup error: {e}")

        # ---- teardown ROS entities ----
        exec.remove_node(node)
        node.destroy_node()

        # guard so we don’t shutdown twice
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
