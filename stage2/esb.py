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
from hero_common.msg import Boolconsensus
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


# esb parameters
tau_on = 0.5
tau_off = 0.3
mu_on = 0.25
mu_off = 0.1
rho_on = 7.0
rho_off = 10.0
SIGMA = 0.09


def _build_robot_colors(names, cmap='tab20'):
    cm = plt.get_cmap(cmap)  # 20 distinct colors
    colors = {}
    for rn in names:
        # stable index from name (avoid Python's randomized hash)
        idx = int(hashlib.md5(rn.encode()).hexdigest()[:8], 16)
        colors[rn] = cm(idx % cm.N)
    return colors


class ConsensusESB(Node):
    def __init__(self):
        super().__init__('consensus_esb')

        # --- parameters ---
        self.declare_parameter(
            'robot_names', ['hero_plus_01'],
            ParameterDescriptor(type=PT.PARAMETER_STRING_ARRAY)
        )
        self.declare_parameter('plot_dir', str(pathlib.Path.home() / 'ros_logs'))
        self.declare_parameter('behavior', 'C1')
        self.declare_parameter('basename', 'metrics_esb' + self.get_parameter('behavior').get_parameter_value().string_value)
        self.plot_dir = pathlib.Path(self.get_parameter('plot_dir').get_parameter_value().string_value)
        self.declare_parameter( "run_id", -1, ParameterDescriptor(type=PT.PARAMETER_INTEGER) )
        # self.declare_parameter( "radius", 0.01, ParameterDescriptor(type=PT.PARAMETER_DOUBLE) )

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


        self.declare_parameter( "radius", 0.01, ParameterDescriptor(type=PT.PARAMETER_DOUBLE) )
        self.radius = self.get_parameter("radius").value

        # Read parameters
        self.run_id = self.get_parameter("run_id").value
        
        self.tau    = self.get_parameter("tau").value
        self.rho    = self.get_parameter("rho").value
        self.hys   = int(self.get_parameter("hyst").value)
        self.sigma  = self.get_parameter("sigma").value
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
        self.pred_subs   = []
        self.boolcon_subs = []
        self.boolcon_pubs = {}
        self.positions = {}
        self.seq = {}
        self.z = {}
        self.last_update = {}
        self.defaulter_count_t = {}
        self.defaulter_count_plt = {}
        self.preds = {}
        self.boolcon_robots = {}
        self.hyst = {}
        self.eps= 0.04
        self.eps_h= 0.1
        self.hyst_max= 7
        self.latest_binary = {}
        self.defaulter_count = {}
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

        self.create_timer(0.5, self.binary_logger)
        self.esb_subs   = []
        self.esb_pubs = {}
        self.esb_state_set = {}
        self.esb_state_rec = {}
        self.esb_internal_state = {}

        for name in self.robot_names:
            pred_topic = f'/prediction/{name}'
            consensus_topic = f'/esbconsensus/{name}'
            self.positions[name] = []
            self.esb_state_set[name] = None
            self.esb_state_rec[name] = Predictions()
            self.defaulter_count[name] = 0
            self.defaulter_count_plt[name] = []
            self.latest_positive[name] = 0.0
            self.defaulter_count_t[name] = []
            self.seq[name] = 0.0
            self.esb_internal_state[name] = [0,0,0.0,0.0]
            self.z[name] = [False, False, False]
            self.last_update[name] = self.get_clock().now()
            self.latest_binary[name] = [False, False, False]
            self.hyst[name] = {"ok_count_x":0, "ok_count_y":0, "ok_count_z":0, "done_x":False, "done_y":False, "done_z":False}
            temp = Predictions()
            self.preds[name] = temp
            temp1= Boolconsensus()
            self.boolcon_robots[name] = temp1
            self.consensus_pub[name] = self.create_publisher(Predictions, consensus_topic, qos)

            self.create_timer(0.8, partial(self.roll_epoch, robot_name=name))
            self.create_timer(0.25, partial(self.bool_contribution, robot_name=name))
            self.create_timer(0.4, partial(self.check_quiet, robot_name=name))
            self.create_timer(0.5, partial(self.esb_algorithm, robot_name=name))
            self.boolcon_pubs[name] = self.create_publisher(Boolconsensus, f'/esbboolconsensus/{name}', qos)
            self.esb_pubs[name] = self.create_publisher(Predictions, f'/esb/{name}', qos)
            self.pred_subs.append(
                self.create_subscription(Predictions, pred_topic, partial(self.pred_cb, robot_name=name), qos)
            )
            self.esb_subs.append(
                self.create_subscription(Predictions, f'/esb/{name}', partial(self.esb_cb, robot_name=name), qos)
            )
            self.boolcon_subs.append(
                self.create_subscription(Boolconsensus, f'/esbboolconsensus/{name}', partial(self.bool_cb, robot_name=name), qos)
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

    def bool_cb(self, msg: Boolconsensus, robot_name: str):
        self.boolcon_robots[robot_name] = msg

    def cleanup(self):
        self.get_logger().info("Cleaning up...")
        self.get_logger().info(f"Last changes esb: {self.last_changes_swarm}")
        self.get_logger().info(f"Influenced count esb: {self.influenced_count}")
        self.get_logger().info(f"Talked to neighbors esb: {self.talkto_neighbors}")
        self.get_logger().info(f"Talked to L1 neighbors esb: {self.talkto_neighbors_l1}")
        """Save two plots + CSVs on shutdown."""
        try:
            stamp = self.get_clock().now().to_msg()
            tag = f"{stamp.sec}_{stamp.nanosec:09d}"
           

            #============== Latest Positive Timestamp per Robot ===============
            # Choose a reference: earliest positive seen, or use self.start_time if you store it.
            valid = []
            for rn in self.robot_names:
                ts = self.latest_positive.get(rn, None)  # can be None or int ns or rclpy Time
                if ts is None:
                    continue

                # Normalize to absolute nanoseconds
                if hasattr(ts, "nanoseconds"):         # rclpy.time.Time
                    abs_ns = ts.nanoseconds
                elif isinstance(ts, (int, np.integer)):  # you stored ns as int
                    abs_ns = int(ts)
                elif isinstance(ts, float):            # you stored seconds as float (fallback)
                    abs_ns = int(ts * 1e9)
                else:
                    continue

                valid.append((rn, abs_ns))

            if valid:
                # reference = earliest positive; alternatively, use self.start_time.nanoseconds
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



    
    def bool_contribution(self,robot_name: str):
        dist = {}
        for i in self.robot_names:
            if i != robot_name:
                # print(self.positions[robot_name], self.positions[i])
                try:
                    dist[i] = ((self.positions[robot_name][0]-self.positions[i][0])**2 + (self.positions[robot_name][1]-self.positions[i][1])**2)**0.5
                except :
                    self.get_logger().warn(f"Position not found for hero_robot {i}")
                    continue

        top = sorted(dist.items(), key=lambda x: x[1])
        topk = [k for k, v in top if v <= self.radius and self.radius > 0]
        # randomly drop neighbors to simulate communication issues (optional, can be commented out)
        

        for j in topk:
            # if self.boolcon_robots[robot_name].k > self.boolcon_robots[j].k:
            #     continue
            time_changed= False
            if self.boolcon_robots[robot_name].k < self.boolcon_robots[j].k:
                time_changed= True
                self.seq[robot_name] = self.boolcon_robots[j].k
                self.z[robot_name] = list(self.latest_binary[robot_name])
            
            if self.boolcon_robots[j].x == True and self.z[robot_name][0] == False:
                self.defaulter_count[robot_name] += 2
                self.defaulter_count_plt[robot_name].append(self.defaulter_count[robot_name])
                self.defaulter_count_t[robot_name].append(self.get_clock().now().to_msg().sec)
            if self.boolcon_robots[j].y == True and self.z[robot_name][1] == False:
                self.defaulter_count[robot_name] += 2
                self.defaulter_count_plt[robot_name].append(self.defaulter_count[robot_name])
                self.defaulter_count_t[robot_name].append(self.get_clock().now().to_msg().sec)
            if self.boolcon_robots[j].z == True and self.z[robot_name][2] == False:
                self.defaulter_count[robot_name] += 2
                self.defaulter_count_plt[robot_name].append(self.defaulter_count[robot_name])
                self.defaulter_count_t[robot_name].append(self.get_clock().now().to_msg().sec)
            if self.boolcon_robots[j].x == self.z[robot_name][0] == True and self.defaulter_count[robot_name] >= 0:
                self.defaulter_count[robot_name] -= 1
                self.defaulter_count_plt[robot_name].append(self.defaulter_count[robot_name])
                self.defaulter_count_t[robot_name].append(self.get_clock().now().to_msg().sec)
            if self.boolcon_robots[j].y == self.z[robot_name][1] == True and self.defaulter_count[robot_name] >= 0:
                self.defaulter_count[robot_name] -= 1
                self.defaulter_count_plt[robot_name].append(self.defaulter_count[robot_name])
                self.defaulter_count_t[robot_name].append(self.get_clock().now().to_msg().sec)
            if self.boolcon_robots[j].z == self.z[robot_name][2] == True and self.defaulter_count[robot_name] >= 0:
                self.defaulter_count[robot_name] -= 1
                self.defaulter_count_plt[robot_name].append(self.defaulter_count[robot_name])
                self.defaulter_count_t[robot_name].append(self.get_clock().now().to_msg().sec)

            new_z_x = self.z[robot_name][0] and self.boolcon_robots[j].x
            new_z_y = self.z[robot_name][1] and self.boolcon_robots[j].y
            new_z_z = self.z[robot_name][2] and self.boolcon_robots[j].z

            

            changed = False

            if new_z_x != self.z[robot_name][0] :
                self.z[robot_name][0] = new_z_x
                changed = True
                # self.get_logger().info(f"here {j}{self.boolcon_robots[j].x} to {robot_name}{self.z[robot_name][0]}")
            if new_z_y != self.z[robot_name][1]:
                self.z[robot_name][1] = new_z_y
                changed = True
                # self.get_logger().info(f"here {j}{self.boolcon_robots[j].y} to {robot_name}{self.z[robot_name][1]}")
            if new_z_z != self.z[robot_name][2]:
                self.z[robot_name][2] = new_z_z
                changed = True
            
            if changed or time_changed:
                self.last_changes_swarm = self.get_clock().now() if changed else self.last_changes_swarm
                self.influenced_count= self.influenced_count + 1 if changed else self.influenced_count
                # print(self.seq[robot_name],"at seq", self.latest_binary[robot_name], new_z_x, new_z_y, new_z_z, robot_name,"influenced by", j)
                bmsg = Boolconsensus()
                bmsg.k = self.seq[robot_name]
                bmsg.x = self.z[robot_name][0]
                bmsg.y = self.z[robot_name][1]
                bmsg.z = self.z[robot_name][2]
                self.boolcon_pubs[robot_name].publish(bmsg)
                self.last_update[robot_name] = self.get_clock().now()
                self.talkto_neighbors= self.talkto_neighbors + 1 if self.latest_positive[robot_name] is None else self.talkto_neighbors

    def roll_epoch(self, robot_name: str):
        self.seq[robot_name] += 1
        changed = False
        if self.z[robot_name] != self.latest_binary[robot_name]:
            changed = True
        self.z[robot_name] = list(self.latest_binary[robot_name])
        # print(f'{robot_name} rolling epoch to {self.seq[robot_name]} with {self.z[robot_name]}')
        bmsg = Boolconsensus()
        bmsg.k = self.seq[robot_name]
        bmsg.x = self.z[robot_name][0]
        bmsg.y = self.z[robot_name][1]
        bmsg.z = self.z[robot_name][2]
        self.boolcon_pubs[robot_name].publish(bmsg)
        self.talkto_neighbors= self.talkto_neighbors + 1 if self.latest_positive[robot_name] is None else self.talkto_neighbors
        if changed:
            self.last_update[robot_name] = self.get_clock().now()
    
    def check_timeout(self, robot_name: str):
        bmsg = Boolconsensus()
        bmsg.k = self.seq[robot_name]
        bmsg.x = self.z[robot_name][0]
        bmsg.y = self.z[robot_name][1]
        bmsg.z = self.z[robot_name][2]
        self.boolcon_pubs[robot_name].publish(bmsg)
        self.talkto_neighbors= self.talkto_neighbors + 1 if self.latest_positive[robot_name] is None else self.talkto_neighbors
    
    
    def check_quiet(self, robot_name: str):
        
        
        now = self.get_clock().now()
        if (now - self.last_update[robot_name]) > Duration(seconds=3.0):
            if not (self.z[robot_name][0] or self.z[robot_name][1] or self.z[robot_name][2]):
                self.latest_positive[robot_name] = None
            
            
            if self.z[robot_name][0] or self.z[robot_name][1] or self.z[robot_name][2]:
                if self.latest_positive[robot_name] is None:
                    now_t = self.get_clock().now().to_msg()
                    t = now_t.sec + now_t.nanosec * 1e-9
                    self.latest_positive[robot_name] = t - self.start_time
            # self.get_logger().info(f'{robot_name} is quiet with {self.z[robot_name]}') 
            consensus_msg = Predictions()
            # get id from robot name string
            id = int(robot_name.split('_')[-1])
            consensus_msg.x = float(id) if self.z[robot_name][0] else float(id - 1)
            consensus_msg.y = float(id) if self.z[robot_name][1] else float(id - 1)
            consensus_msg.z = float(id) if self.z[robot_name][2] else float(id - 1)

            self.consensus_pub[robot_name].publish(consensus_msg)  
        else:
            self.latest_positive[robot_name] = None
            consensus_msg = Predictions()
            # get id from robot name string
            id = int(robot_name.split('_')[-1])
            consensus_msg.x = float(id - 1)
            consensus_msg.y = float(id - 1)
            consensus_msg.z = float(id - 1)

            self.consensus_pub[robot_name].publish(consensus_msg)

        
    def central_debugger(self):
        add = []
        for robot_name in self.robot_names:
            add.append(self.z[robot_name][1])
        # print("Current consensus states:", add)


    def pred_cb(self, msg: Predictions, robot_name: str):
        self.preds[robot_name] = msg

    def esb_cb(self, msg: Predictions, robot_name: str):
        self.esb_state_rec[robot_name] = msg
    
    def esb_algorithm(self, robot_name: str):

        if self.esb_internal_state[robot_name][0] == 0.0 and self.esb_internal_state[robot_name][1] == 0.0 and self.esb_internal_state[robot_name][2] == 0.0:
            self.esb_internal_state[robot_name] = [self.preds[robot_name].x, self.preds[robot_name].y, self.preds[robot_name].z]
        if self.esb_state_set[robot_name] == None:
            self.esb_state_set[robot_name] = Predictions()
            self.esb_state_set[robot_name].x = self.preds[robot_name].x
            self.esb_state_set[robot_name].y = self.preds[robot_name].y
            self.esb_state_set[robot_name].z = self.preds[robot_name].z
        

        dist = {}
        for i in self.robot_names:
            if i != robot_name:
                try:
                    dist[i] = ((self.positions[robot_name][0]-self.positions[i][0])**2 + (self.positions[robot_name][1]-self.positions[i][1])**2)**0.5
                except :
                    self.get_logger().warn(f"Position not found for hero_robot {i}")
                    continue

        top = sorted(dist.items(), key=lambda x: x[1])
        topk = [k for k, v in top if v <= self.radius and self.radius > 0]

        

        s = [0.0, 0.0, 0.0]
        r = [0.0, 0.0, 0.0]

        # --- neighbor disagreement: SUM OF SQUARES ---
        s_s = 0.0
        for j in topk:
            dx = self.esb_state_set[robot_name].x - self.esb_state_rec[j].x
            dy = self.esb_state_set[robot_name].y - self.esb_state_rec[j].y
            dz = self.esb_state_set[robot_name].z - self.esb_state_rec[j].z

            s_s += dx*dx + dy*dy + dz*dz

        # --- local prediction tracking (error) ---
        r[0] = self.preds[robot_name].x - self.esb_state_set[robot_name].x
        r[1] = self.preds[robot_name].y - self.esb_state_set[robot_name].y
        r[2] = self.preds[robot_name].z - self.esb_state_set[robot_name].z

        r_s = float(np.dot(np.asarray(r), np.asarray(r)))

        if r_s >= self.sigma * s_s:
            self.esb_state_set[robot_name].x = self.preds[robot_name].x
            self.esb_state_set[robot_name].y = self.preds[robot_name].y
            self.esb_state_set[robot_name].z = self.preds[robot_name].z
            self.esb_pubs[robot_name].publish(self.preds[robot_name]) 
            self.talkto_neighbors_l1= self.talkto_neighbors_l1 + 1 if self.latest_positive[robot_name] is None else self.talkto_neighbors_l1

        con = [self.esb_state_set[robot_name].x, self.esb_state_set[robot_name].y, self.esb_state_set[robot_name].z]
        preds = [self.preds[robot_name].x, self.preds[robot_name].y, self.preds[robot_name].z]

        self.latest_binary[robot_name] = self.binary_state(con, preds,s_s, robot_name) # replace this with your favorite stage 1 consensus method.
 

    def tf_callback(self, msg: TFMessage):
        for i in msg.transforms:
            child_name = i.child_frame_id
            c = i.transform.translation
            self.positions[TF_TO_ROBOT[child_name]] = [c.x,c.y]
    

def main():
    rclpy.init()
    node = ConsensusESB()  # your node

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
                # Fallback: sync cleanup() (your current method)
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

