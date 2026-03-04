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

def _build_robot_colors(names, cmap='tab20'):
    cm = plt.get_cmap(cmap)  # 20 distinct colors
    colors = {}
    for rn in names:
        # stable index from name (avoid Python's randomized hash)
        idx = int(hashlib.md5(rn.encode()).hexdigest()[:8], 16)
        colors[rn] = cm(idx % cm.N)
    return colors



class Consensusnv(Node):
    def __init__(self):
        super().__init__('consensus_nv')

        # --- parameters ---
        self.declare_parameter(
            'robot_names', ['hero_plus_01'],
            ParameterDescriptor(type=PT.PARAMETER_STRING_ARRAY)
        )
        self.declare_parameter('plot_dir', str(pathlib.Path.home() / 'ros_logs'))
        self.declare_parameter('behavior', 'C1')
        self.declare_parameter('basename', 'metricsnv' + self.get_parameter('behavior').get_parameter_value().string_value)
        self.declare_parameter( "run_id", -1, ParameterDescriptor(type=PT.PARAMETER_INTEGER) )
        self.declare_parameter(
            "eps", 0.1,
            ParameterDescriptor(type=PT.PARAMETER_DOUBLE)
        )

        self.declare_parameter(
            "mar", 0.0,
            ParameterDescriptor(type=PT.PARAMETER_DOUBLE)
        )
        self.declare_parameter(
            "hys", 5.0,
            ParameterDescriptor(type=PT.PARAMETER_DOUBLE)
        )
        self.declare_parameter(
            "eps_h", 0.2,
            ParameterDescriptor(type=PT.PARAMETER_DOUBLE)
        )
        self.declare_parameter( "radius", 0.01, ParameterDescriptor(type=PT.PARAMETER_DOUBLE) )
        self.radius = self.get_parameter("radius").value

         # Read parameters

        self.eps = self.get_parameter('eps').value
        # self.eps_h = self.get_parameter('eps_h').value 
        
        self.hyst_max = int(self.get_parameter('hys').value)

         # Prepare plot directory
        self.plot_dir = pathlib.Path(self.get_parameter('plot_dir').get_parameter_value().string_value)
        self.basename = str(self.get_parameter('run_id').value) + "_" + self.get_parameter('basename').get_parameter_value().string_value
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
        self.nvon_pubs = {}
        self.last_consensus = {rn: None for rn in self.robot_names}
        self.positions = {}
        self.z = {}
        self.last_update = {}
        self.preds = {}
        self.hyst = {}
        #self.eps= 0.08
        self.eps_h= 0.15
        #self.hyst_max= 7
        self.latest_binary = {}
        self.defaulter_count = {}
        self.latest_positive = {}
        self.clock= Clock(clock_type=ClockType.ROS_TIME)
        now_t = self.get_clock().now().to_msg()
        t = now_t.sec + now_t.nanosec * 1e-9
        self.start_time = t
        self.total_binaries= [[],[],[]]
        self.total_binaries_t= []

        self.talkto_neighbors_l1= 0
        self.last_changes_swarm = self.get_clock().now()

        self.create_timer(0.5, self.binary_logger)

        for name in self.robot_names:
            pred_topic = f'/prediction/{name}'
            self.positions[name] = []
            self.defaulter_count[name] = 0
            self.latest_positive[name] = 0.0
            self.z[name] = [False, False, False]
            self.last_update[name] = self.get_clock().now()
            self.latest_binary[name] = [False, False, False]
            self.hyst[name] = {"ok_count_x":0, "ok_count_y":0, "ok_count_z":0, "done_x":False, "done_y":False, "done_z":False}
            temp = Predictions()
            self.preds[name] = temp
            self.create_timer(0.8, partial(self.roll_epoch, robot_name=name))
            self.create_timer(0.5, partial(self.consensus_variance, robot_name=name))
            self.nvon_pubs[name] = self.create_publisher(Boolconsensus, f'/nvonsensus/{name}', qos)
            self.pred_subs.append(
                self.create_subscription(Predictions, pred_topic, partial(self.pred_cb, robot_name=name), qos)
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
        # print(f"Total binaries logged: {self.total_binaries[-1]} at time {self.total_binaries_t[-1]}")
    
    
    def binary_state(self, con: list, preds: list, robot_name: str) -> bool:
        # self.get_logger().info(f"{con[0]} {preds[0]} {robot_name}")
        h = self.hyst[robot_name]

        # ensure keys exist (in case of first call)
        for k in ["ok_count_x","ok_count_y","ok_count_z","done_x","done_y","done_z"]:
            h.setdefault(k, 0 if "ok_count" in k else False)

        # --- pick winner (argmax with optional margin) ---
        # argmax index
        max_idx = max(range(len(preds)), key=lambda i: preds[i])

        
        # --- reset non-selected counters every step (strict exclusivity) ---
        if max_idx != 0:
            h["ok_count_x"] = 0
        if max_idx != 1:
            h["ok_count_y"] = 0
        if max_idx != 2:
            h["ok_count_z"] = 0

        # --- update the selected class only ---
        if max_idx == 0:
            # grow or reset based on confidence gates
            if con[0] <= self.eps:
                h["ok_count_x"] += 1
            elif con[0] > self.eps_h:
                h["ok_count_x"] = 0
            # compute done flags (exclusive)
            h["done_x"] = h["ok_count_x"] >= self.hyst_max
            h["done_y"] = False
            h["done_z"] = False

            # self.get_logger().info(f"{h['done_x']} {h['ok_count_x']} {robot_name} binstate[x]")
        elif max_idx == 1:
            if con[1] <= self.eps:
                h["ok_count_y"] += 1
            elif con[1] > self.eps_h:
                h["ok_count_y"] = 0

            h["done_y"] = h["ok_count_y"] >= self.hyst_max
            h["done_x"] = False
            h["done_z"] = False

            # self.get_logger().info(f"{h['done_y']} {h['ok_count_y']} {robot_name} binstate[y]")
        else:  # max_idx == 2
            if con[2] <= self.eps:
                h["ok_count_z"] += 1
            elif con[2] > self.eps_h:
                h["ok_count_z"] = 0

            h["done_z"] = h["ok_count_z"] >= self.hyst_max
            h["done_x"] = False
            h["done_y"] = False

            self.hyst[robot_name] = h

        return [h["done_x"], h["done_y"], h["done_z"]]
    

    def cleanup(self):
        self.get_logger().info("Cleaning up...")
        self.get_logger().info(f"Last changes NV: {self.last_changes_swarm}")
        
        
        self.get_logger().info(f"Talked to L1 neighbors NV: {self.talkto_neighbors_l1}")
        """Save two plots + CSVs on shutdown."""
        try:
            stamp = self.get_clock().now().to_msg()
            tag = f"{stamp.sec}_{stamp.nanosec:09d}"

            #================  Total Binaries over Time ================
            
            
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


    def roll_epoch(self, robot_name: str):
        self.z[robot_name] = list(self.latest_binary[robot_name])
        bmsg = Boolconsensus()
        bmsg.x = self.z[robot_name][0]
        bmsg.y = self.z[robot_name][1]
        bmsg.z = self.z[robot_name][2]
        self.nvon_pubs[robot_name].publish(bmsg)
        

    def pred_cb(self, msg: Predictions, robot_name: str):
        self.preds[robot_name] = msg
        
    def consensus_variance(self,robot_name: str):    
        
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

        xs , ys, zs = [], [], []    

        for j in topk:
            # measure disagreement through variance in predictions
            xs.append(self.preds[robot_name].x - self.preds[j].x)    
            ys.append(self.preds[robot_name].y - self.preds[j].y)
            zs.append(self.preds[robot_name].z - self.preds[j].z)

        dx = np.asarray(xs)
        dy = np.asarray(ys)
        dz = np.asarray(zs)

        # Per-axis variances (mean of squared diffs)
        sigma2_x = float(np.mean(dx * dx))
        sigma2_y = float(np.mean(dy * dy))
        sigma2_z = float(np.mean(dz * dz))

        con = [sigma2_x, sigma2_y, sigma2_z]
        preds = [self.preds[robot_name].x, self.preds[robot_name].y, self.preds[robot_name].z]

        self.latest_binary[robot_name] = self.binary_state(con, preds, robot_name)
        self.talkto_neighbors_l1= self.talkto_neighbors_l1 + 1 if self.latest_positive[robot_name] is None else self.talkto_neighbors_l1

        self.last_consensus[robot_name] = [sigma2_x, sigma2_y, sigma2_z]
        vals=[self.last_consensus[r] for r in self.robot_names if self.last_consensus[r] is not None]
        if vals:
            xs, ys, zs = zip(*vals)
            sigma2_x = float(np.mean(xs))
            sigma2_y = float(np.mean(ys))
            sigma2_z = float(np.mean(zs))
            self.for_avg_con_x.append(sigma2_x)
            self.for_avg_con_y.append(sigma2_y)
            self.for_avg_con_z.append(sigma2_z)
            self.for_avg_con_t.append(self.get_clock().now().to_msg().sec)


    def tf_callback(self, msg: TFMessage):
        for i in msg.transforms:
            child_name = i.child_frame_id
            c = i.transform.translation
            self.positions[TF_TO_ROBOT[child_name]] = [c.x,c.y]
    
def main():
    rclpy.init()
    node = Consensusnv()  # your node

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


