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
from behavior_hero.constants import TF_TO_ROBOT, NUM_NEIGHBORS, IN_CIRCLE
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


# puresh parameters
tau_on = 0.5
tau_off = 0.3
mu_on = 0.25
rng = np.random.default_rng(42)

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



class ConsensusPureSH(Node):
    def __init__(self):
        super().__init__('consensus_puresh')

        # --- parameters ---
        self.declare_parameter(
            'robot_names', ['hero_plus_01'],
            ParameterDescriptor(type=PT.PARAMETER_STRING_ARRAY)
        )
        self.declare_parameter('plot_dir', str(pathlib.Path.home() / 'ros_logs'))
        self.declare_parameter('behavior', 'C1')
        self.declare_parameter('basename', 'metrics_puresh_' + self.get_parameter('behavior').get_parameter_value().string_value)
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
        self.boolcon_pubs = {}
        self.positions = {}
        self.z = {}
        self.last_update = {}
        self.preds = {}
        self.hyst = {}
        self.hyst_max= 7
        self.latest_binary = {}
        
        self.latest_positive = {}
        self.clock= Clock(clock_type=ClockType.ROS_TIME)
        now_t = self.get_clock().now().to_msg()
        t = now_t.sec + now_t.nanosec * 1e-9
        self.start_time = t
        self.total_binaries= [[],[],[]]
        self.total_binaries_t= []
        self.last_changes_swarm = self.get_clock().now()
        self.influenced_count= 0
        self.talkto_neighbors_l1= 0

        self.create_timer(0.5, self.binary_logger)
        self.psh_subs   = []
        self.psh_pubs = {}
        self.psh_state_set = {}
        self.psh_state_rec = {}
        self.psh_internal_state = {}
        self.prev_binary = {name: [False, False, False] for name in self.robot_names}
        self.latest_step_time = {name: [None, None, None] for name in self.robot_names}

        for name in self.robot_names:
            pred_topic = f'/prediction/{name}'
            self.positions[name] = []
            self.psh_state_set[name] = None
            self.psh_state_rec[name] = Predictions()
            self.latest_positive[name] = 0.0

            
            self.psh_internal_state[name] = [0,0,0.0,0.0]
            self.z[name] = [False, False, False]
            self.last_update[name] = self.get_clock().now()
            self.latest_binary[name] = [False, False, False]
            self.hyst[name] = {"ok_count_x":0, "ok_count_y":0, "ok_count_z":0, "done_x":False, "done_y":False, "done_z":False}
            temp = Predictions()
            self.preds[name] = temp
            
            
            self.create_timer(0.8, partial(self.roll_epoch, robot_name=name))
            self.create_timer(0.5, partial(self.puresh_algorithm, robot_name=name))
            self.boolcon_pubs[name] = self.create_publisher(Boolconsensus, f'/pureshconsensus/{name}', qos)
            self.psh_pubs[name] = self.create_publisher(Predictions, f'/puresh/{name}', qos)
            self.pred_subs.append(
                self.create_subscription(Predictions, pred_topic, partial(self.pred_cb, robot_name=name), qos)
            )
            self.psh_subs.append(
                self.create_subscription(Predictions, f'/puresh/{name}', partial(self.psh_cb, robot_name=name), qos)
            )
        if not self.robot_names:
            self.get_logger().warn('No robot_names provided. Set parameter robot_names:=[...]')

    def binary_logger(self):

        # --- time ---
        now_t = self.get_clock().now().to_msg()
        t = now_t.sec + now_t.nanosec * 1e-9

        # --- detect False -> True transitions ---
        for r in self.robot_names:
            cur = self.latest_binary[r]
            prev = self.prev_binary[r]

            for k in range(3):
                if (not prev[k]) and cur[k]:
                    self.latest_step_time[r][k] = t  # overwrite on new rise

            self.prev_binary[r] = cur.copy()

        # --- compute normalized totals ---
        sum_x = sum(1.0 if self.latest_binary[i][0] else 0.0 for i in self.robot_names)
        sum_y = sum(1.0 if self.latest_binary[i][1] else 0.0 for i in self.robot_names)
        sum_z = sum(1.0 if self.latest_binary[i][2] else 0.0 for i in self.robot_names)

        N = float(len(self.robot_names))
        frac_x = sum_x / N
        frac_y = sum_y / N
        frac_z = sum_z / N

        self.total_binaries[0].append(frac_x)
        self.total_binaries[1].append(frac_y)
        self.total_binaries[2].append(frac_z)

        # --- if any channel reaches full agreement ---
        totals = [frac_x, frac_y, frac_z]

        for k in range(3):
            if totals[k] == 1.0:
                # clear step times of all OTHER channels
                for r in self.robot_names:
                    for j in range(3):
                        if j != k:
                            self.latest_step_time[r][j] = None
        
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
        self.get_logger().info(f"Last changes psh: {self.last_changes_swarm}")
        self.get_logger().info(f"Influenced count psh: {self.influenced_count}")
        self.get_logger().info(f"Talked to L1 neighbors psh: {self.talkto_neighbors_l1}")
        """Save two plots + CSVs on shutdown."""
        try:
            stamp = self.get_clock().now().to_msg()
            tag = f"{stamp.sec}_{stamp.nanosec:09d}"
            
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

    def roll_epoch(self, robot_name: str):
        self.z[robot_name] = list(self.latest_binary[robot_name])
        # print(f'{robot_name} rolling epoch to {self.seq[robot_name]} with {self.z[robot_name]}')
        bmsg = Boolconsensus()        
        bmsg.x = self.z[robot_name][0]
        bmsg.y = self.z[robot_name][1]
        bmsg.z = self.z[robot_name][2]
        self.boolcon_pubs[robot_name].publish(bmsg)
        
    def pred_cb(self, msg: Predictions, robot_name: str):
        self.preds[robot_name] = msg

    def psh_cb(self, msg: Predictions, robot_name: str):
        self.psh_state_rec[robot_name] = msg
    
    def puresh_algorithm(self, robot_name: str):

        if self.psh_internal_state[robot_name][0] == 0.0 and self.psh_internal_state[robot_name][1] == 0.0 and self.psh_internal_state[robot_name][2] == 0.0:
            self.psh_internal_state[robot_name] = [self.preds[robot_name].x, self.preds[robot_name].y, self.preds[robot_name].z]
        if self.psh_state_set[robot_name] == None:
            self.psh_state_set[robot_name] = Predictions()
            self.psh_state_set[robot_name].x = self.preds[robot_name].x
            self.psh_state_set[robot_name].y = self.preds[robot_name].y
            self.psh_state_set[robot_name].z = self.preds[robot_name].z      

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
        if topk:  # non-empty list
            drop_frac = rng.choice([0.0,0.1, 0.2, 0.4])
            keep_n = max(1, int(round(len(topk) * (1.0 - drop_frac))))
            topk = rng.choice(topk, size=keep_n, replace=False).tolist()
        else:
            topk = [] 

        s = [0.0, 0.0, 0.0]
        r = [0.0, 0.0, 0.0]

        # --- neighbor disagreement: SUM OF SQUARES ---
        s_s = 0.0
        for j in topk:
            dx = self.psh_state_set[robot_name].x - self.psh_state_rec[j].x
            dy = self.psh_state_set[robot_name].y - self.psh_state_rec[j].y
            dz = self.psh_state_set[robot_name].z - self.psh_state_rec[j].z

            s_s += dx*dx + dy*dy + dz*dz

        # --- local prediction tracking (error) ---
        r[0] = self.preds[robot_name].x - self.psh_state_set[robot_name].x
        r[1] = self.preds[robot_name].y - self.psh_state_set[robot_name].y
        r[2] = self.preds[robot_name].z - self.psh_state_set[robot_name].z

        r_s = float(np.dot(np.asarray(r), np.asarray(r)))

        if r_s >= self.sigma * s_s :
            self.psh_state_set[robot_name].x = self.preds[robot_name].x
            self.psh_state_set[robot_name].y = self.preds[robot_name].y
            self.psh_state_set[robot_name].z = self.preds[robot_name].z
        self.psh_pubs[robot_name].publish(self.psh_state_set[robot_name]) 
        self.talkto_neighbors_l1= self.talkto_neighbors_l1 + 1 if self.latest_positive[robot_name] is None else self.talkto_neighbors_l1


        con = [self.psh_state_set[robot_name].x, self.psh_state_set[robot_name].y, self.psh_state_set[robot_name].z]
        preds = [self.preds[robot_name].x, self.preds[robot_name].y, self.preds[robot_name].z]

        self.latest_binary[robot_name] = self.binary_state(con, preds,s_s, robot_name)



    def tf_callback(self, msg: TFMessage):
        for i in msg.transforms:
            child_name = i.child_frame_id
            c = i.transform.translation
            self.positions[TF_TO_ROBOT[child_name]] = [c.x,c.y]
    
def _vec3(v, name):
    a = np.asarray(v, dtype=float).flatten()
    if a.size > 3:
        a = a[:3]                               # trim extras
    if a.size < 3:
        raise ValueError(f"{name} has {a.size} elems; need 3")
    return a


def main():
    rclpy.init()
    node = ConsensusPureSH()  # your node

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