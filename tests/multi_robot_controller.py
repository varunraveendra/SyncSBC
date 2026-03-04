#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import ParameterDescriptor, ParameterType as PT
from functools import partial
import subprocess

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

from behavior_hero.known_controllers import BEHAVIORS


class MultiRobotController(Node):
    

    def __init__(self):
        super().__init__('multi_robot_controller')

        # --- parameters ---
        self.declare_parameter(
            'robot_names', ['hero_plus_01'],
            ParameterDescriptor(type=PT.PARAMETER_STRING_ARRAY)
        )
        self.declare_parameter('controller_id', 'C1')

        self.robot_names = [str(x) for x in self.get_parameter('robot_names').value]


        qos = QoSProfile(depth=10)
        self.cmd_vel_pub = {}   # <-- renamed
        self.tof_subs   = []    # <-- renamed
        self.switch_subs = []    # <-- new

        self.linear_true_max = {}
        self.angular_true_max = {}
        self.linear_true_tres = {}
        self.angular_true_tres = {}
        self.linear_false = {}
        self.angular_false = {}
        self.threshold_dist = {}
        self.max_dist = {}

        self.cid = {}




        for name in self.robot_names:
            tof_topic = f'/{name}/tof'
            cmd_topic = f'/{name}/velocity_controller/cmd_vel'
            self.cmd_vel_pub[name] = self.create_publisher(Twist, cmd_topic, qos)
            self.tof_subs.append(
                self.create_subscription(Float32, tof_topic, partial(self.tof_cb, robot_name=name), qos)
            )
            self.cid[name] = self.get_parameter('controller_id').value
            self.initialize_controller(controller_id=self.cid[name], robot_name=name)

            self.get_logger().info(f'[{name}] Sub {tof_topic} → Pub {cmd_topic}')

        if not self.robot_names:
            self.get_logger().warn('No robot_names provided. Set parameter robot_names:=[...]')

    def initialize_controller(self, controller_id='C1', robot_name='hero_plus_01'):
        self.get_logger().info("Initializing Behavior Controller: {} for {}".format(controller_id, robot_name))
        self.linear_true_max[robot_name]   = float(BEHAVIORS[controller_id].vt)
        self.angular_true_max[robot_name]  = float(BEHAVIORS[controller_id].at)
        self.linear_true_tres[robot_name]  = float(BEHAVIORS[controller_id].vt)
        self.angular_true_tres[robot_name] = float(BEHAVIORS[controller_id].at)
        self.linear_false[robot_name]      = float(BEHAVIORS[controller_id].vf)
        self.angular_false[robot_name]     = float(BEHAVIORS[controller_id].af)
        self.threshold_dist[robot_name]    = float(BEHAVIORS[controller_id].th)
        self.max_dist[robot_name]          = float(BEHAVIORS[controller_id].th) * 2.0

    def classify(self, d: float, robot_name: str) -> int:
        if d < self.threshold_dist[robot_name]: 
            return 1
        if d < self.max_dist[robot_name]:       
            return 2
        return 3

    def tof_cb(self, msg: Float32, robot_name: str):
        c = self.classify(float(msg.data), robot_name)
        # self.get_logger().info(f'{c} the tof')
        if   c == 1: self.send_velocity_1(robot_name)
        elif c == 2: self.send_velocity_2(robot_name)
        else:        self.send_velocity_3(robot_name)


    def send_velocity_1(self, robot_name: str):
        cmd = Twist(); cmd.linear.x = self.linear_true_tres[robot_name]; cmd.angular.z = self.angular_true_tres[robot_name]
        self.cmd_vel_pub[robot_name].publish(cmd)

    def send_velocity_2(self, robot_name: str):
        cmd = Twist(); cmd.linear.x = self.linear_true_max[robot_name]; cmd.angular.z = self.angular_true_max[robot_name]
        self.cmd_vel_pub[robot_name].publish(cmd)

    def send_velocity_3(self, robot_name: str):

        cmd = Twist(); cmd.linear.x = self.linear_false[robot_name]; cmd.angular.z = self.angular_false[robot_name]
        self.cmd_vel_pub[robot_name].publish(cmd)
    
    def set_next_behavior(self, robot_name: str, next_control: str='C'):
       if next_control == 'S':
           for name in self.robot_names:
               self.linear_true_max[name]   = 0.0
               self.angular_true_max[name]  = 0.0
               self.linear_true_tres[name]  = 0.0
               self.angular_true_tres[name] = 0.0
               self.linear_false[name]      = 0.0
               self.angular_false[name]     = 0.0
       else:
           self.cid[robot_name] = next_control
           self.initialize_controller(controller_id=next_control, robot_name=robot_name)


def main():
    try:
        script_path = "isaacsimconnector.sh"  # <-- update path if needed
        file_to_run = "initializer.py"  # <-- update path if needed
        subprocess.run(['bash', script_path, file_to_run], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error while running script:", e.stderr)
        return  # Stop here if the script fails


    rclpy.init()
    node = MultiRobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

