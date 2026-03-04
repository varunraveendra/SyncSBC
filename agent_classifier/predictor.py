#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String, ColorRGBA
from hero_common.msg import Predictions as predictions  # keep name used below
import tensorflow as tf
from rcl_interfaces.msg import ParameterDescriptor, ParameterType as PT
import numpy as np
from functools import partial

class KerasPredictionNode(Node):
    def __init__(self, model_path, buffer_size):
        super().__init__('behavior_rec')

        # Load the Keras model
        self.model = tf.keras.models.load_model(model_path)
        self.get_logger().info(f"Model loaded successfully from: {model_path}")

        # Parameters
        self.declare_parameter(
            'robot_names',
            ['hero_plus_01'],
            ParameterDescriptor(type=PT.PARAMETER_STRING_ARRAY)
        )

        # ROS I/O
        self.subscribers = []
        self.pred_publishers = {}
        self.led_publishers = {}
        self.robot_names = [str(x) for x in self.get_parameter('robot_names').value]

        # State
        self.data_buffer = {}
        self.x = {}
        self.y = {}
        self.z = {}
        self.n = {}
        self.time_step = {}
        self.n_values = np.linspace(0.05, 0.4, 151)

        # QoS (depth similar to ROS1 queue_size=10)
        qos_depth = 10

        for robot_name in self.robot_names:
            topic_name = f"/{robot_name}/tof"
            topic_led = f"/{robot_name}/led"

            # Subscribers (use partial to pass robot_name like in ROS1)
            sub = self.create_subscription(
                Float32,
                topic_name,
                partial(self.input_callback, robot_name=robot_name),
                qos_depth
            )
            self.subscribers.append(sub)

            # Publishers
            self.pred_publishers[robot_name] = self.create_publisher(
                predictions, f"/prediction/{robot_name}", qos_depth
            )
            self.led_publishers[robot_name] = self.create_publisher(
                ColorRGBA, topic_led, qos_depth
            )

            # Per-robot buffers/state
            self.data_buffer[robot_name] = []
            self.x[robot_name] = 0.0
            self.y[robot_name] = 0.0
            self.z[robot_name] = 0.0
            self.n[robot_name] = 0.05
            self.time_step[robot_name] = 0

        # Buffer length
        self.buffer_size = buffer_size

    def input_callback(self, msg: Float32, robot_name: str):
        """Buffer incoming data and run prediction when window is full."""
        try:
            self.data_buffer[robot_name].append(msg.data)

            # Trim to last `buffer_size` (your ROS1 code popped ~30; preserving behavior of keeping a sliding window)
            if len(self.data_buffer[robot_name]) > self.buffer_size:
                self.data_buffer[robot_name] = self.data_buffer[robot_name][30:]

            if len(self.data_buffer[robot_name]) == self.buffer_size:
                self.run_prediction(robot_name)
        except Exception as e:
            self.get_logger().error(f"Error buffering data: {e}")

    def run_prediction(self, robot_name: str):
        """Run model inference and post-process."""
        try:
            input_data = np.array(self.data_buffer[robot_name]).reshape(1, self.buffer_size, 1)
            prediction = self.model.predict(input_data, verbose=0)
            self.process_prediction(prediction, robot_name)
            self.led_call(robot_name)
        except Exception as e:
            self.get_logger().error(f"Error during prediction: {e}")

    def process_prediction(self, prediction, robot_name: str):
        """EMA-like smoothing + publish."""
        # Keep your time-stepped n schedule
        if self.time_step[robot_name] < 150:
            self.get_logger().info(f"{self.n[robot_name]}")
            self.n[robot_name] = self.n_values[self.time_step[robot_name]]
        elif self.time_step[robot_name] < 250:
            self.time_step[robot_name] = 0

        alpha = float(self.n[robot_name])
        self.x[robot_name] += alpha * (float(prediction[0][0]) - self.x[robot_name])
        self.y[robot_name] += alpha * (float(prediction[0][1]) - self.y[robot_name])
        self.z[robot_name] += alpha * (float(prediction[0][2]) - self.z[robot_name])
        self.time_step[robot_name] += 1

        r = predictions()
        r.x = self.x[robot_name]
        r.y = self.y[robot_name]
        r.z = self.z[robot_name]
        self.pred_publishers[robot_name].publish(r)

    def led_call(self, robot_name: str):
        """Publish LED color from x,y,z."""
        color = ColorRGBA()
        color.r = float(self.x[robot_name])
        color.g = float(self.y[robot_name])
        color.b = float(self.z[robot_name])
        color.a = 1.0
        self.led_publishers[robot_name].publish(color)


def main():
    model_path = "TCN.keras"
    buffer_size = 600

    rclpy.init()
    node = KerasPredictionNode(model_path, buffer_size)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()