import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import pickle
import numpy as np
from nav_msgs.msg import Odometry

class GPOdometryPredictor(Node):
    def __init__(self):
        super().__init__('gp_odometry_predictor')
        self.subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',  # Topic name might need adjustment based on your setup
            self.listener_callback,
            10)
        # self.subscription  # prevent unused variable warning
        
        # Load the Gaussian Process model
        with open('/home/sd/barc_data/trainingData/models/aggressive_blocking.pkl', 'rb') as f:
            self.gp_model = pickle.load(f)

    def listener_callback(self, msg):
        self.get_logger().info('YAYYAYAYAY')
        # Extract pose data (or whatever data you need for your model)
        pose = msg.pose.pose.position
        x = pose.x
        y = pose.y
        z = pose.z

        # Create an input array (ensure it matches your model's expected input)
        input_features = np.array([[x, y, z]])
        
        # Make prediction
        prediction = self.gp_model.predict(input_features)
        
        # Print prediction
        self.get_logger().info('Predicted Output: %s' % prediction)

def main(args=None):
    rclpy.init(args=args)
    node = GPOdometryPredictor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
