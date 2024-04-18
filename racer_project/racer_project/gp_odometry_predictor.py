import torch
import gpytorch
from rclpy.node import Node
import rclpy
import pickle
from nav_msgs.msg import Odometry
from barcgp.prediction.gpytorch_models import ExactGPModel  # Ensure this is defined somewhere in your project

class GPOdometryPredictor(Node):
    def __init__(self):
        super().__init__('gp_odometry_predictor')
        self.subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',  # Adjust the topic as per your setup
            self.listener_callback,
            10)
        self.subscription  # Prevent unused variable warning

        # Initialize the likelihood and GP model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # Dummy training data, adjust as needed
        train_x = torch.linspace(0, 1, 100)
        train_y = torch.zeros(100)  # Placeholder, replace with actual data
        self.model = ExactGPModel(train_x, train_y, likelihood)
        self.model.eval()  # Set the model to evaluation mode if not training

    def listener_callback(self, msg):
        # Example of processing data and using the model for prediction
        x = torch.tensor([msg.pose.pose.position.x])  # Simplified example
        prediction = self.model(x)
        print(f"Prediction mean: {prediction.mean}, Covariance: {prediction.covariance_matrix}")

def main(args=None):
    rclpy.init(args=args)
    node = GPOdometryPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

