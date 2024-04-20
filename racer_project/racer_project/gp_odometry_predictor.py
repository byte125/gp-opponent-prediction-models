import torch
import gpytorch
from rclpy.node import Node
import rclpy
import pickle
from nav_msgs.msg import Odometry
from barcgp.prediction.gpytorch_models import IndependentMultitaskGPModelApproximate  # Ensure this is defined somewhere in your project

class GPOdometryPredictor(Node):
    def __init__(self):
        super().__init__('gp_odometry_predictor')
        self.subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.listener_callback,
            10)

        # Parameters for the model
        inducing_points_num = 200
        input_dim = 3  # Assuming you're using 3D position data (x, y, z)
        num_tasks = 5  # Number of tasks/output dimensions

        # Placeholder for demonstration, adjust as needed
        train_x = torch.rand(10, input_dim)  # Random training data inputs
        train_y = torch.zeros(10, num_tasks)  # Placeholder training data outputs

        # Initialize the likelihood and GP model
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        self.model = IndependentMultitaskGPModelApproximate(inducing_points_num, input_dim, num_tasks)
        self.model.eval()  # Set the model to evaluation mode if not training


    def listener_callback(self, msg):
        # Extract position data from Odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Format the input tensor for the model
        # Note: The input tensor shape should match the expected input dimensions of the model
        input_tensor = torch.tensor([[x, y, z]], dtype=torch.float32)

        # Perform prediction
        with torch.no_grad():  # Use torch.no_grad to prevent gradient calculations
            prediction = self.model(input_tensor)

        # Print or process the prediction
        print(f"Predicted Mean: {prediction.mean}")
        print(f"Predicted Covariance: {prediction.covariance_matrix}")


def main(args=None):
    rclpy.init(args=args)
    node = GPOdometryPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

