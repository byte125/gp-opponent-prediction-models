import rclpy
from rclpy.node import Node
import torch
from example_gp_ros.simple_gp_model import SimpleGPModel, likelihood

class GPTrajectoryNode(Node):
    def __init__(self):
        super().__init__('gp_trajectory_node')
        self.publisher_ = self.create_publisher(Float64, 'predicted_trajectory', 10)
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Load the model
        self.model = SimpleGPModel(train_x=torch.linspace(0, 1, 100), train_y=None, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.model.load_state_dict(torch.load('model_state.pth'))
        self.model.eval()

    def timer_callback(self):
        test_x = torch.tensor([0.5])  # Example input to the model
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predicted_trajectory = self.model(test_x).mean.item()
        self.publisher_.publish(Float64(data=predicted_trajectory))

def main(args=None):
    rclpy.init(args=args)
    node = GPTrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()