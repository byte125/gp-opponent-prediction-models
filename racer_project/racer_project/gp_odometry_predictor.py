import torch
import gpytorch
from rclpy.node import Node
import rclpy
# import pickle
import os
import numpy as np
from nav_msgs.msg import Odometry
from barcgp.prediction.gpytorch_models import IndependentMultitaskGPModelApproximate  # Ensure this is defined somewhere in your project
from barcgp.prediction.trajectory_predictor import GPPredictor  # Ensure this is defined somewhere in your project
from barcgp.common.utils.scenario_utils import ScenarioDefinition
from barcgp.common.tracks.track_lib import StraightTrack
from barcgp.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehiclePrediction
from barcgp.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from barcgp.simulation.dynamics_simulator import DynamicsSimulator
from barcgp.h2h_configs import *
from barcgp.common.utils.file_utils import *

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
        ##############################################################################################################################################
        use_GPU = False   
        policy_name = "aggressive_blocking"
        M = 50  # Number of samples for GP
        T = 20  # Max number of seconds to run experiment
        t = 0  # Initial time increment
        N = 10  # Number of samples for GP
        ##############################################################################################################################################

        # Placeholder for demonstration, adjust as needed
        train_x = torch.rand(10, input_dim)  # Random training data inputs
        train_y = torch.zeros(10, num_tasks)  # Placeholder training data outputs

        # Initialize the likelihood and GP model
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

        # s, x_tran, e_psi, v_long need to be set to the start values, for both EV and TV
        s = 0.0
        x_tran = 0.0
        e_psi = 0.0
        v_long = 0.0
        ego_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))
        
        s = 1.0
        x_tran = 0.0
        e_psi = 0.0
        v_long = 0.0
        tar_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))

        # Slack is between 0 and 1, allows for constraint violation (1 is no constraint violation)
        # This generates a scenario with a straight track, which can be passed as track_obj
        # I dont know what phase_out, ego_obs_avoid_d, tar_obs_avoid_d are, so I set them to default values
        track_obj = ScenarioDefinition(
            track_type='straight',
            track=StraightTrack(length=10, width=100.0, slack=0.8, phase_out=True),
            ego_init_state=ego_init_state,
            tar_init_state=tar_init_state,
            ego_obs_avoid_d=0.1,
            tar_obs_avoid_d=0.1
        )
        
        scenario_sim_data = pickle_read('/home/sd/barc_data/testcurve.pkl')
        scenario = scenario_sim_data.scenario_def

        track_name = scenario.track_type
        track_obj = scenario.track

        ego_dynamics_simulator = DynamicsSimulator(t, ego_dynamics_config, track=track_obj)
        tar_dynamics_simulator = DynamicsSimulator(t, tar_dynamics_config, track=track_obj)

        # TODO:Have to fill ego dynamics model and params
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name=track_name)
        self.gp_mpcc_ego_controller.initialize()

        self.predictor = GPPredictor(N=N, track=track_obj, policy_name=policy_name, use_GPU=use_GPU, M=M, cov_factor=np.sqrt(2))
        # self.model = IndependentMultitaskGPModelApproximate(inducing_points_num, input_dim, num_tasks)
        # self.model.eval()  # Set the model to evaluation mode if not training


    def listener_callback(self, msg):
        # Extract position data from Odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Format the input tensor for the model
        # Note: The input tensor shape should match the expected input dimensions of the model
        input_tensor = torch.tensor([[x, y, z]], dtype=torch.float32)
        
        # Fill this with ego vehicle state data
        s = 0.0
        x_tran = 0.0
        e_psi = 0.0
        v_long = 0.0
        time_elapsed = 0.0
        ego_state = VehicleState(t=time_elapsed,
                                 p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                 v=BodyLinearVelocity(v_long=v_long))
        
        # Fill this with target vehicle state data
        s = 1.0
        x_tran = 0.0
        e_psi = 0.0
        v_long = 0.0
        time_elapsed = 0.0
        tar_sim_state = VehicleState(t=time_elapsed,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))

        # Get the ego vehicle's predicted path using MPCC
        # TODO: Fill the ego_pred object with the MPCC prediction
        ego_pred = self.gp_mpcc_ego_controller.get_prediction()
        tv_pred = self.predictor.get_prediction(ego_state, tar_sim_state, ego_pred)
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

