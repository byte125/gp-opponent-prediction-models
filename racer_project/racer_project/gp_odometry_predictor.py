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
from barcgp.common.utils.scenario_utils import ScenarioDefinition, ScenarioGenerator, ScenarioGenParams
from barcgp.common.tracks.track_lib import StraightTrack
from barcgp.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehiclePrediction
from barcgp.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from barcgp.simulation.dynamics_simulator import DynamicsSimulator
from barcgp.h2h_configs import *
from barcgp.common.utils.file_utils import *
from barcgp.common_control import run_pid_warmstart

class GPOdometryPredictor(Node):
    def __init__(self):
        super().__init__('gp_odometry_predictor')
        
        tarMin = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 0.9, x_tran=-.3 * width, e_psi=-0.02),
                      v=BodyLinearVelocity(v_long=0.5*factor))
        tarMax = VehicleState(t=0.0,
                            p=ParametricPose(s=offset + 1.2, x_tran=.3* width, e_psi=0.02),
                            v=BodyLinearVelocity(v_long=1.0*factor))
        egoMin = VehicleState(t=0.0,
                            p=ParametricPose(s=offset + 0.2, x_tran=-.3 * width, e_psi=-0.02),
                            v=BodyLinearVelocity(v_long=0.5*factor))
        egoMax = VehicleState(t=0.0,
                            p=ParametricPose(s=offset + 0.4, x_tran=.3 * width, e_psi=0.02),
                            v=BodyLinearVelocity(v_long=1.0*factor))
        
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
        straight_track = StraightTrack(length=10, width=100.0, slack=0.8, phase_out=True)
        # track_obj = ScenarioDefinition(
        #     track_type='straight',
        #     track=straight_track,
        #     ego_init_state=ego_init_state,
        #     tar_init_state=tar_init_state,
        #     ego_obs_avoid_d=0.1,
        #     tar_obs_avoid_d=0.1
        # )
        
        scen_params = ScenarioGenParams(types=['track'], egoMin=egoMin, egoMax=egoMax, tarMin=tarMin, tarMax=tarMax, width=100.0)
        scen_gen = ScenarioGenerator(scen_params)
        scenario = scen_gen.genScenario()
        track_name = scenario.track_type
        track_obj = scenario.track
        
        # scen = track_obj.track
        
        # scenario_sim_data = pickle_read('/home/sd/barc_data/testcurve.pkl')
        # scenario = scenario_sim_data.scenario_def

        # track_name = scenario.track_type
        # track_obj = scenario.track

        ego_dynamics_simulator = DynamicsSimulator(t, ego_dynamics_config, track=straight_track)
        tar_dynamics_simulator = DynamicsSimulator(t, tar_dynamics_config, track=straight_track)

        # scenario = straight_track
        tv_history, ego_history, vehiclestate_history, ego_sim_state, tar_sim_state, egost_list, tarst_list = \
        run_pid_warmstart(scenario, ego_dynamics_simulator, tar_dynamics_simulator, n_iter=n_iter, t=t)
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name=track_name)
        self.gp_mpcc_ego_controller.initialize()

        self.gp_mpcc_ego_controller.set_warm_start(*ego_history)
        self.ego_pred = self.gp_mpcc_ego_controller.get_prediction()

        # TODO:Have to fill ego dynamics model and params
        # self.gp_mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj.track, gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name=track_obj.track_type)
        # self.gp_mpcc_ego_controller.initialize()

        self.predictor = GPPredictor(N=N, track=track_obj, policy_name=policy_name, use_GPU=use_GPU, M=M, cov_factor=np.sqrt(2))
        # self.model = IndependentMultitaskGPModelApproximate(inducing_points_num, input_dim, num_tasks)
        # self.model.eval()  # Set the model to evaluation mode if not training

    # Function to initialize arrays with a single zero
    def zero_array(self):
        return array.array('f', [0.0])

    def listener_callback(self, msg):
        # Extract position data from Odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Format the input tensor for the model
        # Note: The input tensor shape should match the expected input dimensions of the model
        input_tensor = torch.tensor([[x, y, z]], dtype=torch.float32)
        
        # TODO - Yash and Jai to figure out what inputs to give to the model
        # s, x_tran, e_psi and v_long
        
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
        # ego_pred = VehiclePrediction()
        # # Set all attributes to zero
        # ego_pred.t = 0.0
        # ego_pred.x = self.zero_array()
        # ego_pred.y = self.zero_array()
        # ego_pred.v_x = self.zero_array()
        # ego_pred.v_y = self.zero_array()
        # ego_pred.a_x = self.zero_array()
        # ego_pred.a_y = self.zero_array()
        # ego_pred.psi = self.zero_array()
        # ego_pred.psidot = self.zero_array()
        # ego_pred.v_long = self.zero_array()
        # ego_pred.v_tran = self.zero_array()
        # ego_pred.a_long = self.zero_array()
        # ego_pred.a_tran = self.zero_array()
        # ego_pred.e_psi = self.zero_array()
        # ego_pred.s = self.zero_array()
        # ego_pred.x_tran = self.zero_array()
        # ego_pred.u_a = self.zero_array()
        # ego_pred.u_steer = self.zero_array()
        # ego_pred.lap_num = 0
        # ego_pred.sey_cov = np.zeros((1, 1))  # Adjust the dimensions as needed
        # ego_pred.xy_cov = np.zeros((1, 1))  # Adjust the dimensions as needed

# You can now use vehicle_prediction with all values initialized to zero

        tv_pred = self.predictor.get_prediction(ego_state, tar_sim_state, self.ego_pred)
        print (tv_pred)
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

