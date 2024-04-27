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
        
        self.tarMin = VehicleState(t=0.0,
                            p=ParametricPose(s=offset + 0.9, x_tran=-.3 * width, e_psi=-0.02),
                            v=BodyLinearVelocity(v_long=0.5*factor))
        self.tarMax = VehicleState(t=0.0,
                            p=ParametricPose(s=offset + 1.2, x_tran=.3* width, e_psi=0.02),
                            v=BodyLinearVelocity(v_long=1.0*factor))
        self.egoMin = VehicleState(t=0.0,
                            p=ParametricPose(s=offset + 0.2, x_tran=-.3 * width, e_psi=-0.02),
                            v=BodyLinearVelocity(v_long=0.5*factor))
        self.egoMax = VehicleState(t=0.0,
                            p=ParametricPose(s=offset + 0.4, x_tran=.3 * width, e_psi=0.02),
                            v=BodyLinearVelocity(v_long=1.0*factor))
        
        self.subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.listener_callback,
            10)

        # # Parameters for the model
        # inducing_points_num = 200
        # input_dim = 3  # Assuming you're using 3D position data (x, y, z)
        # num_tasks = 5  # Number of tasks/output dimensions
        ##############################################################################################################################################
        use_GPU = False   
        policy_name = "aggressive_blocking"
        self.M = 50  # Number of samples for GP
        self.T = 20  # Max number of seconds to run experiment
        self.t = 0  # Initial time increment
        self.N = 10  # Number of samples for GP
        ##############################################################################################################################################
        # train_x = torch.rand(10, input_dim)  # Random training data inputs
        # train_y = torch.zeros(10, num_tasks)  # Placeholder training data outputs

        # # Initialize the likelihood and GP model
        # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

        # s, x_tran, e_psi, v_long need to be set to the start values, for both EV and TV
        s = 0.0
        x_tran = 0.0
        e_psi = 0.0
        v_long = 0.0
        self.ego_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))
        
        s = 1.0
        x_tran = 0.0
        e_psi = 0.0
        v_long = 0.0
        self.tar_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))

        # Slack is between 0 and 1, allows for constraint violation (1 is no constraint violation)
        # This generates a scenario with a straight track, which can be passed as track_obj
        # I dont know what phase_out, ego_obs_avoid_d, tar_obs_avoid_d are, so I set them to default values
        self.straight_track = StraightTrack(length=10, width=100.0, slack=0.8, phase_out=True)
        # track_obj = ScenarioDefinition(
        #     track_type='straight',
        #     track=straight_track,
        #     ego_init_state=ego_init_state,
        #     tar_init_state=tar_init_state,
        #     ego_obs_avoid_d=0.1,
        #     tar_obs_avoid_d=0.1
        # )
        
        self.scen_params = ScenarioGenParams(types=['track'], egoMin=self.egoMin, egoMax=self.egoMax, tarMin=self.tarMin, tarMax=self.tarMax, width=100.0)
        self.scen_gen = ScenarioGenerator(self.scen_params)
        self.scenario = self.scen_gen.genScenario()
        
        track_name = self.scenario.track_type
        track_obj = self.scenario.track

        self.ego_dynamics_simulator = DynamicsSimulator(self.t, ego_dynamics_config, track=track_obj)
        self.tar_dynamics_simulator = DynamicsSimulator(self.t, tar_dynamics_config, track=track_obj)

        # scenario = straight_track
        self.tv_history, self.ego_history, self.vehiclestate_history, self.ego_sim_state, self.tar_sim_state, self.egost_list, self.tarst_list = \
        run_pid_warmstart(self.scenario, self.ego_dynamics_simulator, self.tar_dynamics_simulator, n_iter=n_iter, t=self.t)
        
        self.gp_mpcc_ego_params = MPCCApproxFullModelParams(
            dt=dt,
            all_tracks=all_tracks,
            solver_dir='' if rebuild else '~/.mpclab_controllers/gp_mpcc_h2h_ego',
            # solver_dir='',
            optlevel=2,

            N=N,
            Qc=50,
            Ql=500.0,
            Q_theta=200.0,
            Q_xref=0.0,
            R_d=2.0,
            R_delta=20.0,

            slack=True,
            l_cs=5,
            Q_cs=2.0,
            Q_vmax=200.0,
            vlong_max_soft=1.4,
            Q_ts=500.0,
            Q_cs_e=8.0,
            l_cs_e=35.0,

            u_a_max=0.55,
            vx_max=1.6,
            u_a_min=-1,
            u_steer_max=0.435,
            u_steer_min=-0.435,
            u_a_rate_max=10,
            u_a_rate_min=-10,
            u_steer_rate_max=2,
            u_steer_rate_min=-2
        )
        
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.ego_dynamics_simulator.model, track_obj, self.gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name=track_name)
        self.gp_mpcc_ego_controller.initialize()

        self.gp_mpcc_ego_controller.set_warm_start(*self.ego_history)
        
        # We should not be calling get_prediction immediately after this I think, need 
        # to call other functions
        
        self.predictor = GPPredictor(N=10, track=track_obj, policy_name=policy_name, use_GPU=use_GPU, M=self.M, cov_factor=np.sqrt(2))

        self.ego_pred = self.gp_mpcc_ego_controller.get_prediction()

        
    # Function to initialize arrays with a single zero
    def zero_array(self):
        return array.array('f', [0.0])

    def listener_callback(self, msg):
        # Extract position data from Odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # Perform prediction
        
        gp_tarpred_list = [None] * n_iter
        egopred_list = [None] * n_iter
        tarpred_list = [None] * n_iter
        
        ego_prediction, tar_prediction, tv_pred = None, None, None
        while self.t < self.T:
            if self.tar_sim_state.p.s >= 1.9 * self.scenario.length or self.ego_sim_state.p.s >= 1.9 * self.scenario.length:
                break
            else:
                if self.predictor:
                    ego_pred = self.gp_mpcc_ego_controller.get_prediction()
                    print("Ego pred", ego_pred)
                    if ego_pred.s is not None:
                        tv_pred = self.predictor.get_prediction(self.ego_sim_state, self.tar_sim_state, ego_pred)
                        gp_tarpred_list.append(tv_pred.copy())
                    else:
                        gp_tarpred_list.append(None)

                # # Target agent
                # info, b, exitflag = mpcc_tv_controller.step(tar_sim_state, tv_state=ego_sim_state, tv_pred=ego_prediction, policy=policy_name)
                # if not info["success"]:
                #     print(f"TV infeasible - Exitflag: {exitflag}")
                #     pass

                # Ego agent
                info, b, exitflag = self.gp_mpcc_ego_controller.step(self.ego_sim_state, tv_state=self.tar_sim_state, tv_pred=tar_prediction)
                if not info["success"]:
                    print(f"EGO infeasible - Exitflag: {exitflag}")
                    pass
                    # return

                # step forward
                # tar_prediction = mpcc_tv_controller.get_prediction().copy()
                # tar_prediction.t = tar_sim_state.t
                # tar_dynamics_simulator.step(tar_sim_state)
                # track_obj.update_curvature(tar_sim_state)

                ego_prediction = self.gp_mpcc_ego_controller.get_prediction().copy()
                ego_prediction.t = self.ego_sim_state.t
                self.ego_dynamics_simulator.step(self.ego_sim_state)

                # log states
                self.egost_list.append(self.ego_sim_state.copy())
                self.tarst_list.append(self.tar_sim_state.copy())
                egopred_list.append(ego_prediction)
                tarpred_list.append(tar_prediction)
                print(f"Current time: {round(self.ego_sim_state.t, 2)}")
        # with torch.no_grad():  # Use torch.no_grad to prevent gradient calculations
        #     prediction = self.model(input_tensor)

        # # Print or process the prediction
        # print(f"Predicted Mean: {prediction.mean}")
        # print(f"Predicted Covariance: {prediction.covariance_matrix}")


def main(args=None):
    rclpy.init(args=args)
    node = GPOdometryPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

