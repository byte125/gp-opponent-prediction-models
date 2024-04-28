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
from barcgp.common.utils.scenario_utils import *
from barcgp.common.utils.file_utils import *
from barcgp.common_control import run_pid_warmstart

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class GPOdometryPredictor(Node):
    def __init__(self):
        super().__init__('gp_odometry_predictor')
        
        # TODO - All of the parameters need not be "self", can make some of them
        # local once we have everything working
        
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
        
        self.subscription_ego = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.ego_listener_callback,
            10)
        
        self.pub_drive_ego = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pub_drive_tar = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)
        
        self.subscription_opp = self.create_subscription(
            Odometry,
            '/ego_racecar/opp_odom',
            self.opp_listener_callback,
            10)

        # # Parameters for the model
        # inducing_points_num = 200
        # input_dim = 3  # Assuming you're using 3D position data (x, y, z)
        # num_tasks = 5  # Number of tasks/output dimensions
        ##############################################################################################################################################
        use_GPU = False   
        self.policy_name = "aggressive_blocking"
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
        s, x_tran, e_psi, v_long = self.random_init(self.egoMin, self.egoMax)
        self.ego_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))
        
        s, x_tran, e_psi, v_long = self.random_init(self.egoMin, self.egoMax)
        self.tar_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))

        # Slack is between 0 and 1, allows for constraint violation (1 is no constraint violation)
        # This generates a scenario with a straight track, which can be passed as track_obj
        # I dont know what phase_out, ego_obs_avoid_d, tar_obs_avoid_d are, so I set them to default values
        self.straight_track = StraightTrack(length=10, width=100.0, slack=0.8, phase_out=True)
        
        self.scen_params = ScenarioGenParams(types=['track'], egoMin=self.egoMin, egoMax=self.egoMax, tarMin=self.tarMin, tarMax=self.tarMax, width=100.0)
        self.scen_gen = ScenarioGenerator(self.scen_params)
        self.scenario = self.scen_gen.genScenario()
        
        self.track_name = self.scenario.track_type
        self.track_obj = self.scenario.track

        self.ego_dynamics_simulator = DynamicsSimulator(self.t, ego_dynamics_config, track=self.track_obj)
        self.tar_dynamics_simulator = DynamicsSimulator(self.t, tar_dynamics_config, track=self.track_obj)

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

        self.mpcc_tv_params = MPCCApproxFullModelParams(
            dt=dt,
            all_tracks=all_tracks,
            solver_dir='' if rebuild else '~/.mpclab_controllers/mpcc_h2h_tv',
            # solver_dir='',
            optlevel=2,

            N=N,
            Qc=75,
            Ql=500.0,
            Q_theta=30.0,
            Q_xref=0.0,
            R_d=5.0,
            R_delta=25.0,

            slack=True,
            l_cs=10,
            Q_cs=2.0,
            Q_vmax=200.0,
            vlong_max_soft=1.0,
            Q_ts=500.0,
            Q_cs_e=8.0,
            l_cs_e=35.0,

            u_a_max=0.45,
            vx_max=1.3,
            u_a_min=-1,
            u_steer_max=0.435,
            u_steer_min=-0.435,
            u_a_rate_max=10,
            u_a_rate_min=-10,
            u_steer_rate_max=2,
            u_steer_rate_min=-2
        )
        
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.ego_dynamics_simulator.model, self.track_obj, self.gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name=self.track_name)
        self.gp_mpcc_ego_controller.initialize()

        self.gp_mpcc_ego_controller.set_warm_start(*self.ego_history)

        self.mpcc_tv_params.vectorize_constraints()
        self.mpcc_tv_controller = MPCC_H2H_approx(self.tar_dynamics_simulator.model, self.track_obj, self.mpcc_tv_params, name="mpcc_h2h_tv", track_name=self.track_name)
        self.mpcc_tv_controller.initialize()
        self.mpcc_tv_controller.set_warm_start(*self.tv_history)
        self.predictor = GPPredictor(N=10, track=self.track_obj, policy_name=self.policy_name, use_GPU=use_GPU, M=self.M, cov_factor=np.sqrt(2))

        # Initial prediction is expected to return None, step the simulation and move on
        self.ego_pred = self.gp_mpcc_ego_controller.get_prediction()

    def random_init(self, stateMin, stateMax):
        s = np.random.uniform(stateMin.p.s, stateMax.p.s)
        x_tran = np.random.uniform(stateMin.p.x_tran, stateMax.p.x_tran)
        e_psi = np.random.uniform(stateMin.p.e_psi, stateMax.p.e_psi)
        v_long = np.random.uniform(stateMin.v.v_long, stateMax.v.v_long)
        # print(s, x_tran, e_psi, v_long)
        return s, x_tran, e_psi, v_long
    
    # Function to initialize arrays with a single zero
    def zero_array(self):
        return array.array('f', [0.0])

    def ego_listener_callback(self, msg):
        # Listen to ego racecar's odometry message 
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # use the inputs coming from self.tar_odom_x,y and z for target vehicle pose
        
        gp_tarpred_list = [None] * n_iter
        
        ego_prediction, tar_prediction, tv_pred = None, None, None
        while self.t < self.T:
            if self.tar_sim_state.p.s >= 1.9 * self.scenario.length or self.ego_sim_state.p.s >= 1.9 * self.scenario.length:
                break
            else:
                if self.predictor:
                    # ego_pred is what the mpcc controller predicts
                    ego_pred = self.gp_mpcc_ego_controller.get_prediction()
                    if ego_pred.s is not None:
                        # based on the output from the mpcc controller, provide the current
                        # states of the ego and target vehicle to the GP Predictor and get a 
                        # prediction for the TV, that we use to maneuver
                        tv_pred = self.predictor.get_prediction(self.ego_sim_state, self.tar_sim_state, ego_pred)
                        gp_tarpred_list.append(tv_pred.copy())
                    else:
                        gp_tarpred_list.append(None)

                # Target agent
                info, tar_acc, tar_pos, tar_steering_angle = self.mpcc_tv_controller.step_racer(self.tar_sim_state, tv_state=self.ego_sim_state, tv_pred=ego_prediction, policy=self.policy_name)
                if not info["success"]:
                    print(f"TV infeasible")
                    pass

                # Ego agent
                info, ego_acc, ego_pos, ego_steering_angle = self.gp_mpcc_ego_controller.step_racer(self.ego_sim_state, tv_state=self.tar_sim_state, tv_pred=tar_prediction)
                if not info["success"]:
                    print(f"EGO infeasible")
                    pass
                
                # TODO - figure out what to do for speed
                
                new_drive_message = AckermannDriveStamped()
                new_drive_message.drive.acceleration = ego_acc
                new_drive_message.drive.steering_angle = ego_steering_angle
                new_drive_message.drive.speed = 0.5
                
                new_drive_message_target = AckermannDriveStamped()
                new_drive_message_target.drive.acceleration = tar_acc
                new_drive_message_target.drive.steering_angle = tar_steering_angle
                new_drive_message_target.drive.speed = 0.5
                
                self.pub_drive_ego.publish(new_drive_message_target)
                self.pub_drive_tar.publish(new_drive_message_target)

                
    def opp_listener_callback(self, msg):
        # Extract position data from Odometry message
        print("Inside opp listener callback")
        self.tar_odom_x = msg.pose.pose.position.x
        self.tar_odom_y = msg.pose.pose.position.y
        self.tar_odom_z = msg.pose.pose.position.z

def main(args=None):
    rclpy.init(args=args)
    node = GPOdometryPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

