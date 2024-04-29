import torch
import gpytorch
from rclpy.node import Node
import rclpy
# import pickle
import math
import os
import numpy as np
from nav_msgs.msg import Odometry
from barcgp.prediction.gpytorch_models import IndependentMultitaskGPModelApproximate  # Ensure this is defined somewhere in your project
from barcgp.prediction.trajectory_predictor import GPPredictor  # Ensure this is defined somewhere in your project
from barcgp.common.utils.scenario_utils import ScenarioDefinition, ScenarioGenerator, ScenarioGenParams
from barcgp.common.tracks.track_lib import StraightTrack
from barcgp.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehiclePrediction,ParametricVelocity
from barcgp.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from barcgp.simulation.dynamics_simulator import DynamicsSimulator
from barcgp.h2h_configs import *
from barcgp.common.utils.scenario_utils import *
from barcgp.common.utils.file_utils import *
from barcgp.common_control import run_pid_warmstart

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

class GPOdometryPredictor(Node):
    def __init__(self):
        super().__init__('gp_odometry_predictor')
        
        # TODO - All of the parameters need not be "self", can make some of them
        # local once we have everything working
        
        self.tarMin = VehicleState(t=0.0,
                            p=ParametricPose(s=0.0, x_tran=-2.0, e_psi=-0.5),
                            v=BodyLinearVelocity(v_long=0.5*factor))
        self.tarMax = VehicleState(t=0.0,
                            p=ParametricPose(s=5.0, x_tran=2.0, e_psi=0.5),
                            v=BodyLinearVelocity(v_long=1.0*factor))
        self.egoMin = VehicleState(t=0.0,
                            p=ParametricPose(s=offset + 0.2, x_tran=-2.0, e_psi=-0.5),
                            v=BodyLinearVelocity(v_long=0.5*factor))
        self.egoMax = VehicleState(t=0.0,
                            p=ParametricPose(s=offset + 0.4, x_tran=2.0, e_psi=0.5),
                            v=BodyLinearVelocity(v_long=1.0*factor))
        
        self.subscription_opp = self.create_subscription(
            Odometry,
            '/opp_racecar/odom',
            self.opp_listener_callback,
            10)
        
        self.subscription_ego = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.ego_listener_callback,
            10)
        
        self.pub_drive_ego = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pub_drive_tar = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)
        self.pred_path_pub = self.create_publisher(Path, '/pred_path', 10)

        # # Parameters for the model
        # inducing_points_num = 200
        # input_dim = 3  # Assuming you're using 3D position data (x, y, z)
        # num_tasks = 5  # Number of tasks/output dimensions
        ##############################################################################################################################################
        use_GPU = True   
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
        # s, x_tran, e_psi, v_long = self.random_init(self.egoMin, self.egoMax)
        # self.ego_init_state = VehicleState(t=0.0,
        #                               p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
        #                               v=BodyLinearVelocity(v_long=v_long))
        
        # s, x_tran, e_psi, v_long = self.random_init(self.egoMin, self.egoMax)
        # self.tar_init_state = VehicleState(t=0.0,
        #                               p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
        #                               v=BodyLinearVelocity(v_long=v_long))

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
        
        self.scen_params = ScenarioGenParams(types=['straight'], egoMin=self.egoMin, egoMax=self.egoMax, tarMin=self.tarMin, tarMax=self.tarMax, width=100.0)
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
        print(self.gp_mpcc_ego_controller.get_prediction())

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

    def odom_to_vehicle_state(self,msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        psi = math.atan2(2 * msg.pose.pose.orientation.w * msg.pose.pose.orientation.z, 1 - 2 * (msg.pose.pose.orientation.z ** 2))
        psi = (psi + 2 * np.pi) % (2 * np.pi)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        w = msg.twist.twist.angular.z
        
        state = VehicleState(t=0.0, 
                             x=Position(x,y,0),
                             v=BodyLinearVelocity(vx,vy,0),
                             w=BodyAngularVelocity(0,0,w),
                             e=OrientationEuler(0,0,psi),
                             # TODO - Figure out what n should be 
                             p=ParametricPose(s=x,x_tran=y,n=0,e_psi=psi),
                             # TODO - Figure out what dn must be
                             pt=ParametricVelocity(ds=vx*np.cos(psi),dx_tran=vx*np.sin(psi),dn=0,de_psi=w)
                             )
        
        
        return state
    def ego_listener_callback(self, msg):
        # Listen to ego racecar's odometry message 
        # print("Inside ego listener callback")
        self.ego_sim_state = self.odom_to_vehicle_state(msg)
        # print("ego car state {}",self.ego_sim_state)
        # use the inputs coming from self.tar_odom_x,y and z for target vehicle pose
        
        gp_tarpred_list = [None] * n_iter
        
        ego_prediction, tar_prediction, tv_pred = None, None, None
        # while True:
            # self.t += 1
            # if self.tar_sim_state.p.s >= 1.9 * self.scenario.length or self.ego_sim_state.p.s >= 1.9 * self.scenario.length:
            #     break
            # else:
            #     if self.predictor:
            #         # ego_pred is what the mpcc controller predicts
            #         ego_pred = self.gp_mpcc_ego_controller.get_prediction()
            #         # print(ego_pred)
            #         if ego_pred.s is not None:
            #             # based on the output from the mpcc controller, provide the current
            #             # states of the ego and target vehicle to the GP Predictor and get a 
            #             # prediction for the TV, that we use to maneuver
            #             print(tv_pred)
            #             # gp_tarpred_list.append(tv_pred.copy())
            #         # else:
            #         #     gp_tarpred_list.append(None)
        ego_pred = VehiclePrediction()
        ego_pred.x = [7.0, 7.0, 7.0, 7.0, 7.0]
        self.tar_sim_state.v.v_long = 0.5 
        tv_pred = self.predictor.get_prediction(self.ego_sim_state, self.tar_sim_state, ego_pred)
        # print(tv_pred.print())
        pred_path_msg = Path()
        pred_path_msg.header.frame_id = 'map'
        # tv_pred.print()
        # print (self.tar_sim_state)
        for i in range(len(ego_pred.x)):
            pose = PoseStamped()
            pose.pose.position.x = tv_pred.s[i]
            pose.pose.position.y = tv_pred.x_tran[i]
            pred_path_msg.poses.append(pose)

        self.pred_path_pub.publish(pred_path_msg)

                # Target agent
                # info, tar_acc, tar_pos, tar_steering_angle = self.mpcc_tv_controller.step_racer(self.tar_sim_state, tv_state=self.ego_sim_state, tv_pred=ego_prediction, policy=self.policy_name)
                # if not info["success"]:
                #     print(f"TV infeasible")
                #     pass

                # Ego agent
                # info, ego_acc, ego_pos, ego_steering_angle = self.gp_mpcc_ego_controller.step_racer(self.ego_sim_state, tv_state=self.tar_sim_state, tv_pred=tar_prediction)
                # if not info["success"]:
                    # print(f"EGO infeasible")
                    # pass
                
                # TODO - figure out what to do for speed
                
                # print("HEREEEE")
                
                # new_drive_message = AckermannDriveStamped()
                # new_drive_message.drive.acceleration = 0
                # new_drive_message.drive.steering_angle = ego_steering_angle         
                # new_drive_message.drive.speed = msg.twist.twist.linear.x + (ego_acc / 30.0)
                
                # new_drive_message_target = AckermannDriveStamped()
                # new_drive_message_target.drive.acceleration = tar_acc
                # new_drive_message_target.drive.steering_angle = tar_steering_angle
                # new_drive_message_target.drive.speed = 0.15
                
                # print(new_drive_message)
                
                # self.pub_drive_ego.publish(new_drive_message)
            
                # self.pub_drive_tar.publish(new_drive_message_target)


                
    def opp_listener_callback(self, msg):
        # Extract position data from Odometry message
        print("Inside opp listener callback")
        self.tar_sim_state = self.odom_to_vehicle_state(msg)
        # print(msg.pose.pose.position.x)
        new_drive_message_target = AckermannDriveStamped()
        new_drive_message_ego = AckermannDriveStamped()
        # new_drive_message_target.drive.acceleration = 0
        new_drive_message_target.drive.steering_angle = 0.1
        new_drive_message_target.drive.speed = 1.0
        new_drive_message_ego.drive.speed = 0.0
        self.pub_drive_tar.publish(new_drive_message_target)
        self.pub_drive_ego.publish(new_drive_message_ego)

        # print("tar car state {}",self.tar_sim_state)


def main(args=None):
    rclpy.init(args=args)
    node = GPOdometryPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

