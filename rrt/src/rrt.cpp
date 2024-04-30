// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well

#include "rrt/rrt.h"
#include <fstream>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

/*
Stuff that can be improved
- Convert to RRT*
- Laser scan marking (use bresenham)
- Collision checking (use bresenham)
- Map size keep as dynamic (so that goal is always inside the map)
- If goal is not free, deal with it properly instead of just giving the old path
- Improve safety bounds while marking with laser
- Use hash map instead of vector to store tree, so that retrieval is fast.
*/

#define OCCUPIED 100
#define FREE 0
#define UNKNOWN -1

geometry_msgs::msg::PoseStamped RRT_Node::get_posestamped_msg() {
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "map";
  pose.pose.position.x = pt.x;
  pose.pose.position.y = pt.y;

  return pose;
}

// Destructor of the RRT class
RRT::~RRT() {
  // Do something in here, free up used memory, print message, etc.
  RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "RRT shutting down");
}

// Constructor of the RRT class
long RRT::pose_to_oneD_coords(const Point &pt) {
  
  long int y_cell = lround(floor((pt.y - center_.y) / res_));
  long int x_cell = lround(floor((pt.x - center_.x) / res_));
  
  if (y_cell < 0 || y_cell >= map_cell_height_) {
    return -1;
  }

  if (x_cell < 0 || x_cell >= map_cell_width_) {
    return -1;
  }

  long int return_val = y_cell * map_cell_width_ + x_cell;
  
  return return_val;
}

Point RRT::oneD_coords_to_pose(long idx) {
  Point pt;
  pt.x = (idx % map_cell_width_) * res_ + center_.x + (res_ / 2.0);
  pt.y = (idx / map_cell_width_) * res_ + center_.y + (res_ / 2.0);

  return pt;
}

double RRT::dist_bw_pts(const Point &a, const Point &b) {
  return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double RRT::dist_to_pt(const Point &a) { return sqrt(a.x * a.x + a.y * a.y); }

bool RRT::read_waypoints(string file_path) {
  ifstream csv_file = ifstream(file_path, std::ios_base::in);

  if (!csv_file.good()) {
    return false;
  }

  string x_str;
  string y_str;

  int32_t id = 0;
  int count = 0;

  while (getline(csv_file, x_str, ',') && getline(csv_file, y_str, ',')) {
    string dummy;
    getline(csv_file, dummy);

    Point waypoint;
    waypoint.x = strtod(x_str.c_str(), NULL);
    waypoint.y = strtod(y_str.c_str(), NULL);

    if (visualize_) {
      auto marker = visualization_msgs::msg::Marker();
      marker.action = marker.ADD;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      marker.pose.position.x = waypoint.x;
      marker.pose.position.y = waypoint.y;
      marker.pose.position.z = 0.1;
      marker.header.frame_id = "map";
      marker.scale.x = 0.1;
      marker.scale.y = 0.1;
      marker.scale.z = 0.1;
      marker.id = id;

      id++;
      marker_array_.markers.push_back(marker);
    }

    waypoints_.push_back(waypoint);
  }

  RCLCPP_INFO_STREAM(get_logger(),
                     "Waypoints loaded. Size: " << waypoints_.size());
  csv_file.close();
  return true;
}

geometry_msgs::msg::PoseStamped
RRT::transform_point_to_new_frame(Point input, geometry_msgs::msg::Pose frame) {
  tf2::Quaternion frame_q(frame.orientation.x, frame.orientation.y,
                          frame.orientation.z, frame.orientation.w);

  tf2::Matrix3x3 frame_matrix(frame_q);
  double roll, pitch, yaw;
  frame_matrix.getRPY(roll, pitch, yaw);
  double theta = atan2(input.y - frame.position.y, input.x - frame.position.x);
  double theta_diff = theta - yaw;

  geometry_msgs::msg::PoseStamped transformed_point;
  transformed_point.pose.position.x =
      dist_bw_pts(input, frame.position) * cos(theta_diff);
  transformed_point.pose.position.y =
      dist_bw_pts(input, frame.position) * sin(theta_diff);

  return transformed_point;
}

geometry_msgs::msg::Pose RRT::invert_pose(geometry_msgs::msg::Pose frame) {

  tf2::Quaternion frame_q(frame.orientation.x, frame.orientation.y,
                          frame.orientation.z, frame.orientation.w);

  tf2::Matrix3x3 frame_matrix(frame_q);
  double roll, pitch, yaw;
  frame_matrix.getRPY(roll, pitch, yaw);

  double theta = atan2(-frame.position.y, -frame.position.x);

  double theta_new = theta - yaw;
  tf2::Quaternion new_frame_q;
  new_frame_q.setRPY(0, 0, -yaw);

  double length = sqrt(frame.position.x * frame.position.x +
                       frame.position.y * frame.position.y);

  geometry_msgs::msg::Pose new_pose;
  new_pose.position.x = length * cos(theta_new);
  new_pose.position.y = length * sin(theta_new);
  new_pose.orientation.x = new_frame_q.getX();
  new_pose.orientation.y = new_frame_q.getY();
  new_pose.orientation.z = new_frame_q.getZ();
  new_pose.orientation.w = new_frame_q.getW();

  return new_pose;
}

RRT::RRT() : rclcpp::Node("rrt_node"), gen((std::random_device())()) {

  // ROS publishers
  // TODO: create publishers for the the drive topic, and other topics you might
  // need

  // ROS subscribers
  // TODO: create subscribers as you need
  string odom_topic = "ego_racecar/odom";
  string pose_topic = "pf/viz/inferred_pose";
  string scan_topic = "scan";

  map_height_ = 5.0; // LENGTH OF MAP ALONG Y AXIS
  map_width_ = 3.0;  // LENGTH OF MAP ALONG X AXIS
  res_ = 0.1;
  center_.x = 0.0;
  center_.y = -map_height_ / 2.0;
  max_expansion_dist_ = 0.2;
  goal_thresh_ = 0.1;
  // max_iters_ = 5000;
  max_total_iters_ = 5000;
  goal_biasing_prob_ = 0.0;
  neighbor_radius_ = 0.75;
  car_size_ = 0.5;
  visualize_ = true;
  simulation_ = true;
  run_pp_ = true;
  pp_lookahead_dist_ = 0.5;
  map_cell_height_ = long(map_height_ / res_);
  map_cell_width_ = long(map_width_ / res_);

  // std::string file_path = "/home/racer/f1tenth_ws/src/F1TenthWS/rrt/waypoints_rrt/locker_wp.csv";
  std::string file_path = "/home/kshitij/f110/test_ws/src/F1TenthWS/rrt/waypoints_rrt/levine_wp_dense.csv";

  if (read_waypoints(file_path)) {
    RCLCPP_INFO_STREAM(get_logger(), "Waypoints loaded successfully");
  } else {
    RCLCPP_ERROR_STREAM(get_logger(), "Error loading waypoints. Ending node.");
    return;
  }

  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, 1,
      std::bind(&RRT::odom_callback, this, std::placeholders::_1));
  scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, 1,
      std::bind(&RRT::scan_callback, this, std::placeholders::_1));

  opp_path_sub_ = create_subscription<nav_msgs::msg::Path>(
      "pred_path", 1,
      std::bind(&RRT::opp_path_callback, this, std::placeholders::_1));

  map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("local_map", 10);
  path_pub_ = create_publisher<nav_msgs::msg::Path>("rrt_path", 10);
  drive_pub_ =
      create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("drive", 10);
  


  if (simulation_) {
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, 1,
        std::bind(&RRT::odom_callback, this, std::placeholders::_1));
  } else {
    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        pose_topic, 1,
        std::bind(&RRT::pose_callback, this, std::placeholders::_1));
  }

  if (visualize_) {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/waypoints", 1);
    marker_pub_->publish(marker_array_);
  }
  current_wp_index_ = 0;

  // TODO: create a occupancy grid
  local_map_.data = std::vector<int8_t>(
      uint32_t(map_cell_width_ * map_cell_height_), UNKNOWN);

  generated_path_map_frame_.header.frame_id = "map";

  if (simulation_) {
    local_map_.header.frame_id = "ego_racecar/laser";
    generated_path_.header.frame_id = "ego_racecar/laser";
  } else {
    local_map_.header.frame_id = "laser";
    generated_path_.header.frame_id = "laser";
  }

  local_map_.info.origin.orientation.w = 1;
  local_map_.info.origin.position.x = center_.x;
  local_map_.info.origin.position.y = center_.y;
  local_map_.info.resolution = res_;
  local_map_.info.height = map_cell_height_;
  local_map_.info.width = map_cell_width_;

  Point map_corner1 = oneD_coords_to_pose(0);
  Point map_corner2 =
      oneD_coords_to_pose(map_cell_height_ * map_cell_width_ - 1);

  RCLCPP_INFO_STREAM(get_logger(),
                     "Corner 1 " << map_corner1.x << ", " << map_corner1.y);
  RCLCPP_INFO_STREAM(get_logger(),
                     "Corner 2 " << map_corner2.x << ", " << map_corner2.y);

  x_dist = std::uniform_real_distribution<>(map_corner1.x, map_corner2.x);
  y_dist = std::uniform_real_distribution<>(map_corner1.y, map_corner2.y);

  RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "Created new RRT Object.");
  RCLCPP_INFO_STREAM(rclcpp::get_logger("RRT"),
                     "Size of waypoints is " << waypoints_.size());

  first_run_ = true;
}

void RRT::scan_callback(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
  // The scan callback, update your occupancy grid here
  // Args:
  //    scan_msg (*LaserScan): pointer to the incoming scan message
  // Returns:
  //
  // RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "Started scan callback.");
  local_map_.data.clear();
  local_map_.data = std::vector<int8_t>(
      uint32_t(map_cell_width_ * map_cell_height_), UNKNOWN);

  for (uint32_t i = 0; i < map_cell_width_ * map_cell_height_; i++) {
    Point p = oneD_coords_to_pose(i);
    double dist_to_orig = dist_to_pt(p);
    double slope = atan2(p.y, p.x);

    for (int k = 0; k < scan_msg->ranges.size(); k++) {
      double ray_slope = scan_msg->angle_min + k * scan_msg->angle_increment;
      double ray_distance = scan_msg->ranges[k];

      bool slope_equal = (abs(slope - ray_slope) < 0.05);
      bool dist_correct = (abs(dist_to_orig - ray_distance) < 0.2);
      bool dist_less = (dist_to_orig < (ray_distance - 0.2));
      if (slope_equal) {
        if (dist_correct) {
          local_map_.data[i] = OCCUPIED;
        } else if (dist_less) {
          local_map_.data[i] = FREE;
        }
        break;
      }
    }
  }
  // TODO: update your occupancy grid
  local_map_.header.stamp = get_clock()->now();

  // geometry_msgs::msg::Pose map_in_car_frame = invert_pose(current_pose_);


  for (auto opp_pose_in_map_frame : opp_path_.poses) {
    auto opp_pose = transform_point_to_new_frame(opp_pose_in_map_frame.pose.position, current_pose_);
    // auto opp_pose = opp_pose_in_map_frame;
    // RCLCPP_INFO_STREAM(get_logger(), "Opponent pose " << opp_pose.pose.position.x << " " << opp_pose.pose.position.y);
    for (double x = opp_pose.pose.position.x - (car_size_ / 2.0); x < opp_pose.pose.position.x + (car_size_ / 2.0); x += res_) {
      for (double y = opp_pose.pose.position.y - (car_size_ / 2.0); y < opp_pose.pose.position.y + (car_size_ / 2.0); y += res_) {
        Point p;
        p.x = x;
        p.y = y;
        
        long p_oneD = pose_to_oneD_coords(p);

        if (p_oneD < 0 || p_oneD > local_map_.data.size()) {
          continue;
        }
        
        local_map_.data[p_oneD] = OCCUPIED;
        // RCLCPP_INFO_STREAM(get_logger(), "Occupied " << p_oneD);
        // RCLCPP_INFO_STREAM(get_logger(), "Occupied " << p.x << " " << p.y);

      }
    }    
  }

  if (visualize_) {
    map_pub_->publish(local_map_);
  }
}

void RRT::odom_callback(
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg) {
  rrt_loop(odom_msg->pose.pose);
}

void RRT::pose_callback(
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg) {
  rrt_loop(pose_msg->pose);
}

void RRT::rrt_loop(geometry_msgs::msg::Pose current_pose) {
  // The pose callback when subscribed to particle filter's inferred pose
  // The RRT main loop happens here
  // Args:
  //    pose_msg (*PoseStamped): pointer to the incoming pose message
  // Returns:
  //

  // tree as std::vector
  // RCLCPP_INFO_STREAM(this->get_logger(), "Inside RRT loop");

  if (stop_running) {
    return;
  }

  current_pose_ = current_pose;

  geometry_msgs::msg::Pose map_in_car_frame = invert_pose(current_pose);

  // if (first_run_) {
  //   first_run_ = false;
    
  //   double min_dist = __DBL_MAX__;
  //   for (int i = 0; i < int(waypoints_.size()); i++) {
  //     double dist = dist_bw_pts(current_pose.position, waypoints_[i]);
  //     if (dist < min_dist) {
  //       min_dist = dist;
  //       current_wp_index_ = i;
  //     }
  //   }
  // }

  long int next_wp_index;
  if (current_wp_index_ == (int(waypoints_.size()) - 1)) {
    next_wp_index = 0;
  } else {
    next_wp_index = current_wp_index_ + 1;
  }
  // RCLCPP_INFO_STREAM(this->get_logger(), "Inside RRT loop2 "
                                            //  << "Cur " << current_wp_index_
                                            //  << "; next " << next_wp_index);

  double dist_to_curr_wp =
      dist_bw_pts(current_pose.position, waypoints_[current_wp_index_]);
  double dist_curr_to_next_wp =
      dist_bw_pts(waypoints_[current_wp_index_], waypoints_[next_wp_index]);
  double dist_to_next_wp =
      dist_bw_pts(current_pose.position, waypoints_[next_wp_index]);
  // RCLCPP_INFO_STREAM(this->get_logger(), "Inside RRT loop3");

  if ((dist_to_curr_wp < pp_lookahead_dist_) ||
      (dist_to_next_wp < dist_curr_to_next_wp)) {
    current_wp_index_ = next_wp_index;
  }

  std::vector<RRT_Node> tree;

  auto goal_pose =
      transform_point_to_new_frame(waypoints_[current_wp_index_], current_pose);
  // RCLCPP_INFO_STREAM(this->get_logger(), "Inside RRT loop4");

  auto goal_pt = goal_pose.pose.position;

  long int goal_idx = pose_to_oneD_coords(goal_pt);

  // RCLCPP_INFO_STREAM(this->get_logger(), "Inside RRT loop5");

  if (goal_idx < 0 || goal_idx > local_map_.data.size()) {
    RCLCPP_WARN_STREAM(get_logger(),
                       "Goal is outside map. Publishing previous path");
    if (visualize_) {
      path_pub_->publish(generated_path_);
    }

    if (run_pp_) {
      pub_drive_msg(current_pose);
    }

    return;
  }

  // if (local_map_.data[goal_idx] == OCCUPIED) {
  //   RCLCPP_WARN_STREAM(get_logger(),
  //                      "Goal is not free. Publishing previous path");
  //   if (visualize_) {
  //     path_pub_->publish(generated_path_);
  //   }

  //   if (run_pp_) {
  //     pub_drive_msg(current_pose);
  //   }

  //   return;
  // }

  RRT_Node start_node;
  start_node.parent = -1;
  start_node.cost = 0.0;
  tree.push_back(start_node);
  // TODO: fill in the RRT main loop
  long int total_iters = 0;

  for (int i = 0; i < max_total_iters_; i++) {
    Point sampled_pt = sample(goal_pt);

    int nearest_neighbour_idx = nearest(tree, sampled_pt);
    RRT_Node new_node = steer(tree[nearest_neighbour_idx], sampled_pt);

    if (check_collision(tree[nearest_neighbour_idx], new_node)) {
      // RCLCPP_WARN_STREAM(get_logger(), "Discrading node\n");
      continue;
    }
    // i++;

    // if (total_iters > max_total_iters_) {
    //   RCLCPP_WARN_STREAM(get_logger(),
    //                      "Max number of total iterations exceeded and goal not "
    //                      "found. Returning path from nearest node to goal.");
    //   // if (visualize_) {
    //     // path_pub_->publish(generated_path_);
    //   // }

    //   // if (run_pp_) {
    //     // pub_drive_msg(current_pose);
    //   // }

    //   break;
    // }

    new_node.parent = nearest_neighbour_idx;
    new_node.cost = tree[nearest_neighbour_idx].cost + line_cost(tree[nearest_neighbour_idx], new_node);
    tree.push_back(new_node);

    auto nearest_nodes = near(tree, new_node);
    for (auto &idx : nearest_nodes) {
      if (check_collision(tree[idx], new_node)) {
        continue;
      }

      if (idx == nearest_neighbour_idx) {
        continue;
      }

      double new_cost = new_node.cost + line_cost(tree[idx], new_node);
      if (new_cost < tree[idx].cost) {
        tree[idx].parent = tree.size() - 1;
        tree[idx].cost = new_cost;
      }
    }

    if (is_goal(new_node, goal_pt)) {
      find_path(tree, new_node, map_in_car_frame);
      if (visualize_) {
        path_pub_->publish(generated_path_);
      }

      if (run_pp_) {
        pub_drive_msg(current_pose);
      }

      return;
    }
  }
  int nearest_goal_idx = nearest(tree, goal_pt);
  double dist = dist_bw_pts(tree[nearest_goal_idx].pt, goal_pt);
  // RCLCPP_WARN_STREAM(get_logger(),
                    //  "Maximum number of iterations exceeded and goal not "
                    //  "found, using the point at a distance of"
                        //  << dist << "instead.");
  find_path(tree, tree[nearest_goal_idx], map_in_car_frame);
  if (visualize_) {
    path_pub_->publish(generated_path_);
  }

  if (run_pp_) {
    pub_drive_msg(current_pose);
  }
}

Point RRT::sample(const Point &goal) {
  // This method returns a sampled point from the free space
  // You should restrict so that it only samples a small region
  // of interest around the car's current position
  // Args:
  // Returns:
  //     sampled_point (Point): the sampled point in free space

  Point sampled_point;
  // TODO: fill in this method
  // look up the documentation on how to use std::mt19937 devices with a
  // distribution the generator and the distribution is created for you (check
  // the header file)
  if (goal_bias(gen) < goal_biasing_prob_) {
    sampled_point.x = goal.x;
    sampled_point.y = goal.y;
  } else {
    sampled_point.x = x_dist(gen);
    sampled_point.y = y_dist(gen);
  }
  return sampled_point;
}

int RRT::nearest(std::vector<RRT_Node> &tree, Point &sampled_point) {
  // This method returns the nearest node on the tree to the sampled point
  // Args:
  //     tree (std::vector<RRT_Node>): the current RRT tree
  //     sampled_point (Point): the sampled point in free space
  // Returns:
  //     nearest_node (int): index of nearest node on the tree

  int nearest_node = 0;
  // TODO: fill in this method
  double nearest_node_dist = dist_bw_pts(sampled_point, tree[0].pt);
  for (int i = 1; i < int(tree.size()); i++) {
    double dist = dist_bw_pts(sampled_point, tree[i].pt);
    if (dist < nearest_node_dist) {
      nearest_node_dist = dist;
      nearest_node = i;
    }
  }

  return nearest_node;
}

RRT_Node RRT::steer(RRT_Node &nearest_node, Point &sampled_point) {
  // The function steer:(x,y)->z returns a point such that z is “closer”
  // to y than x is. The point z returned by the function steer will be
  // such that z minimizes ||z−y|| while at the same time maintaining
  //||z−x|| <= max_expansion_dist, for a prespecified max_expansion_dist > 0

  // basically, expand the tree towards the sample point (within a max dist)

  // Args:
  //    nearest_node (RRT_Node): nearest node on the tree to the sampled point
  //    sampled_point (Point): the sampled point in free space
  // Returns:
  //    new_node (RRT_Node): new node created from steering

  RRT_Node new_node;
  // TODO: fill in this method
  double dist = dist_bw_pts(sampled_point, nearest_node.pt);
  if (dist < max_expansion_dist_) {
    new_node.pt = sampled_point;
    return new_node;
  }

  double slope = atan2(sampled_point.y - nearest_node.pt.y,
                       sampled_point.x - nearest_node.pt.x);

  new_node.pt.x = nearest_node.pt.x + cos(slope) * max_expansion_dist_;
  new_node.pt.y = nearest_node.pt.y + sin(slope) * max_expansion_dist_;
  return new_node;
}

bool RRT::check_collision(RRT_Node &nearest_node, RRT_Node &new_node) {
  // This method returns a boolean indicating if the path between the
  // nearest node and the new node created from steering is collision free
  // Args:
  //    nearest_node (RRT_Node): nearest node on the tree to the sampled point
  //    new_node (RRT_Node): new node created from steering
  // Returns:
  //    collision (bool): true if in collision, false otherwise

  // TODO: fill in this method

  // static int num_times = 0;

  // if (num_times > 10) {
    // num_times = 0;
    // stop_running = true;
  // }
  double dist = dist_bw_pts(nearest_node.pt, new_node.pt);
  double slope = atan2(new_node.pt.y - nearest_node.pt.y,
                       new_node.pt.x - nearest_node.pt.x);

    // RCLCPP_WARN_STREAM(get_logger(), "new_node" << new_node.pt.x << " " << new_node.pt.y);
    // RCLCPP_WARN_STREAM(get_logger(), "nearest_node" << nearest_node.pt.x << " " << nearest_node.pt.y);
  

  for (double len = 0; len <= dist; len += (res_ / 2.0)) {
    Point interp_pt;
    interp_pt.x = nearest_node.pt.x + cos(slope) * len;
    interp_pt.y = nearest_node.pt.y + sin(slope) * len;
    // RCLCPP_INFO_STREAM(get_logger(), "midpoints" << interp_pt.x << " " << interp_pt.y);


    if (local_map_.data[pose_to_oneD_coords(interp_pt)] == OCCUPIED) {
      
      return true;
    }
  }

  if (local_map_.data[pose_to_oneD_coords(new_node.pt)] == OCCUPIED) {
    return true;
  }
  // RCLCPP_WARN_STREAM(get_logger(), "Done" << new_node.pt.x << " " << new_node.pt.y);

  return false;
}

bool RRT::is_goal(RRT_Node &latest_added_node, const Point &goal_pt) {
  // This method checks if the latest node added to the tree is close
  // enough (defined by goal_threshold) to the goal so we can terminate
  // the search and find a path
  // Args:
  //   latest_added_node (RRT_Node): latest addition to the tree
  //   goal_x (double): x coordinate of the current goal
  //   goal_y (double): y coordinate of the current goal
  // Returns:
  //   close_enough (bool): true if node close enough to the goal

  // TODO: fill in this method
  if (dist_bw_pts(latest_added_node.pt, goal_pt) < goal_thresh_) {
    return true;
  }

  return false;
}

void RRT::find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node,
                    geometry_msgs::msg::Pose map_frame) {
  // This method traverses the tree from the node that has been determined
  // as goal
  // Args:
  //   latest_added_node (RRT_Node): latest addition to the tree that has been
  //      determined to be close enough to the goal
  // Returns:
  //   path (std::vector<RRT_Node>): the vector that represents the order of
  //      of the nodes traversed as the found path

  generated_path_.poses.clear();
  generated_path_map_frame_.poses.clear();
  // TODO: fill in this method
  RRT_Node curr_node = latest_added_node;
  generated_path_.header.stamp = get_clock()->now();
  generated_path_map_frame_.header.stamp = get_clock()->now();

  while (curr_node.parent != -1) {
    generated_path_.poses.push_back(curr_node.get_posestamped_msg());

    auto map_pose = transform_point_to_new_frame(
        curr_node.get_posestamped_msg().pose.position, map_frame);
    map_pose.header.frame_id = "map";
    generated_path_map_frame_.poses.push_back(map_pose);

    // bool iscollide = check_collision(tree[curr_node.parent], curr_node);
    // if (iscollide) {
      // RCLCPP_WARN_STREAM(get_logger(), "Collision between nodes!");
      // break;
    // }

    curr_node = tree[curr_node.parent];
  }

  generated_path_.poses.push_back(curr_node.get_posestamped_msg());

  auto map_pose = transform_point_to_new_frame(
      curr_node.get_posestamped_msg().pose.position, map_frame);
  map_pose.header.frame_id = "map";
  generated_path_map_frame_.poses.push_back(map_pose);

  std::reverse(generated_path_.poses.begin(), generated_path_.poses.end());
  std::reverse(generated_path_map_frame_.poses.begin(),
               generated_path_map_frame_.poses.end());
}

void RRT::pub_drive_msg(geometry_msgs::msg::Pose current_pose) {
  // TODO: transform goal point to vehicle frame of reference

  Point goal_in_vehicle_frame = generated_path_.poses.back().pose.position;
  for (auto wp_pose : generated_path_.poses) {
    if (dist_to_pt(wp_pose.pose.position) > pp_lookahead_dist_) {
      goal_in_vehicle_frame = wp_pose.pose.position;
      break;
    }
  }

  // TODO: calculate curvature/steering angle
  double L =
      sqrt(pow(goal_in_vehicle_frame.x, 2) + pow(goal_in_vehicle_frame.y, 2));

  double steering_angle = (2 * goal_in_vehicle_frame.y / (L * L));
  if (steering_angle > (M_PI / 6)) {
    steering_angle = M_PI / 6;
  } else if (steering_angle < (-M_PI / 6)) {
    steering_angle = -M_PI / 6;
  }

  double final_speed;

  if (abs(steering_angle) < (10 * M_PI / 180)) {
    final_speed = 0.75;
  } else if (abs(steering_angle) < (20 * M_PI / 180)) {
    final_speed = 0.5;
  } else {
    final_speed = 0.25;
  }

  // TODO: publish drive message, don't forget to limit the steering angle.
  auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
  drive_msg.drive.steering_angle = steering_angle;
  drive_msg.header.stamp = this->get_clock()->now();
  drive_msg.drive.speed = final_speed;
  drive_pub_->publish(drive_msg);
}

// RRT* methods
double RRT::cost(std::vector<RRT_Node> &tree, RRT_Node &node) {
  // This method returns the cost associated with a node
  // Args:
  //    tree (std::vector<RRT_Node>): the current tree
  //    node (RRT_Node): the node the cost is calculated for
  // Returns:
  //    cost (double): the cost value associated with the node

  double cost = 0;
  // TODO: fill in this method

  return cost;
}

double RRT::line_cost(RRT_Node &n1, RRT_Node &n2) {
  // This method returns the cost of the straight line path between two nodes
  // Args:
  //    n1 (RRT_Node): the RRT_Node at one end of the path
  //    n2 (RRT_Node): the RRT_Node at the other end of the path
  // Returns:
  //    cost (double): the cost value associated with the path

  if (check_collision(n1, n2)) {
    RCLCPP_WARN_STREAM(get_logger(), "Collision between nodes!");
    return __DBL_MAX__;
  }

  return dist_bw_pts(n1.pt, n2.pt); 
  // TODO: fill in this method
}

std::vector<int> RRT::near(std::vector<RRT_Node> &tree, RRT_Node &node) {
  // This method returns the set of Nodes in the neighborhood of a
  // node.
  // Args:
  //   tree (std::vector<RRT_Node>): the current tree
  //   node (RRT_Node): the node to find the neighborhood for
  // Returns:
  //   neighborhood (std::vector<int>): the index of the nodes in the
  //   neighborhood

  std::vector<int> neighborhood;
  // TODO:: fill in this method
  for (int i = 0; i < int(tree.size()); i++) {
    if (dist_bw_pts(tree[i].pt, node.pt) < neighbor_radius_) {
      neighborhood.push_back(i);
    }
  }

  return neighborhood;
}

void RRT::opp_path_callback(
    const nav_msgs::msg::Path::ConstSharedPtr path_msg) {
  // The opponent path callback, update your occupancy grid here
  // Args:
  //    path_msg (*Path): pointer to the incoming path message
  // Returns:
  //
  // RCLCPP_WARN_STREAM(get_logger(), "Opponent path received");
  opp_path_ = *path_msg;
}