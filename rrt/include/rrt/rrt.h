// RRT assignment

// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf

#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include <tf2_ros/transform_broadcaster.h>

/// CHECK: include needed ROS msg type headers and libraries

using namespace std;
typedef geometry_msgs::msg::Point Point;

// Struct defining the RRT_Node object in the RRT tree.
// More fields could be added to thiis struct if more info needed.
// You can choose to use this or not

typedef struct RRT_Node {
    Point pt;
    double cost; // only used for RRT*
    int parent; // index of parent node in the tree vector
    bool is_root = false;
    
    geometry_msgs::msg::PoseStamped get_posestamped_msg();
} RRT_Node;


class RRT : public rclcpp::Node {
public:
    RRT();
    virtual ~RRT();
private:

    // TODO: add the publishers and subscribers you need

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr opp_path_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

    std::vector<Point> waypoints_;
    visualization_msgs::msg::MarkerArray marker_array_;
    nav_msgs::msg::Path generated_path_;
    nav_msgs::msg::Path generated_path_map_frame_;

    // random generator, use this
    std::mt19937 gen;
    std::uniform_real_distribution<> x_dist;
    std::uniform_real_distribution<> y_dist;
    std::uniform_real_distribution<> goal_bias;

    //parameters
    double map_height_;
    double map_width_;
    uint32_t map_cell_height_;
    uint32_t map_cell_width_;
    double res_;
    Point center_;
    double max_expansion_dist_;
    double goal_thresh_;
    long int max_iters_;
    long int max_total_iters_;
    double goal_biasing_prob_;
    double neighbor_radius_;
    double pp_lookahead_dist_;
    bool visualize_;
    bool simulation_;
    bool run_pp_;
    bool first_run_;
    bool stop_running = false;
    // callbacks
    // where rrt actually happens
    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg);
    void pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg);
    // updates occupancy grid
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);

    nav_msgs::msg::OccupancyGrid local_map_;

    long int current_wp_index_;
    bool read_waypoints(string file_path);
    geometry_msgs::msg::PoseStamped transform_point_to_new_frame(Point input, geometry_msgs::msg::Pose frame);
    geometry_msgs::msg::Pose invert_pose(geometry_msgs::msg::Pose pose);

    // RRT methods
    Point sample(const Point& goal);
    int nearest(std::vector<RRT_Node> &tree, Point &sampled_point);
    RRT_Node steer(RRT_Node &nearest_node, Point &sampled_point);
    bool check_collision(RRT_Node &nearest_node, RRT_Node &new_node);
    bool is_goal(RRT_Node &latest_added_node, const Point& goal_pt);
    void find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node, geometry_msgs::msg::Pose map_frame);
    uint32_t pose_to_oneD_coords(const Point& pt);
    Point oneD_coords_to_pose (const uint32_t idx);
    double dist_bw_pts(const Point& a, const Point& b);
    double dist_to_pt(const Point& a);
    void pub_drive_msg(geometry_msgs::msg::Pose current_pose);

    void rrt_loop(geometry_msgs::msg::Pose current_pose);
    // RRT* methods
    double cost(std::vector<RRT_Node> &tree, RRT_Node &node);
    double line_cost(RRT_Node &n1, RRT_Node &n2);
    std::vector<int> near(std::vector<RRT_Node> &tree, RRT_Node &node);

};

