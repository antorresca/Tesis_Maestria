#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

std::string robot_name = "mobile_manipulator";
ros::Publisher pub;

void gazeboCallback(const gazebo_msgs::ModelStates::ConstPtr& msg) {
    int index = -1;
    for (size_t i = 0; i < msg->name.size(); i++) {
        if (msg->name[i] == robot_name) { index = i; break; }
    }
    if (index == -1) return;

    geometry_msgs::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = ros::Time::now();
    pose_msg.header.frame_id = "odom";
    pose_msg.pose.pose = msg->pose[index];

    // Covarianza: Valores pequeños = Alta confianza del EKF en Gazebo
    pose_msg.pose.covariance[0] = 0.01; 
    pose_msg.pose.covariance[7] = 0.01;
    pose_msg.pose.covariance[35] = 0.01;

    pub.publish(pose_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "gazebo_to_ekf");
    ros::NodeHandle nh;
    pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/mobile_manipulator/model_states_pose", 1);
    ros::Subscriber sub = nh.subscribe("/gazebo/model_states", 1, gazeboCallback);
    ros::spin();
    return 0;
}