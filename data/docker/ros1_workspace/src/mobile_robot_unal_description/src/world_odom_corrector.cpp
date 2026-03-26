#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <tf/transform_broadcaster.h>

std::string robot_name = "mobile_manipulator";
tf::TransformBroadcaster* br;

void gazeboCallback(const gazebo_msgs::ModelStates::ConstPtr& msg) {
    int index = -1;
    for (size_t i = 0; i < msg->name.size(); i++) {
        if (msg->name[i] == robot_name) { index = i; break; }
    }
    if (index == -1) return;

    // Publicar world -> odom (Identidad)
    br->sendTransform(tf::StampedTransform(
        tf::Transform(tf::Quaternion(0,0,0,1), tf::Vector3(0,0,0)),
        ros::Time::now(), "world", "odom"));

    // Publicar odom -> base_footprint (Ground Truth)
    const auto& pose = msg->pose[index];
    br->sendTransform(tf::StampedTransform(
        tf::Transform(
            tf::Quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            tf::Vector3(pose.position.x, pose.position.y, pose.position.z)
        ),
        ros::Time::now(), "odom", robot_name + "/base_footprint"));
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "ground_truth_tf_bridge");
    ros::NodeHandle nh;
    br = new tf::TransformBroadcaster();
    ros::Subscriber sub = nh.subscribe("/gazebo/model_states", 1, gazeboCallback);
    ros::spin();
    return 0;
}