#pragma once

namespace tf2_ros {
class StaticTransformBroadcaster;
}  // namespace tf2_ros

#include "depthai_bridge/ImageConverter.hpp"
#include "image_transport/image_transport.hpp"
#include "opencv2/core.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "depthai/common/CameraBoardSocket.hpp"
#include "depthai/common/CameraFeatures.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/sensor_wrapper.hpp"

namespace dai {
class Pipeline;
class Device;
class MessageQueue;
class InputQueue;
class ADatatype;
class ImgFrame;
namespace node {
class ImageAlign;
class StereoDepth;
class NeuralDepth;
}  // namespace node
}  // namespace dai

namespace rclcpp {
class Node;
class Parameter;
}  // namespace rclcpp

namespace depthai_ros_driver {
namespace param_handlers {
class StereoParamHandler;
}

namespace dai_nodes {
namespace link_types {
enum class StereoLinkType { stereo, left, right, align };
};

namespace sensor_helpers {
class ImagePubliser;
}
class RGBD;
class ConfidenceMask;
class StereoNodeWrapper {
   public:
    StereoNodeWrapper() {}
    std::shared_ptr<dai::node::StereoDepth> stereo;
    std::shared_ptr<dai::node::NeuralDepth> neuralDepth;
    auto getNode();
};

class Stereo : public BaseNode {
   public:
    explicit Stereo(const std::string& daiNodeName,
                    std::shared_ptr<rclcpp::Node> node,
                    std::shared_ptr<dai::Pipeline> pipeline,
                    std::shared_ptr<dai::Device> device,
                    bool rsCompat,
                    dai::CameraBoardSocket leftSocket = dai::CameraBoardSocket::CAM_B,
                    dai::CameraBoardSocket rightSocket = dai::CameraBoardSocket::CAM_C);
    ~Stereo();
    void setupQueues(std::shared_ptr<dai::Device> dvice) override;
    void updateParams(const std::vector<rclcpp::Parameter>& params) override;
    void link(dai::Node::Input& in, int linkType = 1) override;
    dai::Node::Input& getInput(int linkType = 0) override;
    void setNames() override;
    void setInOut(std::shared_ptr<dai::Pipeline> pipeline) override;
    void closeQueues() override;
    std::vector<std::shared_ptr<sensor_helpers::ImagePublisher>> getPublishers() override;
    std::shared_ptr<dai::node::StereoDepth> getUnderlyingNode();
    std::shared_ptr<dai::node::NeuralDepth> getNeuralDepthNode();
    bool isAligned();
    dai::CameraBoardSocket getSocketID();
    std::shared_ptr<SensorWrapper> getLeftSensor();
    std::shared_ptr<SensorWrapper> getRightSensor();
    int getWidth();
    int getHeight();

   private:
    void setupStereoQueue(std::shared_ptr<dai::Device> device);
    void setupConfidenceQueue(std::shared_ptr<dai::Device> device);
    void setupLeftRectQueue(std::shared_ptr<dai::Device> device);
    void setupRightRectQueue(std::shared_ptr<dai::Device> device);
    void setupRectQueue(std::shared_ptr<dai::Device> device, dai::CameraFeatures& sensorInfo, std::shared_ptr<sensor_helpers::ImagePublisher> pub, bool isLeft);
    void computeRectifyRecipe(std::shared_ptr<dai::Device> device);
    std::array<double, 9> rectifyRLeft{}, rectifyRRight{};
    std::array<double, 12> rectifyPLeft{}, rectifyPRight{};
    // Raw intrinsics (at i_width x i_height) and the 8-coefficient distortion
    // the recipe was computed with; the host wide rect rebuilds from these.
    cv::Mat rectifyKLeft, rectifyKRight, rectifyDLeft, rectifyDRight;
    // Host-side "wide" rectification: same R1/R2 as the firmware mesh, but the
    // projection is the largest sensor-size rectangle fully inside the raw
    // content (anisotropic focal), so the whole lens FOV survives with no black
    // borders. Published next to the firmware rect as <side>_rect_wide.
    struct WideRect {
        std::shared_ptr<dai::MessageQueue> q;
        int cbId = -1;
        cv::Mat map1, map2;
        sensor_msgs::msg::CameraInfo info;
        std::shared_ptr<depthai_bridge::ImageConverter> conv;
        image_transport::CameraPublisher pub;
        bool warned = false;
    };
    WideRect wideLeft, wideRight;
    void setupWideRectQueues();
    void publishWideRect(const std::shared_ptr<dai::ADatatype>& data, bool isLeft);
    std::shared_ptr<sensor_helpers::ImagePublisher> stereoPub, leftRectPub, rightRectPub, confidencePub;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> rectTfBroadcaster;
    StereoNodeWrapper stereoNodeWrapper;
    std::shared_ptr<dai::node::StereoDepth> stereoCamNode;
    std::shared_ptr<dai::node::NeuralDepth> neuralDepthNode;
    std::shared_ptr<dai::node::ImageAlign> alignNode;
    std::shared_ptr<ConfidenceMask> confMaskNode;
    dai::Platform platform;
    std::unique_ptr<RGBD> rgbdNodeLeft, rgbdNodeRight;
    std::shared_ptr<SensorWrapper> left, right;
    std::unique_ptr<BaseNode> featureTrackerLeftR, featureTrackerRightR, nnNodeLeft, nnNodeRight;
    std::unique_ptr<param_handlers::StereoParamHandler> ph;
    std::shared_ptr<dai::MessageQueue> leftRectQ, rightRectQ;
    std::shared_ptr<dai::InputQueue> neuralControl;
    std::string stereoQName, leftRectQName, rightRectQName, confidenceQName;
    dai::CameraFeatures leftSensInfo, rightSensInfo;
    bool aligned;
    dai::Node::Output* leftOut;
    dai::Node::Output* rightOut;
};

}  // namespace dai_nodes
}  // namespace depthai_ros_driver
