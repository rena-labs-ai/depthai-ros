#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "camera_info_manager/camera_info_manager.hpp"
#include "depthai/capabilities/ImgFrameCapability.hpp"
#include "depthai/common/CameraBoardSocket.hpp"
#include "depthai/device/Device.hpp"
#include "depthai/pipeline/MessageQueue.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/datatype/ImgFrame.hpp"
#include "depthai/pipeline/datatype/PointCloudData.hpp"
#include "depthai/pipeline/datatype/StereoDepthConfig.hpp"
#include "depthai/pipeline/node/Camera.hpp"
#include "depthai/pipeline/node/StereoDepth.hpp"
#include "depthai/pipeline/node/host/RGBD.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "depthai_bridge/PointCloudConverter.hpp"
#include "depthai_bridge/depthaiUtility.hpp"
#include "depthai_ros_driver_v3/dai_nodes/base_node.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/ir_alternator.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/sensor_helpers.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/sensor_wrapper.hpp"
#include "depthai_ros_driver_v3/param_handlers/base_param_handler.hpp"
#include "depthai_ros_driver_v3/param_handlers/pipeline_gen_param_handler.hpp"
#include "depthai_ros_driver_v3/param_handlers/rgbd_param_handler.hpp"
#include "depthai_ros_driver_v3/param_handlers/stereo_param_handler.hpp"
#include "depthai_ros_driver_v3/pipeline/base_pipeline.hpp"
#include "depthai_ros_driver_v3/pipeline/base_types.hpp"
#include "depthai_ros_driver_v3/utils.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/node.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace depthai_ros_driver {
namespace pipeline_gen {

namespace {

// Force a bool parameter to false, overriding any YAML override the user may
// have set. ROS applies overrides at declare_parameter time; so we declare,
// then explicitly set, to win over the YAML default.
void forceBoolParamFalse(std::shared_ptr<rclcpp::Node> node, const std::string& name) {
    if(!node->has_parameter(name)) {
        node->declare_parameter<bool>(name, false);
    }
    node->set_parameter(rclcpp::Parameter(name, false));
}

// Single host-side dispatcher node. Owns:
//   - The shared StereoDepth + RGBD nodes
//   - Eight publishers (left/right/stereo/pcl × dot_on/dot_off)
//   - Host queues for L mono, R mono, depth, and pcl
// Each queue's callback inspects the frame's sequence number, computes parity
// (with phase offset), and publishes to the appropriate per-branch topic.
class AlternateRouter : public dai_nodes::BaseNode {
   public:
    AlternateRouter(std::shared_ptr<rclcpp::Node> node,
                    std::shared_ptr<dai::Pipeline> pipeline,
                    std::shared_ptr<dai::Device> device,
                    bool rsCompat,
                    std::shared_ptr<param_handlers::StereoParamHandler> stereoPh,
                    std::shared_ptr<param_handlers::RGBDParamHandler> rgbdPh,
                    dai::Node::Output* leftMonoOut,
                    dai::Node::Output* rightMonoOut,
                    dai::Node::Output* rgbAlignOut,
                    dai::Node::Output* rgbColorOut,
                    dai::CameraBoardSocket leftSocket,
                    dai::CameraBoardSocket rightSocket,
                    dai::CameraBoardSocket alignSocket,
                    const std::string& thisNs,
                    const std::string& otherNs,
                    int phaseOffset)
        : BaseNode("alternate", node, pipeline, device->getDeviceName(), rsCompat),
          thisNs_(thisNs),
          otherNs_(otherNs),
          phaseOffset_(phaseOffset),
          stereoPh_(stereoPh),
          rgbdPh_(rgbdPh),
          leftMonoOut_(leftMonoOut),
          rightMonoOut_(rightMonoOut),
          leftSocket_(leftSocket),
          rightSocket_(rightSocket),
          alignSocket_(alignSocket) {
        stereoNode_ = pipeline->create<dai::node::StereoDepth>();
        stereoPh_->declareParams(stereoNode_);

        leftMonoOut_->link(stereoNode_->left);
        rightMonoOut_->link(stereoNode_->right);
        rgbAlignOut->link(stereoNode_->inputAlignTo);
        stereoNode_->inputAlignTo.setBlocking(false);

        rgbdNode_ = pipeline->create<dai::node::RGBD>()->build();
        rgbdPh_->declareParams(rgbdNode_, alignSocket_);
        rgbColorOut->link(rgbdNode_->inColor);
        stereoNode_->depth.link(rgbdNode_->inDepth);
    }

    ~AlternateRouter() override = default;

    void setNames() override {}
    void setInOut(std::shared_ptr<dai::Pipeline> /*pipeline*/) override {}

    void setupQueues(std::shared_ptr<dai::Device> device) override {
        using ParamNames = param_handlers::ParamNames;

        auto calHandler = device->readCalibration();

        int monoW = stereoPh_->getParam<int>(ParamNames::WIDTH);
        int monoH = stereoPh_->getParam<int>(ParamNames::HEIGHT);
        int qSize = std::max(stereoPh_->getParam<int>(ParamNames::MAX_Q_SIZE), 2);
        int pclSize = std::max(rgbdPh_->getParam<int>(ParamNames::MAX_Q_SIZE), 2);

        // Left mono
        setupImagePair(
            device, calHandler, monoW, monoH,
            leftMonoOut_, qSize,
            leftSocket_, dai_nodes::sensor_helpers::getNodeName(getROSNode(), dai_nodes::sensor_helpers::NodeNameEnum::Left),
            /*rectified=*/false, /*isStereoDepth=*/false,
            leftPair_);

        // Right mono
        setupImagePair(
            device, calHandler, monoW, monoH,
            rightMonoOut_, qSize,
            rightSocket_, dai_nodes::sensor_helpers::getNodeName(getROSNode(), dai_nodes::sensor_helpers::NodeNameEnum::Right),
            /*rectified=*/false, /*isStereoDepth=*/false,
            rightPair_);

        // Stereo depth (aligned to RGB, so use alignSocket frame)
        setupImagePair(
            device, calHandler, monoW, monoH,
            &stereoNode_->depth, qSize,
            alignSocket_, dai_nodes::sensor_helpers::getNodeName(getROSNode(), dai_nodes::sensor_helpers::NodeNameEnum::Stereo),
            /*rectified=*/true, /*isStereoDepth=*/true,
            depthPair_);

        // PCL
        setupPclPair(device, calHandler, &rgbdNode_->pcl, pclSize);
    }

    void closeQueues() override {
        for(auto* p : {&leftPair_, &rightPair_, &depthPair_}) {
            if(p->q) {
                p->q->removeCallback(p->cbId);
                p->q->close();
            }
        }
        if(pclPair_.q) {
            pclPair_.q->removeCallback(pclPair_.cbId);
            pclPair_.q->close();
        }
    }

   private:
    struct ImagePair {
        std::string baseName;
        std::shared_ptr<depthai_bridge::ImageConverter> conv;
        sensor_msgs::msg::CameraInfo info;
        image_transport::CameraPublisher thisPub;
        image_transport::CameraPublisher otherPub;
        std::shared_ptr<dai::MessageQueue> q;
        int cbId = -1;
    };
    struct PclPair {
        std::unique_ptr<depthai_bridge::PointCloudConverter> conv;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr otherPub;
        std::shared_ptr<dai::MessageQueue> q;
        int cbId = -1;
    };

    void setupImagePair(std::shared_ptr<dai::Device> device,
                        const dai::CalibrationHandler& calHandler,
                        int width,
                        int height,
                        dai::Node::Output* src,
                        int qSize,
                        dai::CameraBoardSocket socket,
                        const std::string& baseName,
                        bool rectified,
                        bool isStereoDepth,
                        ImagePair& pair) {
        pair.baseName = baseName;
        std::string sockName = depthai_bridge::getSocketName(socket, getDeviceName(), rsCompatibilityMode());
        std::string tfPrefix = getOpticalFrameName(sockName);

        pair.conv = std::make_shared<depthai_bridge::ImageConverter>(tfPrefix, false, false);
        pair.conv->setUpdateRosBaseTimeOnToRosMsg(false);
        if(isStereoDepth) {
            try {
                double baseline = calHandler.getBaselineDistance(leftSocket_, rightSocket_, false);
                pair.conv->convertDispToDepth(baseline);
            } catch(const std::exception& e) {
                RCLCPP_WARN(getLogger(), "Failed to compute baseline for depth conversion: %s", e.what());
            }
        }

        pair.info = dai_nodes::sensor_helpers::getCalibInfo(getLogger(), pair.conv, calHandler, socket, width, height);
        if(rectified) {
            std::fill(pair.info.d.begin(), pair.info.d.end(), 0.0);
            pair.info.r[0] = pair.info.r[4] = pair.info.r[8] = 1.0;
        }

        std::string suffix = rsCompatibilityMode() ? "/image_rect_raw" : (rectified ? "/image_raw" : "/image_raw");
        std::string thisTopic = std::string("~/") + thisNs_ + "/" + baseName + suffix;
        std::string otherTopic = std::string("~/") + otherNs_ + "/" + baseName + suffix;
        pair.thisPub = image_transport::create_camera_publisher(getROSNode().get(), thisTopic);
        pair.otherPub = image_transport::create_camera_publisher(getROSNode().get(), otherTopic);

        pair.q = src->createOutputQueue(qSize, false);
        pair.cbId = pair.q->addCallback([this, &pair](const std::shared_ptr<dai::ADatatype>& data) {
            this->onImage(data, pair);
        });
    }

    void setupPclPair(std::shared_ptr<dai::Device> /*device*/,
                      const dai::CalibrationHandler& /*calHandler*/,
                      dai::Node::Output* src,
                      int qSize) {
        std::string sockName = depthai_bridge::getSocketName(alignSocket_, getDeviceName(), rsCompatibilityMode());
        std::string tfPrefix = getOpticalFrameName(sockName);

        pclPair_.conv = std::make_unique<depthai_bridge::PointCloudConverter>(tfPrefix, false);
        pclPair_.conv->setUpdateRosBaseTimeOnToRosMsg(false);
        pclPair_.conv->setDepthUnit(dai::StereoDepthConfig::AlgorithmControl::DepthUnit::METER);

        rclcpp::PublisherOptions po;
        po.qos_overriding_options = rclcpp::QosOverridingOptions();
        pclPair_.thisPub = getROSNode()->create_publisher<sensor_msgs::msg::PointCloud2>(
            std::string("~/") + thisNs_ + "/rgbd/points", qSize, po);
        pclPair_.otherPub = getROSNode()->create_publisher<sensor_msgs::msg::PointCloud2>(
            std::string("~/") + otherNs_ + "/rgbd/points", qSize, po);

        pclPair_.q = src->createOutputQueue(qSize, false);
        pclPair_.cbId = pclPair_.q->addCallback([this](const std::shared_ptr<dai::ADatatype>& data) {
            this->onPcl(data);
        });
    }

    void onImage(const std::shared_ptr<dai::ADatatype>& data, ImagePair& pair) {
        if(!rclcpp::ok()) return;
        auto img = std::dynamic_pointer_cast<dai::ImgFrame>(data);
        if(!img) return;
        bool isThis = (((static_cast<int64_t>(img->getSequenceNum()) + phaseOffset_) % 2) == 0);
        auto& pub = isThis ? pair.thisPub : pair.otherPub;
        if(pub.getNumSubscribers() == 0) return;
        auto rawMsg = pair.conv->toRosMsgRawPtr(img);
        sensor_msgs::msg::CameraInfo info = pair.info;
        info.header = rawMsg.header;
        pub.publish(rawMsg, info);
    }

    void onPcl(const std::shared_ptr<dai::ADatatype>& data) {
        if(!rclcpp::ok()) return;
        auto pcl = std::dynamic_pointer_cast<dai::PointCloudData>(data);
        if(!pcl) return;
        bool isThis = (((static_cast<int64_t>(pcl->getSequenceNum()) + phaseOffset_) % 2) == 0);
        auto& pub = isThis ? pclPair_.thisPub : pclPair_.otherPub;
        if(pub->get_subscription_count() == 0 && pub->get_intra_process_subscription_count() == 0) return;
        std::deque<sensor_msgs::msg::PointCloud2> deq;
        pclPair_.conv->toRosMsg(pcl, deq);
        while(!deq.empty()) {
            pub->publish(deq.front());
            deq.pop_front();
        }
    }

    std::string thisNs_;
    std::string otherNs_;
    int phaseOffset_;
    std::shared_ptr<param_handlers::StereoParamHandler> stereoPh_;
    std::shared_ptr<param_handlers::RGBDParamHandler> rgbdPh_;
    std::shared_ptr<dai::node::StereoDepth> stereoNode_;
    std::shared_ptr<dai::node::RGBD> rgbdNode_;
    dai::Node::Output* leftMonoOut_;
    dai::Node::Output* rightMonoOut_;
    dai::CameraBoardSocket leftSocket_;
    dai::CameraBoardSocket rightSocket_;
    dai::CameraBoardSocket alignSocket_;
    ImagePair leftPair_;
    ImagePair rightPair_;
    ImagePair depthPair_;
    PclPair pclPair_;
};

}  // namespace

std::vector<std::unique_ptr<dai_nodes::BaseNode>> RGBDAlternate::createPipeline(std::shared_ptr<rclcpp::Node> node,
                                                                                std::shared_ptr<dai::Device> device,
                                                                                std::shared_ptr<dai::Pipeline> pipeline,
                                                                                std::shared_ptr<param_handlers::PipelineGenParamHandler> ph,
                                                                                const std::string& deviceName,
                                                                                bool rsCompat,
                                                                                const std::string& /*nnType*/) {
    using namespace dai_nodes::sensor_helpers;
    using ParamNames = param_handlers::ParamNames;

    if(!ph->getParam<bool>("i_enable_rgbd")) {
        RCLCPP_WARN(node->get_logger(), "rgbd_alternate selected but pipeline_gen.i_enable_rgbd=false; nothing to publish.");
    }

    auto leftSocket = dai::CameraBoardSocket::CAM_B;
    auto rightSocket = dai::CameraBoardSocket::CAM_C;
    auto alignSocket = dai::CameraBoardSocket::CAM_A;

    auto thisNs = node->get_parameter("driver.this.namespace").as_string();
    auto otherNs = node->get_parameter("driver.other.namespace").as_string();
    auto thisLaser = static_cast<float>(node->get_parameter("driver.this.r_laser_dot_intensity").as_double());
    auto thisFlood = static_cast<float>(node->get_parameter("driver.this.r_floodlight_intensity").as_double());
    auto otherLaser = static_cast<float>(node->get_parameter("driver.other.r_laser_dot_intensity").as_double());
    auto otherFlood = static_cast<float>(node->get_parameter("driver.other.r_floodlight_intensity").as_double());
    auto phaseOffset = static_cast<int>(node->get_parameter("driver.i_alternate_phase_offset").as_int());

    // L/R raw publishers are owned by AlternateRouter (per-branch namespaced).
    // Force off any user-set i_publish_topic on the L/R sensor wrappers.
    forceBoolParamFalse(node, getNodeName(node, NodeNameEnum::Left) + ".i_publish_topic");
    forceBoolParamFalse(node, getNodeName(node, NodeNameEnum::Right) + ".i_publish_topic");

    std::vector<std::unique_ptr<dai_nodes::BaseNode>> daiNodes;

    auto rgb = std::make_unique<dai_nodes::SensorWrapper>(getNodeName(node, NodeNameEnum::RGB), node, pipeline, deviceName, rsCompat, alignSocket);
    auto left =
        std::make_unique<dai_nodes::SensorWrapper>(getNodeName(node, NodeNameEnum::Left), node, pipeline, deviceName, rsCompat, leftSocket, false);
    auto right =
        std::make_unique<dai_nodes::SensorWrapper>(getNodeName(node, NodeNameEnum::Right), node, pipeline, deviceName, rsCompat, rightSocket, false);

    auto leftCam = left->getUnderlyingNode();
    auto rightCam = right->getUnderlyingNode();
    auto rgbCam = rgb->getUnderlyingNode();

    int monoW = node->get_parameter(getNodeName(node, NodeNameEnum::Left) + "." + ParamNames::WIDTH).as_int();
    int monoH = node->get_parameter(getNodeName(node, NodeNameEnum::Left) + "." + ParamNames::HEIGHT).as_int();
    auto monoFps = static_cast<float>(node->get_parameter(getNodeName(node, NodeNameEnum::Left) + "." + ParamNames::FPS).as_double());

    auto* leftOut = leftCam->requestOutput(std::pair<int, int>(monoW, monoH), dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::CROP, monoFps);
    auto* rightOut = rightCam->requestOutput(std::pair<int, int>(monoW, monoH), dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::CROP, monoFps);

    auto ira = std::make_unique<dai_nodes::IrAlternator>(
        "ir_alternator",
        node,
        pipeline,
        deviceName,
        rsCompat,
        dai_nodes::IrAlternator::BranchIntensities{thisLaser, thisFlood},
        dai_nodes::IrAlternator::BranchIntensities{otherLaser, otherFlood},
        phaseOffset);
    leftOut->link(ira->getTickInput());

    auto stereoPh = std::make_shared<param_handlers::StereoParamHandler>(node, "stereo", deviceName, rsCompat);
    stereoPh->updateSocketsFromParams(leftSocket, rightSocket, alignSocket);
    auto rgbdPh = std::make_shared<param_handlers::RGBDParamHandler>(node, "rgbd", deviceName, rsCompat);

    auto rgbW = node->get_parameter(getNodeName(node, NodeNameEnum::RGB) + "." + ParamNames::WIDTH).as_int();
    auto rgbH = node->get_parameter(getNodeName(node, NodeNameEnum::RGB) + "." + ParamNames::HEIGHT).as_int();
    auto rgbFps = static_cast<float>(node->get_parameter(getNodeName(node, NodeNameEnum::RGB) + "." + ParamNames::FPS).as_double());
    auto* rgbColorOut = rgbCam->requestOutput(std::pair<int, int>(rgbW, rgbH), dai::ImgFrame::Type::RGB888i, dai::ImgResizeMode::CROP, rgbFps, true);
    auto* rgbAlignOut = rgb->getDefaultOut();

    auto router = std::make_unique<AlternateRouter>(
        node, pipeline, device, rsCompat, stereoPh, rgbdPh,
        leftOut, rightOut, rgbAlignOut, rgbColorOut,
        leftSocket, rightSocket, alignSocket,
        thisNs, otherNs, phaseOffset);

    daiNodes.push_back(std::move(rgb));
    daiNodes.push_back(std::move(left));
    daiNodes.push_back(std::move(right));
    daiNodes.push_back(std::move(ira));
    daiNodes.push_back(std::move(router));
    return daiNodes;
}

}  // namespace pipeline_gen
}  // namespace depthai_ros_driver

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(depthai_ros_driver::pipeline_gen::RGBDAlternate, depthai_ros_driver::pipeline_gen::BasePipeline)
