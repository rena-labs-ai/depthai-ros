#include "depthai_ros_driver_v3/dai_nodes/sensors/stereo.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "depthai/capabilities/ImgFrameCapability.hpp"
#include "depthai/device/DeviceBase.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/datatype/ADatatype.hpp"
#include "depthai/pipeline/datatype/ImgFrame.hpp"
#include "depthai/pipeline/node/NeuralDepth.hpp"
#include "depthai/pipeline/node/StereoDepth.hpp"
#include "depthai_ros_driver_v3/dai_nodes/nn/nn_helpers.hpp"
#include "depthai_ros_driver_v3/dai_nodes/nn/spatial_nn_wrapper.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/confidence_mask.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/feature_tracker.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/img_pub.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/rgbd.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/sensor_helpers.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/sensor_wrapper.hpp"
#include "depthai_ros_driver_v3/param_handlers/base_param_handler.hpp"
#include "depthai_ros_driver_v3/param_handlers/stereo_param_handler.hpp"
#include "depthai_ros_driver_v3/utils.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "rclcpp/node.hpp"

namespace depthai_ros_driver {
namespace dai_nodes {
Stereo::Stereo(const std::string& daiNodeName,
               std::shared_ptr<rclcpp::Node> node,
               std::shared_ptr<dai::Pipeline> pipeline,
               std::shared_ptr<dai::Device> device,
               bool rsCompat,
               dai::CameraBoardSocket leftSocket,
               dai::CameraBoardSocket rightSocket)
    : BaseNode(daiNodeName, node, pipeline, device->getDeviceName(), rsCompat) {
    using ParamNames = param_handlers::ParamNames;
    RCLCPP_DEBUG(getLogger(), "Creating node %s", daiNodeName.c_str());
    setNames();
    platform = device->getPlatform();
    ph = std::make_unique<param_handlers::StereoParamHandler>(node, daiNodeName, device->getDeviceName(), rsCompat);
    if(ph->getParam<bool>("i_use_neural_depth") && (platform == dai::Platform::RVC2)) {
        throw std::runtime_error("Neural depth is not supported on RVC2");
    }
    auto alignSocket = dai::CameraBoardSocket::CAM_A;
    if(device->getDeviceName() == "OAK-D-SR" || device->getDeviceName() == "OAK-D-SR-POE") {
        alignSocket = dai::CameraBoardSocket::CAM_C;
    }
    ph->updateSocketsFromParams(leftSocket, rightSocket, alignSocket);
    auto features = device->getConnectedCameraFeatures();
    for(auto f : features) {
        if(f.socket == leftSocket) {
            leftSensInfo = f;
            leftSensInfo.name = getSocketName(leftSocket);
        } else if(f.socket == rightSocket) {
            rightSensInfo = f;
            rightSensInfo.name = getSocketName(rightSocket);
        } else {
            continue;
        }
    }
    RCLCPP_DEBUG(getLogger(),
                 "Creating stereo node with left sensor %s and right sensor %s",
                 getSocketName(leftSensInfo.socket).c_str(),
                 getSocketName(rightSensInfo.socket).c_str());
    left = std::make_shared<SensorWrapper>(getSocketName(leftSensInfo.socket), node, pipeline, device->getDeviceName(), rsCompat, leftSensInfo.socket, false);
    right =
        std::make_shared<SensorWrapper>(getSocketName(rightSensInfo.socket), node, pipeline, device->getDeviceName(), rsCompat, rightSensInfo.socket, false);
    if(ph->getParam<bool>("i_use_neural_depth")) {
        neuralDepthNode = pipeline->create<dai::node::NeuralDepth>();
        ph->declareParams(neuralDepthNode);
        leftOut = left->getUnderlyingNode()->requestFullResolutionOutput();
        rightOut = right->getUnderlyingNode()->requestFullResolutionOutput();
        neuralDepthNode->build(*leftOut, *rightOut, ph->getModel());
    } else {
        stereoCamNode = pipeline->create<dai::node::StereoDepth>();
        ph->declareParams(stereoCamNode);
        leftOut =
            left->getUnderlyingNode()->requestOutput(std::make_pair<int, int>(ph->getParam<int>(ParamNames::WIDTH), ph->getParam<int>(ParamNames::HEIGHT)),
                                                     std::nullopt,
                                                     dai::ImgResizeMode::CROP,
                                                     ph->getParam<float>(ParamNames::FPS));
        rightOut =
            right->getUnderlyingNode()->requestOutput(std::make_pair<int, int>(ph->getParam<int>(ParamNames::WIDTH), ph->getParam<int>(ParamNames::HEIGHT)),
                                                      std::nullopt,
                                                      dai::ImgResizeMode::CROP,
                                                      ph->getParam<float>(ParamNames::FPS));
        leftOut->link(stereoCamNode->left);
        rightOut->link(stereoCamNode->right);
    }

    aligned = ph->getParam<bool>(param_handlers::ParamNames::ALIGNED);
    if(ph->getParam<bool>("i_enable_left_spatial_nn")) {
        nnNodeLeft = std::make_unique<SpatialNNWrapper>(
            getName() + "_" + left->getName() + "_spatial_nn", getROSNode(), pipeline, device->getDeviceName(), rsCompat, *left, *this);
    }
    if(ph->getParam<bool>("i_enable_right_spatial_nn")) {
        nnNodeRight = std::make_unique<SpatialNNWrapper>(
            getName() + "_" + right->getName() + "_spatial_nn", getROSNode(), pipeline, device->getDeviceName(), rsCompat, *right, *this);
    }
    if(ph->getParam<bool>("i_enable_left_rgbd")) {
        rgbdNodeLeft = std::make_unique<dai_nodes::RGBD>(
            getName() + "_" + left->getName() + "_rgbd", node, pipeline, device, rsCompat, *left, getUnderlyingNode(), aligned);
    }
    if(ph->getParam<bool>("i_enable_right_rgbd")) {
        rgbdNodeRight = std::make_unique<dai_nodes::RGBD>(
            getName() + "_" + right->getName() + "_rgbd", node, pipeline, device, rsCompat, *right, getUnderlyingNode(), aligned);
    }

    // Optional confidence mask (classic StereoDepth only). Zeroes depth where the
    // confidence map is below threshold. It MUST run in the rectified frame
    // (depth + confidenceMap share that frame), so it sits BEFORE any alignment;
    // the masked depth is then aligned downstream exactly like raw depth.
    bool maskEnabled = ph->getParam<bool>("i_enable_confidence_mask") && !ph->getParam<bool>("i_use_neural_depth");
    if(maskEnabled) {
        confMaskNode = pipeline->create<ConfidenceMask>();
        // Reuse the stereo confidence threshold (same scale as the confidence
        // map, higher = stricter). The mask hard-guarantees it on the output,
        // regardless of upstream post-filters re-adding sub-threshold pixels.
        confMaskNode->setThreshold(ph->getParam<int>("i_stereo_conf_threshold"));
        stereoCamNode->depth.link(confMaskNode->inDepth);
        stereoCamNode->confidenceMap.link(confMaskNode->inConf);
    }
    // Depth source feeding alignment / publishing: the masked depth if enabled.
    dai::Node::Output* depthSrc;
    if(confMaskNode) {
        depthSrc = &confMaskNode->out;
    } else if(ph->getParam<bool>("i_use_neural_depth")) {
        depthSrc = &neuralDepthNode->depth;
    } else {
        depthSrc = &stereoCamNode->depth;
    }

    // Check alignment, if board socket is one of the pairs, align.
    // if not it should be aligned externally by calling align method in pipeline creation
    auto socketID = ph->getSocketID();
    if(aligned) {
        // Use an explicit ImageAlign node on RVC4, OR whenever masking is on: that
        // lets the mask run pre-alignment in the rectified frame and aligns the
        // MASKED depth here, instead of letting StereoDepth align internally
        // (which would leave the confidence map in a different frame). RVC2 has no
        // on-device ImageAlign, so run it on host there.
        bool explicitAlign = (platform == dai::Platform::RVC4) || maskEnabled;
        if(explicitAlign) {
            alignNode = pipeline->create<dai::node::ImageAlign>();
            alignNode->setRunOnHost(platform == dai::Platform::RVC4 ? ph->getParam<bool>("i_run_align_on_host") : true);
            depthSrc->link(alignNode->input);
            alignNode->input.setBlocking(false);
            alignNode->inputAlignTo.setBlocking(false);
        }
        if(socketID == leftSensInfo.socket) {
            leftOut->link(getInput(static_cast<int>(link_types::StereoLinkType::align)));
        } else if(socketID == rightSensInfo.socket) {
            rightOut->link(getInput(static_cast<int>(link_types::StereoLinkType::align)));
        } else {
            RCLCPP_DEBUG(getLogger(), "Socket aligned to a different ID: %d, make sure you call align method in pipeline creation", static_cast<int>(socketID));
        }
    }
    setInOut(pipeline);
    RCLCPP_DEBUG(getLogger(), "Node %s created", daiNodeName.c_str());
}
Stereo::~Stereo() = default;
void Stereo::setNames() {
    stereoQName = getName() + "_stereo";
    leftRectQName = getName() + "_left_rect";
    rightRectQName = getName() + "_right_rect";
    confidenceQName = getName() + "_confidence";
}

std::shared_ptr<dai::node::StereoDepth> Stereo::getUnderlyingNode() {
    if(ph->getParam<bool>("i_use_neural_depth")) {
        return nullptr;
    }
    return stereoCamNode;
}

std::shared_ptr<dai::node::NeuralDepth> Stereo::getNeuralDepthNode() {
    if(!ph->getParam<bool>("i_use_neural_depth")) {
        return nullptr;
    }
    return neuralDepthNode;
}

bool Stereo::isAligned() {
    return aligned;
}

void Stereo::setInOut(std::shared_ptr<dai::Pipeline> pipeline) {
    bool outputDisparity = ph->getParam<bool>("i_output_disparity");
    bool lowBandwidth = ph->getParam<bool>("i_low_bandwidth");
    if(ph->getParam<bool>("i_publish_topic")) {
        utils::VideoEncoderConfig encConf;
        encConf.profile = static_cast<dai::VideoEncoderProperties::Profile>(ph->getParam<int>("i_low_bandwidth_profile"));
        encConf.bitrate = ph->getParam<int>("i_low_bandwidth_bitrate");
        encConf.frameFreq = ph->getParam<int>("i_low_bandwidth_frame_freq");
        encConf.quality = ph->getParam<int>("i_low_bandwidth_quality");
        encConf.enabled = lowBandwidth;

        if(outputDisparity || lowBandwidth) {
            if(ph->getParam<bool>("i_use_neural_depth")) {
                stereoPub = setupOutput(pipeline, stereoQName, &neuralDepthNode->disparity, ph->getParam<bool>("i_synced"), encConf);
            } else {
                stereoPub = setupOutput(pipeline, stereoQName, &stereoCamNode->disparity, ph->getParam<bool>("i_synced"), encConf);
            }
        } else {
            // Select the depth to publish (mask + align nodes are built in the
            // constructor): aligned output if an align node exists (masked-then-
            // aligned, or plain aligned on RVC4); else masked depth; else raw depth.
            dai::Node::Output* depthOut;
            if(aligned && alignNode) {
                depthOut = &alignNode->outputAligned;
            } else if(confMaskNode) {
                depthOut = &confMaskNode->out;
            } else if(ph->getParam<bool>("i_use_neural_depth")) {
                depthOut = &neuralDepthNode->depth;
            } else {
                depthOut = &stereoCamNode->depth;
            }
            stereoPub = setupOutput(pipeline, stereoQName, depthOut, ph->getParam<bool>("i_synced"), encConf);
        }
    }

    // Debug: publish the raw confidence map (RAW8, rectified frame) so the mask
    // can be verified per-pixel against the depth. confidenceMap exists only on
    // classic StereoDepth (not NeuralDepth) and is unaligned, so pixel-comparison
    // to depth is valid only with i_aligned:false.
    if(ph->getParam<bool>("i_publish_confidence") && !ph->getParam<bool>("i_use_neural_depth")) {
        utils::VideoEncoderConfig encConf;
        encConf.enabled = false;
        confidencePub = setupOutput(pipeline, confidenceQName, &stereoCamNode->confidenceMap, ph->getParam<bool>("i_synced"), encConf);
    }

    if(ph->getParam<bool>("i_left_rect_publish_topic")) {
        utils::VideoEncoderConfig encConf;
        encConf.profile = static_cast<dai::VideoEncoderProperties::Profile>(ph->getParam<int>("i_left_rect_low_bandwidth_profile"));
        encConf.bitrate = ph->getParam<int>("i_left_rect_low_bandwidth_bitrate");
        encConf.frameFreq = ph->getParam<int>("i_left_rect_low_bandwidth_frame_freq");
        encConf.quality = ph->getParam<int>("i_left_rect_low_bandwidth_quality");
        encConf.enabled = ph->getParam<bool>("i_left_rect_low_bandwidth");

        if(ph->getParam<bool>("i_use_neural_depth")) {
            leftRectPub = setupOutput(pipeline, leftRectQName, &neuralDepthNode->rectifiedLeft, ph->getParam<bool>("i_left_rect_synced"), encConf);
        } else {
            leftRectPub = setupOutput(pipeline, leftRectQName, &stereoCamNode->rectifiedLeft, ph->getParam<bool>("i_left_rect_synced"), encConf);
        }
    }

    if(ph->getParam<bool>("i_right_rect_publish_topic")) {
        utils::VideoEncoderConfig encConf;
        encConf.profile = static_cast<dai::VideoEncoderProperties::Profile>(ph->getParam<int>("i_right_rect_low_bandwidth_profile"));
        encConf.bitrate = ph->getParam<int>("i_right_rect_low_bandwidth_bitrate");
        encConf.frameFreq = ph->getParam<int>("i_right_rect_low_bandwidth_frame_freq");
        encConf.quality = ph->getParam<int>("i_right_rect_low_bandwidth_quality");
        encConf.enabled = ph->getParam<bool>("i_right_rect_low_bandwidth");
        if(ph->getParam<bool>("i_use_neural_depth")) {
            rightRectPub = setupOutput(pipeline, rightRectQName, &neuralDepthNode->rectifiedRight, ph->getParam<bool>("i_right_rect_synced"), encConf);
        } else {
            rightRectPub = setupOutput(pipeline, rightRectQName, &stereoCamNode->rectifiedRight, ph->getParam<bool>("i_right_rect_synced"), encConf);
        }
    }

    if(ph->getParam<bool>("i_left_rect_enable_feature_tracker")) {
        featureTrackerLeftR = std::make_unique<FeatureTracker>(
            leftSensInfo.name + std::string("_rect_feature_tracker"), getROSNode(), pipeline, getDeviceName(), rsCompatibilityMode());
        auto in = featureTrackerLeftR->getInput();
        if(ph->getParam<bool>("i_use_neural_depth")) {
            neuralDepthNode->rectifiedLeft.link(in);
        } else {
            stereoCamNode->rectifiedLeft.link(in);
        }
    }

    if(ph->getParam<bool>("i_right_rect_enable_feature_tracker")) {
        featureTrackerRightR = std::make_unique<FeatureTracker>(
            rightSensInfo.name + std::string("_rect_feature_tracker"), getROSNode(), pipeline, getDeviceName(), rsCompatibilityMode());
        auto in = featureTrackerRightR->getInput();
        if(ph->getParam<bool>("i_use_neural_depth")) {
            neuralDepthNode->rectifiedRight.link(in);
        } else {
            stereoCamNode->rectifiedRight.link(in);
        }
    }
}

// The firmware rectifies BOTH streams into the LEFT camera's intrinsics, with
// R1/R2 re-derived from the extrinsics via cv::stereoRectify — the EEPROM-stored
// rectification rotations are not what the mesh uses, so an info derived from
// them misses the actual rectified geometry by several pixels. Derive the same
// recipe once here: the rect infos publish its result, and the raw infos carry
// it as their R/P so self-rectifying consumers land in the mesh geometry.
void Stereo::computeRectifyRecipe(std::shared_ptr<dai::Device> device) {
    auto calHandler = device->readCalibration();
    auto toCv = [](const std::vector<std::vector<float>>& m, int rows, int cols) {
        cv::Mat out(rows, cols, CV_64F);
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++) out.at<double>(i, j) = m[i][j];
        return out;
    };
    const int w = ph->getOtherNodeParam<int>(getSocketName(leftSensInfo.socket), "i_width");
    const int h = ph->getOtherNodeParam<int>(getSocketName(leftSensInfo.socket), "i_height");
    cv::Mat K1 = toCv(calHandler.getCameraIntrinsics(leftSensInfo.socket, w, h), 3, 3);
    cv::Mat K2 = toCv(calHandler.getCameraIntrinsics(rightSensInfo.socket, w, h), 3, 3);
    auto d1 = calHandler.getDistortionCoefficients(leftSensInfo.socket);
    auto d2 = calHandler.getDistortionCoefficients(rightSensInfo.socket);
    // The firmware mesh uses only the first 8 distortion coefficients
    // (StereoDepthProperties), so truncating a 14-coeff perspective model
    // here keeps this computation identical to the mesh. Fisheye sensors
    // return just 4 coefficients — copy what exists, zero-pad the rest.
    cv::Mat D1 = cv::Mat::zeros(1, 8, CV_64F), D2 = cv::Mat::zeros(1, 8, CV_64F);
    for(size_t i = 0; i < 8 && i < d1.size(); i++) D1.at<double>(i) = d1[i];
    for(size_t i = 0; i < 8 && i < d2.size(); i++) D2.at<double>(i) = d2[i];
    auto ext = calHandler.getCameraExtrinsics(leftSensInfo.socket, rightSensInfo.socket);
    cv::Mat R(3, 3, CV_64F), T(3, 1, CV_64F);
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) R.at<double>(i, j) = ext[i][j];
        T.at<double>(i) = ext[i][3] / 100.0;  // cm -> m
    }
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K1, D1, K2, D2, cv::Size(w, h), R, T, R1, R2, P1, P2, Q);
    rectifyKLeft = K1.clone();
    rectifyKRight = K2.clone();
    rectifyDLeft = D1.clone();
    rectifyDRight = D2.clone();

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            rectifyRLeft[i * 3 + j] = R1.at<double>(i, j);
            rectifyRRight[i * 3 + j] = R2.at<double>(i, j);
            rectifyPLeft[i * 4 + j] = K1.at<double>(i, j);
            rectifyPRight[i * 4 + j] = K1.at<double>(i, j);
        }
    }
    // Signed baseline from stereoRectify's own P2 (Tx flips with reversed
    // socket order), rescaled to the mesh camera's focal since our P carries
    // left-K rather than stereoRectify's projection.
    rectifyPRight[3] = K1.at<double>(0, 0) * (P2.at<double>(0, 3) / P2.at<double>(0, 0));
}

void Stereo::setupRectQueue(std::shared_ptr<dai::Device> device,
                            dai::CameraFeatures& sensorInfo,
                            std::shared_ptr<sensor_helpers::ImagePublisher> pub,
                            bool isLeft) {
    auto sensorName = getSocketName(sensorInfo.socket);
    // The rectified stream lives in the mesh camera's frame, rotated from the
    // physical sensor by R1/R2 — stamp it with its own optical frame; the
    // static TF below anchors it to the sensor frame.
    auto tfPrefix = getOpticalFrameName(sensorName + "_rect");
    utils::ImgConverterConfig convConfig;
    convConfig.tfPrefix = tfPrefix;
    convConfig.interleaved = false;
    convConfig.getBaseDeviceTimestamp = ph->getParam<bool>("i_get_base_device_timestamp");
    convConfig.updateROSBaseTimeOnRosMsg = ph->getParam<bool>("i_update_ros_base_time_on_ros_msg");
    convConfig.lowBandwidth = ph->getParam<bool>(isLeft ? "i_left_rect_low_bandwidth" : "i_right_rect_low_bandwidth");
    convConfig.encoding = dai::ImgFrame::Type::GRAY8;
    convConfig.addExposureOffset = ph->getParam<bool>(isLeft ? "i_left_rect_add_exposure_offset" : "i_right_rect_add_exposure_offset");
    convConfig.expOffset = static_cast<dai::CameraExposureOffset>(ph->getParam<int>(isLeft ? "i_left_rect_exposure_offset" : "i_right_rect_exposure_offset"));
    convConfig.reverseSocketOrder = ph->getParam<bool>("i_reverse_stereo_socket_order");

    utils::ImgPublisherConfig pubConfig;
    pubConfig.daiNodeName = sensorName;
    pubConfig.rectified = true;
    pubConfig.undistorted = true;
    pubConfig.width = ph->getOtherNodeParam<int>(sensorName, "i_width");
    pubConfig.height = ph->getOtherNodeParam<int>(sensorName, "i_height");
    // Own namespace: image_transport derives camera_info as the base topic's
    // sibling, so "~/left/image_rect" would collide with the raw publisher's
    // "~/left/camera_info" (raw and rectified calibration interleaving).
    pubConfig.topicName = "~/" + sensorName + "_rect";
    pubConfig.topicSuffix = rsCompatibilityMode() ? "/image_rect_raw" : "/image_rect";
    pubConfig.maxQSize = ph->getOtherNodeParam<int>(sensorName, "i_max_q_size");
    pubConfig.socket = sensorInfo.socket;
    pubConfig.infoMgrSuffix = "rect";
    pubConfig.publishCompressed = ph->getParam<bool>(isLeft ? "i_left_rect_publish_compressed" : "i_right_rect_publish_compressed");

    // The firmware rectifies BOTH streams into the LEFT camera's intrinsics
    // (see computeRectifyRecipe) — publish the mesh camera instead of an info
    // derived from the EEPROM-stored rotations (validated by raw->rect feature
    // reprojection at <1 px).
    {
        const auto& Rrect = isLeft ? rectifyRLeft : rectifyRRight;
        const auto& P = isLeft ? rectifyPLeft : rectifyPRight;
        sensor_msgs::msg::CameraInfo info;
        info.width = pubConfig.width;
        info.height = pubConfig.height;
        info.distortion_model = "plumb_bob";
        info.d.assign(8, 0.0);
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                info.k[i * 3 + j] = P[i * 4 + j];
                info.r[i * 3 + j] = Rrect[i * 3 + j];
            }
        }
        info.p = P;
        pubConfig.overrideInfo = info;
        pubConfig.hasOverrideInfo = true;

        // camera_info.r maps sensor-frame points into the rectified frame
        // (x_rect = R * x_cam), so the child frame's orientation in the
        // sensor frame is R^T.
        if(!rectTfBroadcaster) {
            rectTfBroadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(getROSNode());
        }
        tf2::Matrix3x3 rectRot(Rrect[0], Rrect[1], Rrect[2], Rrect[3], Rrect[4], Rrect[5], Rrect[6], Rrect[7], Rrect[8]);
        tf2::Quaternion q;
        rectRot.transpose().getRotation(q);
        geometry_msgs::msg::TransformStamped tfMsg;
        tfMsg.header.stamp = getROSNode()->get_clock()->now();
        tfMsg.header.frame_id = getOpticalFrameName(sensorName);
        tfMsg.child_frame_id = tfPrefix;
        tfMsg.transform.rotation.x = q.x();
        tfMsg.transform.rotation.y = q.y();
        tfMsg.transform.rotation.z = q.z();
        tfMsg.transform.rotation.w = q.w();
        rectTfBroadcaster->sendTransform(tfMsg);
    }

    pub->setup(device, convConfig, pubConfig);
}

void Stereo::setupWideRectQueues() {
    const int w = ph->getOtherNodeParam<int>(getSocketName(leftSensInfo.socket), "i_width");
    const int h = ph->getOtherNodeParam<int>(getSocketName(leftSensInfo.socket), "i_height");
    cv::Mat R1(3, 3, CV_64F), R2(3, 3, CV_64F);
    for(int i = 0; i < 9; i++) {
        R1.at<double>(i / 3, i % 3) = rectifyRLeft[i];
        R2.at<double>(i / 3, i % 3) = rectifyRRight[i];
    }
    // Where the raw content ends in the rectified frame, along the rectified
    // principal row/column (normalized coordinates). For a barrel lens the
    // outline bulges outward at the corners, so these four points bound the
    // largest rectangle fully inside the content. Found by bisection on the
    // forward model, exact for any distortion strength.
    auto extents = [&](const cv::Mat& K, const cv::Mat& D, const cv::Mat& R) {
        auto pixel = [&](double x, double y, int axis) {
            cv::Mat ray = R.t() * (cv::Mat_<double>(3, 1) << x, y, 1.0);
            std::vector<cv::Point3d> obj{cv::Point3d(ray.at<double>(0), ray.at<double>(1), ray.at<double>(2))};
            std::vector<cv::Point2d> img;
            cv::projectPoints(obj, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K, D, img);
            return axis == 0 ? img[0].x : img[0].y;
        };
        auto edge = [&](int axis, double target) {
            const double sign = target == 0.0 ? -1.0 : 1.0;
            double lo = 0.0, hi = 8.0;
            for(int i = 0; i < 200; i++) {
                const double m = (lo + hi) / 2;
                const double px = axis == 0 ? pixel(sign * m, 0.0, 0) : pixel(0.0, sign * m, 1);
                const bool inside = target == 0.0 ? px > 0.0 : px < target;
                (inside ? lo : hi) = m;
            }
            return sign * lo;
        };
        return std::array<double, 4>{edge(0, 0.0), edge(0, w - 1.0), edge(1, 0.0), edge(1, h - 1.0)};
    };
    const auto eL = extents(rectifyKLeft, rectifyDLeft, R1);
    const auto eR = extents(rectifyKRight, rectifyDRight, R2);
    const double xmin = std::max(eL[0], eR[0]), xmax = std::min(eL[1], eR[1]);
    const double ymin = std::max(eL[2], eR[2]), ymax = std::min(eL[3], eR[3]);
    const double margin = 0.01;
    const double fx = w / (xmax - xmin) * (1.0 - margin);
    const double fy = h / (ymax - ymin) * (1.0 - margin);
    const double cx = -xmin * fx + (w - fx * (xmax - xmin)) / 2.0;
    const double cy = -ymin * fy + (h - fy * (ymax - ymin)) / 2.0;
    const double baseline = -rectifyPRight[3] / rectifyPLeft[0];  // signed, from the recipe
    const double hfov = (std::atan((w - cx) / fx) + std::atan(cx / fx)) * 180.0 / M_PI;
    const double hfovMesh = (std::atan((w - rectifyPLeft[2]) / rectifyPLeft[0]) + std::atan(rectifyPLeft[2] / rectifyPLeft[0])) * 180.0 / M_PI;
    RCLCPP_INFO(getLogger(),
                "wide rect: %dx%d, HFOV %.0f deg (firmware rect %.0f deg), fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                w, h, hfov, hfovMesh, fx, fy, cx, cy);

    auto setupSide = [&](WideRect& wr, dai::Node::Output* out, const dai::CameraFeatures& sensorInfo, const cv::Mat& K, const cv::Mat& D,
                         const cv::Mat& R, bool isLeft) {
        const auto sensorName = getSocketName(sensorInfo.socket);
        cv::Mat P = (cv::Mat_<double>(3, 4) << fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0);
        if(!isLeft) P.at<double>(0, 3) = -fx * baseline;
        cv::initUndistortRectifyMap(K, D, R, P, cv::Size(w, h), CV_16SC2, wr.map1, wr.map2);

        wr.info = sensor_msgs::msg::CameraInfo();
        wr.info.width = w;
        wr.info.height = h;
        wr.info.distortion_model = "plumb_bob";
        wr.info.d.assign(8, 0.0);
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                wr.info.k[i * 3 + j] = P.at<double>(i, j);
                wr.info.r[i * 3 + j] = R.at<double>(i, j);
            }
            for(int j = 0; j < 4; j++) wr.info.p[i * 4 + j] = P.at<double>(i, j);
        }
        // Same rectified frame as the firmware rect (same R), so the static TF
        // published by setupRectQueue covers this stream too.
        wr.conv = std::make_shared<depthai_bridge::ImageConverter>(getOpticalFrameName(sensorName + "_rect"), false, ph->getParam<bool>("i_get_base_device_timestamp"));
        wr.conv->setUpdateRosBaseTimeOnToRosMsg(ph->getParam<bool>("i_update_ros_base_time_on_ros_msg"));
        if(ph->getParam<bool>("i_reverse_stereo_socket_order")) {
            wr.conv->reverseStereoSocketOrder();
        }
        wr.pub = image_transport::create_camera_publisher(getROSNode().get(), "~/" + sensorName + "_rect_wide/image_rect");
        wr.q = out->createOutputQueue(ph->getParam<int>(param_handlers::ParamNames::MAX_Q_SIZE), false);
        wr.cbId = wr.q->addCallback([this, isLeft](const std::shared_ptr<dai::ADatatype>& data) { publishWideRect(data, isLeft); });
    };
    setupSide(wideLeft, leftOut, leftSensInfo, rectifyKLeft, rectifyDLeft, R1, true);
    setupSide(wideRight, rightOut, rightSensInfo, rectifyKRight, rectifyDRight, R2, false);
}

void Stereo::publishWideRect(const std::shared_ptr<dai::ADatatype>& data, bool isLeft) {
    auto& wr = isLeft ? wideLeft : wideRight;
    if(wr.pub.getNumSubscribers() == 0) {
        return;
    }
    auto frame = std::dynamic_pointer_cast<dai::ImgFrame>(data);
    if(!frame) {
        return;
    }
    // The converter owns timestamp/frame_id semantics (device->ROS time base,
    // socket order); we only re-sample its pixels.
    auto raw = wr.conv->toRosMsgRawPtr(frame);
    cv_bridge::CvImagePtr gray;
    try {
        gray = cv_bridge::toCvCopy(raw, sensor_msgs::image_encodings::MONO8);
    } catch(const std::exception& e) {
        if(!wr.warned) {
            RCLCPP_WARN(getLogger(), "wide rect: cannot convert %s frames to mono8: %s", raw.encoding.c_str(), e.what());
            wr.warned = true;
        }
        return;
    }
    if(gray->image.cols != wr.map1.cols || gray->image.rows != wr.map1.rows) {
        if(!wr.warned) {
            RCLCPP_WARN(getLogger(), "wide rect: frame %dx%d does not match the %dx%d recipe; not publishing", gray->image.cols, gray->image.rows, wr.map1.cols, wr.map1.rows);
            wr.warned = true;
        }
        return;
    }
    cv::Mat rect;
    cv::remap(gray->image, rect, wr.map1, wr.map2, cv::INTER_LINEAR);
    auto msg = cv_bridge::CvImage(raw.header, sensor_msgs::image_encodings::MONO8, rect).toImageMsg();
    wr.info.header = raw.header;
    wr.pub.publish(*msg, wr.info);
}

void Stereo::setupLeftRectQueue(std::shared_ptr<dai::Device> device) {
    setupRectQueue(device, leftSensInfo, leftRectPub, true);
}

void Stereo::setupRightRectQueue(std::shared_ptr<dai::Device> device) {
    setupRectQueue(device, rightSensInfo, rightRectPub, false);
}

void Stereo::setupStereoQueue(std::shared_ptr<dai::Device> device) {
    using param_handlers::ParamNames;
    std::string tfPrefix;
    tfPrefix = getOpticalFrameName(ph->getParam<std::string>("i_socket_name"));
    utils::ImgConverterConfig convConfig;
    convConfig.getBaseDeviceTimestamp = ph->getParam<bool>(ParamNames::GET_BASE_DEVICE_TIMESTAMP);
    convConfig.tfPrefix = tfPrefix;
    convConfig.updateROSBaseTimeOnRosMsg = ph->getParam<bool>(ParamNames::UPDATE_ROS_BASE_TIME_ON_ROS_MSG);
    convConfig.lowBandwidth = ph->getParam<bool>(ParamNames::LOW_BANDWIDTH);
    convConfig.encoding = dai::ImgFrame::Type::RAW8;
    convConfig.addExposureOffset = ph->getParam<bool>(ParamNames::ADD_EXPOSURE_OFFSET);
    convConfig.expOffset = static_cast<dai::CameraExposureOffset>(ph->getParam<int>(ParamNames::EXPOSURE_OFFSET));
    convConfig.reverseSocketOrder = ph->getParam<bool>(ParamNames::REVERSE_STEREO_SOCKET_ORDER);
    convConfig.alphaScalingEnabled = ph->getParam<bool>("i_enable_alpha_scaling");
    if(convConfig.alphaScalingEnabled) {
        convConfig.alphaScaling = ph->getParam<double>("i_alpha_scaling");
    }
    convConfig.outputDisparity = ph->getParam<bool>("i_output_disparity");
    convConfig.isStereo = true;

    utils::ImgPublisherConfig pubConf;
    pubConf.daiNodeName = getName();
    pubConf.topicName = "~/" + getName();
    pubConf.topicSuffix = rsCompatibilityMode() ? "/image_rect_raw" : "/image_raw";
    pubConf.rectified = !convConfig.alphaScalingEnabled;
    pubConf.undistorted = !convConfig.alphaScalingEnabled;
    pubConf.width = ph->getParam<int>(ParamNames::WIDTH);
    pubConf.height = ph->getParam<int>(ParamNames::HEIGHT);
    pubConf.socket = ph->getSocketID();
    pubConf.calibrationFile = ph->getParam<std::string>(ParamNames::CALIBRATION_FILE);
    pubConf.leftSocket = leftSensInfo.socket;
    pubConf.rightSocket = rightSensInfo.socket;
    pubConf.lazyPub = ph->getParam<bool>(ParamNames::ENABLE_LAZY_PUBLISHER);
    pubConf.maxQSize = ph->getParam<int>(ParamNames::MAX_Q_SIZE);
    pubConf.publishCompressed = ph->getParam<bool>(ParamNames::PUBLISH_COMPRESSED);
    pubConf.logLatency = ph->getParam<bool>("i_log_latency");
    pubConf.pngLevel = ph->getParam<int>("i_depth_png_level");
    pubConf.enableCompressed = ph->getParam<bool>("i_enable_compressed");

    stereoPub->setup(device, convConfig, pubConf);
    if(ph->getParam<bool>("i_use_neural_depth")) {
        neuralControl = neuralDepthNode->inputConfig.createInputQueue();
    }
}

void Stereo::setupConfidenceQueue(std::shared_ptr<dai::Device> device) {
    using param_handlers::ParamNames;
    utils::ImgConverterConfig convConfig;
    convConfig.tfPrefix = getOpticalFrameName(ph->getParam<std::string>("i_socket_name"));
    convConfig.getBaseDeviceTimestamp = ph->getParam<bool>(ParamNames::GET_BASE_DEVICE_TIMESTAMP);
    convConfig.updateROSBaseTimeOnRosMsg = ph->getParam<bool>(ParamNames::UPDATE_ROS_BASE_TIME_ON_ROS_MSG);
    convConfig.encoding = dai::ImgFrame::Type::RAW8;
    convConfig.reverseSocketOrder = ph->getParam<bool>(ParamNames::REVERSE_STEREO_SOCKET_ORDER);
    convConfig.isStereo = false;

    utils::ImgPublisherConfig pubConf;
    pubConf.daiNodeName = getName();
    pubConf.topicName = "~/" + getName();
    pubConf.topicSuffix = "/confidence";
    pubConf.rectified = true;
    pubConf.undistorted = true;
    pubConf.width = ph->getParam<int>(ParamNames::WIDTH);
    pubConf.height = ph->getParam<int>(ParamNames::HEIGHT);
    pubConf.socket = ph->getSocketID();
    pubConf.calibrationFile = ph->getParam<std::string>(ParamNames::CALIBRATION_FILE);
    pubConf.leftSocket = leftSensInfo.socket;
    pubConf.rightSocket = rightSensInfo.socket;
    pubConf.lazyPub = ph->getParam<bool>(ParamNames::ENABLE_LAZY_PUBLISHER);
    pubConf.maxQSize = ph->getParam<int>(ParamNames::MAX_Q_SIZE);

    confidencePub->setup(device, convConfig, pubConf);
}

void Stereo::setupQueues(std::shared_ptr<dai::Device> device) {
    computeRectifyRecipe(device);
    // Raw K/D stay the sensor's own; R/P carry the rectification recipe.
    for(auto& pub : left->getPublishers()) {
        pub->setRectifyOverride(rectifyRLeft, rectifyPLeft);
    }
    for(auto& pub : right->getPublishers()) {
        pub->setRectifyOverride(rectifyRRight, rectifyPRight);
    }
    left->setupQueues(device);
    right->setupQueues(device);
    if(ph->getParam<bool>("i_publish_topic")) {
        setupStereoQueue(device);
    }
    if(ph->getParam<bool>("i_publish_confidence") && !ph->getParam<bool>("i_use_neural_depth")) {
        setupConfidenceQueue(device);
    }
    if(ph->getParam<bool>("i_enable_left_rgbd")) {
        rgbdNodeLeft->setupQueues(device);
    }
    if(ph->getParam<bool>("i_enable_right_rgbd")) {
        rgbdNodeRight->setupQueues(device);
    }
    if(ph->getParam<bool>("i_left_rect_publish_topic")) {
        setupLeftRectQueue(device);
    }
    if(ph->getParam<bool>("i_right_rect_publish_topic")) {
        setupRightRectQueue(device);
    }
    if(ph->getParam<bool>("i_rect_wide_publish_topic")) {
        setupWideRectQueues();
    }
    if(ph->getParam<bool>("i_left_rect_enable_feature_tracker")) {
        featureTrackerLeftR->setupQueues(device);
    }
    if(ph->getParam<bool>("i_right_rect_enable_feature_tracker")) {
        featureTrackerRightR->setupQueues(device);
    }
    if(ph->getParam<bool>("i_enable_left_spatial_nn")) {
        nnNodeLeft->setupQueues(device);
    }
    if(ph->getParam<bool>("i_enable_right_spatial_nn")) {
        nnNodeRight->setupQueues(device);
    }
}
void Stereo::closeQueues() {
    left->closeQueues();
    right->closeQueues();
    if(ph->getParam<bool>("i_publish_topic")) {
        stereoPub->closeQueue();
    }
    if(ph->getParam<bool>("i_publish_confidence") && !ph->getParam<bool>("i_use_neural_depth")) {
        confidencePub->closeQueue();
    }
    if(ph->getParam<bool>("i_enable_left_rgbd")) {
        rgbdNodeLeft->closeQueues();
    }
    if(ph->getParam<bool>("i_enable_right_rgbd")) {
        rgbdNodeRight->closeQueues();
    }
    if(ph->getParam<bool>("i_left_rect_publish_topic")) {
        leftRectPub->closeQueue();
    }
    if(ph->getParam<bool>("i_right_rect_publish_topic")) {
        rightRectPub->closeQueue();
    }
    for(auto* wr : {&wideLeft, &wideRight}) {
        if(wr->q && !wr->q->isClosed()) {
            wr->q->removeCallback(wr->cbId);
            wr->q->close();
        }
    }
    if(ph->getParam<bool>("i_left_rect_enable_feature_tracker")) {
        featureTrackerLeftR->closeQueues();
    }
    if(ph->getParam<bool>("i_right_rect_enable_feature_tracker")) {
        featureTrackerRightR->closeQueues();
    }
    if(ph->getParam<bool>("i_enable_left_spatial_nn")) {
        nnNodeLeft->closeQueues();
    }
    if(ph->getParam<bool>("i_enable_right_spatial_nn")) {
        nnNodeRight->closeQueues();
    }
}

void Stereo::link(dai::Node::Input& in, int linkType) {
    if(linkType == static_cast<int>(link_types::StereoLinkType::stereo)) {
        if(aligned && alignNode) {
            alignNode->outputAligned.link(in);
        } else if(confMaskNode) {
            confMaskNode->out.link(in);
        } else if(ph->getParam<bool>("i_use_neural_depth")) {
            neuralDepthNode->depth.link(in);
        } else {
            stereoCamNode->depth.link(in);
        }
    } else if(linkType == static_cast<int>(link_types::StereoLinkType::left)) {
        if(ph->getParam<bool>("i_use_neural_depth")) {
            neuralDepthNode->rectifiedLeft.link(in);
        } else {
            stereoCamNode->rectifiedLeft.link(in);
        }
    } else if(linkType == static_cast<int>(link_types::StereoLinkType::right)) {
        if(ph->getParam<bool>("i_use_neural_depth")) {
            neuralDepthNode->rectifiedRight.link(in);
        } else {
            stereoCamNode->rectifiedRight.link(in);
        }
    } else {
        throw std::runtime_error("Wrong link type specified!");
    }
}

std::vector<std::shared_ptr<sensor_helpers::ImagePublisher>> Stereo::getPublishers() {
    std::vector<std::shared_ptr<sensor_helpers::ImagePublisher>> pubs;
    if(ph->getParam<bool>("i_publish_topic") && ph->getParam<bool>("i_synced")) {
        pubs.push_back(stereoPub);
    }
    if(ph->getParam<bool>("i_left_rect_publish_topic") && ph->getParam<bool>("i_left_rect_synced")) {
        pubs.push_back(leftRectPub);
    }
    if(ph->getParam<bool>("i_right_rect_publish_topic") && ph->getParam<bool>("i_right_rect_synced")) {
        pubs.push_back(rightRectPub);
    }
    auto pubsLeft = left->getPublishers();
    if(!pubsLeft.empty()) {
        pubs.insert(pubs.end(), pubsLeft.begin(), pubsLeft.end());
    }
    auto pubsRight = right->getPublishers();
    if(!pubsRight.empty()) {
        pubs.insert(pubs.end(), pubsRight.begin(), pubsRight.end());
    }
    return pubs;
}
dai::CameraBoardSocket Stereo::getSocketID() {
    return ph->getSocketID();
}

dai::Node::Input& Stereo::getInput(int linkType) {
    if(linkType == static_cast<int>(link_types::StereoLinkType::left)) {
        if(ph->getParam<bool>("i_use_neural_depth")) {
            return neuralDepthNode->left;
        } else {
            return stereoCamNode->left;
        }
    } else if(linkType == static_cast<int>(link_types::StereoLinkType::right)) {
        if(ph->getParam<bool>("i_use_neural_depth")) {
            return neuralDepthNode->right;
        } else {
            return stereoCamNode->right;
        }
    } else if(linkType == static_cast<int>(link_types::StereoLinkType::align)) {
        // If we built an explicit ImageAlign node (RVC4, or masking on RVC2), the
        // align-to image feeds it; otherwise StereoDepth aligns depth internally.
        if(alignNode) {
            return alignNode->inputAlignTo;
        }
        return stereoCamNode->inputAlignTo;
    } else {
        throw std::runtime_error("Wrong link type specified!");
    }
}
std::shared_ptr<SensorWrapper> Stereo::getLeftSensor() {
    return left;
}
std::shared_ptr<SensorWrapper> Stereo::getRightSensor() {
    return right;
}

int Stereo::getWidth() {
    return ph->getParam<int>(param_handlers::ParamNames::WIDTH);
}
int Stereo::getHeight() {
    return ph->getParam<int>(param_handlers::ParamNames::HEIGHT);
}
void Stereo::updateParams(const std::vector<rclcpp::Parameter>& params) {
    if(ph->getParam<bool>("i_use_neural_depth")) {
        auto ctrl = ph->setRuntimeParams(params);
        neuralControl->send(ctrl);
    }
}

}  // namespace dai_nodes
}  // namespace depthai_ros_driver
