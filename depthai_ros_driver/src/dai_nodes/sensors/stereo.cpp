#include "depthai_ros_driver_v3/dai_nodes/sensors/stereo.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

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

    // Before the pipeline is built: the mesh is part of the device's
    // configuration, and every published info is derived from the same recipe.
    computeRectifyRecipe(device);
    if(meshRectification()) {
        uploadRectifyMesh();
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

// The rectified pair every stream in this node is expressed in.
//
// R1/R2 come from cv::stereoRectify on the EEPROM extrinsics — the stored
// rectification rotations are NOT what rectification uses, so an info derived
// from them misses the actual geometry by several pixels.
//
// The projection keeps the WHOLE lens: the largest sensor-sized rectangle that
// lies inside both eyes' undistorted content, with an anisotropic focal so it
// fills the frame. The device's own recipe projects into the left camera's K
// instead, which crops a 129 deg OAK-W to ~97 deg and throws away the
// periphery. `uploadRectifyMesh` hands this projection to the device as a warp
// mesh, so <side>_rect carries it directly and no host remap is involved.
//
// Sizes come from the outputs actually linked into the depth node, not from the
// sensor's own i_width/i_height: those are different parameters and a mismatch
// would put the maps on a different grid than the frames.
void Stereo::computeRectifyRecipe(std::shared_ptr<dai::Device> device) {
    auto calHandler = device->readCalibration();
    auto toCv = [](const std::vector<std::vector<float>>& m, int rows, int cols) {
        cv::Mat out(rows, cols, CV_64F);
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++) out.at<double>(i, j) = m[i][j];
        return out;
    };
    const auto size = rectSize();
    const int w = size.first, h = size.second;
    cv::Mat K1 = toCv(calHandler.getCameraIntrinsics(leftSensInfo.socket, w, h), 3, 3);
    cv::Mat K2 = toCv(calHandler.getCameraIntrinsics(rightSensInfo.socket, w, h), 3, 3);
    auto d1 = calHandler.getDistortionCoefficients(leftSensInfo.socket);
    auto d2 = calHandler.getDistortionCoefficients(rightSensInfo.socket);
    // Only R1/R2 are wanted from stereoRectify, and it derives those from the
    // extrinsics alone — the distortion never enters them (verified: R1/R2 agree
    // to 0.0e+00 whether this is given 8 coefficients or 14). So truncating here
    // is free; the full model below is what the maps need.
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
    rectifyDLeft = lensModel(d1);
    rectifyDRight = lensModel(d2);
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            rectifyRLeft[i * 3 + j] = R1.at<double>(i, j);
            rectifyRRight[i * 3 + j] = R2.at<double>(i, j);
        }
    }
    // Signed baseline (Tx flips with reversed socket order), as a length: each
    // projection below scales it by its own focal.
    const double baseline = P2.at<double>(0, 3) / P2.at<double>(0, 0);

    // The device applies whatever mesh it is given; without one it uses its own
    // left-K recipe, and the info must say so.
    double fx, fy, cx, cy;
    if(meshRectification()) {
        std::tie(fx, fy, cx, cy) = wideProjection(R1, R2, w, h);
    } else {
        fx = K1.at<double>(0, 0);
        fy = K1.at<double>(1, 1);
        cx = K1.at<double>(0, 2);
        cy = K1.at<double>(1, 2);
    }
    rectifyPLeft = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};
    rectifyPRight = rectifyPLeft;
    rectifyPRight[3] = fx * baseline;

    const double hfov = (std::atan((w - cx) / fx) + std::atan(cx / fx)) * 180.0 / M_PI;
    const double hfovK = (std::atan((w - K1.at<double>(0, 2)) / K1.at<double>(0, 0)) + std::atan(K1.at<double>(0, 2) / K1.at<double>(0, 0))) * 180.0 / M_PI;
    RCLCPP_INFO(getLogger(),
                "rect: %dx%d, HFOV %.0f deg (%s; the device's own left-K recipe would be %.0f deg), fx=%.1f fy=%.1f cx=%.1f cy=%.1f, %d distortion coefficients, mesh step %d",
                w,
                h,
                hfov,
                meshRectification() ? "uploaded mesh, full lens" : "device recipe",
                hfovK,
                fx,
                fy,
                cx,
                cy,
                rectifyDLeft.cols,
                meshRectification() ? kMeshStep : 0);
}

// OpenCV takes 4, 5, 8, 12 or 14 distortion coefficients. The EEPROM this
// calibration writes carries 14 (rational + tilt), and dropping the tail is not
// the same lens: on a 129 deg OAK-W the tilt terms are worth 1-2 px at the
// periphery this projection exists to keep. Pad whatever the sensor reports up
// to the next accepted length rather than truncating -- a fisheye module
// reporting 4 is as valid as a rational one reporting 14.
cv::Mat Stereo::lensModel(const std::vector<float>& raw) const {
    const int given = static_cast<int>(raw.size());
    int n = 14;
    for(int s : {4, 5, 8, 12, 14}) {
        if(s >= given) {
            n = s;
            break;
        }
    }
    cv::Mat D = cv::Mat::zeros(1, n, CV_64F);
    for(int i = 0; i < std::min(given, n); i++) D.at<double>(i) = raw[i];
    return D;
}

// The largest rectangle, in the rectified plane, whose whole BORDER maps back
// inside both raw images -- returned as (fx, fy, cx, cy) for a sensor-sized
// frame.
//
// Step 1 bisects along the rectified principal row and column for a starting
// box. That alone is not a guarantee: tangential, thin-prism and tilt terms move
// the content boundary off those two axes, so step 2 shrinks the box about its
// centre until every point sampled along all four edges -- both eyes -- lands
// inside the raw image. Without step 2 the corners can fall outside the content
// and the device fills them with black.
std::tuple<double, double, double, double> Stereo::wideProjection(const cv::Mat& R1, const cv::Mat& R2, int w, int h) const {
    auto pixel = [&](const cv::Mat& K, const cv::Mat& D, const cv::Mat& R, double x, double y) {
        cv::Mat ray = R.t() * (cv::Mat_<double>(3, 1) << x, y, 1.0);
        std::vector<cv::Point3d> obj{cv::Point3d(ray.at<double>(0), ray.at<double>(1), ray.at<double>(2))};
        std::vector<cv::Point2d> img;
        cv::projectPoints(obj, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K, D, img);
        return img[0];
    };
    auto extents = [&](const cv::Mat& K, const cv::Mat& D, const cv::Mat& R) {
        auto edge = [&](int axis, double target) {
            const double sign = target == 0.0 ? -1.0 : 1.0;
            double lo = 0.0, hi = 8.0;
            for(int i = 0; i < 200; i++) {
                const double m = (lo + hi) / 2;
                const auto p = axis == 0 ? pixel(K, D, R, sign * m, 0.0) : pixel(K, D, R, 0.0, sign * m);
                const double px = axis == 0 ? p.x : p.y;
                (target == 0.0 ? px > 0.0 : px < target) ? lo = m : hi = m;
            }
            return sign * lo;
        };
        return std::array<double, 4>{edge(0, 0.0), edge(0, w - 1.0), edge(1, 0.0), edge(1, h - 1.0)};
    };
    const auto eL = extents(rectifyKLeft, rectifyDLeft, R1);
    const auto eR = extents(rectifyKRight, rectifyDRight, R2);
    double xmin = std::max(eL[0], eR[0]), xmax = std::min(eL[1], eR[1]);
    double ymin = std::max(eL[2], eR[2]), ymax = std::min(eL[3], eR[3]);

    const double xc = (xmin + xmax) / 2, yc = (ymin + ymax) / 2;
    const double xr = (xmax - xmin) / 2, yr = (ymax - ymin) / 2;
    constexpr int kBorderSamples = 64;
    auto borderInside = [&](double s) {
        for(const auto& [K, D, R] : {std::tuple(rectifyKLeft, rectifyDLeft, R1), std::tuple(rectifyKRight, rectifyDRight, R2)}) {
            for(int i = 0; i <= kBorderSamples; i++) {
                const double t = -1.0 + 2.0 * i / kBorderSamples;
                for(const auto& [x, y] : {std::pair(t, -1.0), std::pair(t, 1.0), std::pair(-1.0, t), std::pair(1.0, t)}) {
                    const auto p = pixel(K, D, R, xc + s * xr * x, yc + s * yr * y);
                    if(!(p.x >= 0.0 && p.x <= w - 1.0 && p.y >= 0.0 && p.y <= h - 1.0)) return false;
                }
            }
        }
        return true;
    };
    double lo = 0.0, hi = 1.0;
    if(borderInside(hi)) {
        lo = hi;
    } else {
        for(int i = 0; i < 40; i++) {
            const double m = (lo + hi) / 2;
            (borderInside(m) ? lo : hi) = m;
        }
    }
    // A hair inside the last passing scale: the bisection converges from below,
    // and the map is sampled at every pixel, not only at the border points.
    const double s = lo * (1.0 - 1e-3);
    xmin = xc - s * xr;
    xmax = xc + s * xr;
    ymin = yc - s * yr;
    ymax = yc + s * yr;
    const double fx = w / (xmax - xmin);
    const double fy = h / (ymax - ymin);
    return {fx, fy, -xmin * fx, -ymin * fy};
}

// Hand the recipe to the device as a warp mesh: a sparse grid of the raw (y, x)
// each output pixel samples, which the firmware interpolates between. A loaded
// mesh overrides the device's own rectification entirely (StereoDepth docs), so
// the rectified pair -- and the depth computed from it -- comes out in OUR
// projection, at no host cost. The device's own mesh is built from the left
// camera's K and only the first 8 distortion coefficients; this one keeps the
// whole lens and the full stored model.
void Stereo::uploadRectifyMesh() {
    const auto size = rectSize();
    const int w = size.first, h = size.second;
    auto mesh = [&](const cv::Mat& K, const cv::Mat& D, const std::array<double, 9>& Rarr, const std::array<double, 12>& Parr) {
        cv::Mat R(3, 3, CV_64F), P(3, 4, CV_64F);
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) R.at<double>(i, j) = Rarr[i * 3 + j];
            for(int j = 0; j < 4; j++) P.at<double>(i, j) = Parr[i * 4 + j];
        }
        cv::Mat mapX, mapY;
        cv::initUndistortRectifyMap(K, D, R, P, cv::Size(w, h), CV_32FC1, mapX, mapY);
        // Documented format: (h / step + 1) x (w / step + 1) points, each a
        // (y, x) float pair, row-major.
        std::vector<std::uint8_t> out;
        out.reserve(static_cast<size_t>(h / kMeshStep + 1) * (w / kMeshStep + 1) * 2 * sizeof(float));
        auto push = [&out](float v) {
            const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
            out.insert(out.end(), p, p + sizeof(float));
        };
        for(int my = 0; my <= h / kMeshStep; my++) {
            const int y = std::min(my * kMeshStep, h - 1);
            for(int mx = 0; mx <= w / kMeshStep; mx++) {
                const int x = std::min(mx * kMeshStep, w - 1);
                push(mapY.at<float>(y, x));
                push(mapX.at<float>(y, x));
            }
        }
        return out;
    };
    stereoCamNode->setMeshStep(kMeshStep, kMeshStep);
    stereoCamNode->loadMeshData(mesh(rectifyKLeft, rectifyDLeft, rectifyRLeft, rectifyPLeft), mesh(rectifyKRight, rectifyDRight, rectifyRRight, rectifyPRight));
}

// Only the classic StereoDepth node takes a warp mesh. On the neural-depth path
// the device rectifies by its own recipe and the recipe above stays the left-K
// one, so the published info keeps matching the pixels.
bool Stereo::meshRectification() const {
    return stereoCamNode != nullptr;
}

// The size of the outputs linked into the depth node -- what the rectified
// frames, and therefore the maps and the mesh, are on.
std::pair<int, int> Stereo::rectSize() {
    using ParamNames = param_handlers::ParamNames;
    if(meshRectification()) {
        return {ph->getParam<int>(ParamNames::WIDTH), ph->getParam<int>(ParamNames::HEIGHT)};
    }
    const auto name = getSocketName(leftSensInfo.socket);
    return {ph->getOtherNodeParam<int>(name, "i_width"), ph->getOtherNodeParam<int>(name, "i_height")};
}

void Stereo::setupRectQueue(std::shared_ptr<dai::Device> device,
                            dai::CameraFeatures& sensorInfo,
                            std::shared_ptr<sensor_helpers::ImagePublisher> pub,
                            bool isLeft) {
    auto sensorName = getSocketName(sensorInfo.socket);
    // The rectified stream lives in the rectified camera's frame (see
    // publishRectFrames), rotated from the physical sensor by R1/R2.
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

    // Publish the recipe computeRectifyRecipe derived — the projection the
    // uploaded mesh puts the pixels in — rather than an info built from the
    // EEPROM-stored rotations (validated by raw->rect feature reprojection at
    // <1 px).
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

    }

    pub->setup(device, convConfig, pubConfig);
}

// The rectified frame per eye, rotated from the sensor by R1/R2 and anchored
// once here so either eye's stream is placed whether or not the other is on.
// camera_info.r maps sensor-frame points into the
// rectified frame (x_rect = R * x_cam), so the child's orientation is R^T.
void Stereo::publishRectFrames() {
    if(!rectTfBroadcaster) {
        rectTfBroadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(getROSNode());
    }
    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    for(const auto& [sensorInfo, Rrect] : {std::pair(&leftSensInfo, &rectifyRLeft), std::pair(&rightSensInfo, &rectifyRRight)}) {
        const auto& R = *Rrect;
        const auto sensorName = getSocketName(sensorInfo->socket);
        tf2::Matrix3x3 rectRot(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);
        tf2::Quaternion q;
        rectRot.transpose().getRotation(q);
        geometry_msgs::msg::TransformStamped tfMsg;
        tfMsg.header.stamp = getROSNode()->get_clock()->now();
        tfMsg.header.frame_id = getOpticalFrameName(sensorName);
        tfMsg.child_frame_id = getOpticalFrameName(sensorName + "_rect");
        tfMsg.transform.rotation.x = q.x();
        tfMsg.transform.rotation.y = q.y();
        tfMsg.transform.rotation.z = q.z();
        tfMsg.transform.rotation.w = q.w();
        tfs.push_back(tfMsg);
    }
    rectTfBroadcaster->sendTransform(tfs);
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
    // Unaligned depth lives in the rectified left frame, so its info is that
    // camera -- which, with a mesh uploaded, is our projection and not the one
    // the device calibration would give. Aligned depth is warped to another
    // camera and keeps that camera's own info.
    if(meshRectification() && !aligned) {
        sensor_msgs::msg::CameraInfo info;
        info.width = pubConf.width;
        info.height = pubConf.height;
        info.distortion_model = "plumb_bob";
        info.d.assign(8, 0.0);
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                info.k[i * 3 + j] = rectifyPLeft[i * 4 + j];
                info.r[i * 3 + j] = rectifyRLeft[i * 3 + j];
            }
        }
        info.p = rectifyPLeft;
        pubConf.overrideInfo = info;
        pubConf.hasOverrideInfo = true;
    }

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
    const bool left_ = ph->getParam<bool>("i_left_rect_publish_topic");
    const bool right_ = ph->getParam<bool>("i_right_rect_publish_topic");
    if(left_ || right_) {
        publishRectFrames();
    }
    if(left_) {
        setupLeftRectQueue(device);
    }
    if(right_) {
        setupRightRectQueue(device);
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
