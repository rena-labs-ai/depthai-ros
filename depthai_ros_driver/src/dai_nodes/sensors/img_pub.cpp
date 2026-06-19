#include "depthai_ros_driver_v3/dai_nodes/sensors/img_pub.hpp"

#include <rclcpp/logging.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>

#include "camera_info_manager/camera_info_manager.hpp"
#include "compressed_depth_image_transport/codec.h"
#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/VideoEncoder.hpp"
#include "depthai/properties/VideoEncoderProperties.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "depthai_ros_driver_v3/dai_nodes/sensors/sensor_helpers.hpp"
#include "depthai_ros_driver_v3/utils.hpp"
#include "ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp"
#include "image_transport/image_transport.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"

namespace depthai_ros_driver {
namespace dai_nodes {
namespace sensor_helpers {
namespace {
const char* usbSpeedStr(dai::UsbSpeed s) {
    switch(s) {
        case dai::UsbSpeed::LOW:
            return "LOW";
        case dai::UsbSpeed::FULL:
            return "FULL";
        case dai::UsbSpeed::HIGH:
            return "HIGH (USB2)";
        case dai::UsbSpeed::SUPER:
            return "SUPER (USB3)";
        case dai::UsbSpeed::SUPER_PLUS:
            return "SUPER_PLUS (USB3)";
        default:
            return "UNKNOWN";
    }
}
}  // namespace
ImagePublisher::ImagePublisher(std::shared_ptr<rclcpp::Node> node,
                               std::shared_ptr<dai::Pipeline> pipeline,
                               const std::string& qName,
                               dai::Node::Output* out,
                               bool synced,
                               bool ipcEnabled,
                               const utils::VideoEncoderConfig& encoderConfig)
    : node(node), encConfig(encoderConfig), out(out), qName(qName), ipcEnabled(ipcEnabled), synced(synced) {
    if(encoderConfig.enabled) {
        encoder = createEncoder(pipeline, encoderConfig);
        this->out->link(encoder->input);
    }
}
void ImagePublisher::setup(std::shared_ptr<dai::Device> device, const utils::ImgConverterConfig& convConf, const utils::ImgPublisherConfig& pubConf) {
    convConfig = convConf;
    pubConfig = pubConf;
    createImageConverter(device);
    createInfoManager(device);
    if(pubConfig.topicName.empty()) {
        throw std::runtime_error("Topic name cannot be empty!");
    }
    rclcpp::PublisherOptions pubOptions;
    pubOptions.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
    rclcpp::QoS imgQos(10);
    if(pubConfig.publishCompressed) {
        if(encConfig.profile == dai::VideoEncoderProperties::Profile::MJPEG) {
            compressedImgPub =
                node->create_publisher<sensor_msgs::msg::CompressedImage>(pubConfig.topicName + pubConfig.compressedTopicSuffix, imgQos, pubOptions);
        } else {
            ffmpegPub = node->create_publisher<ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(
                pubConfig.topicName + pubConfig.compressedTopicSuffix, imgQos, pubOptions);
        }
        infoPub =
            node->create_publisher<sensor_msgs::msg::CameraInfo>(pubConfig.topicName + pubConfig.infoSuffix + "/camera_info", rclcpp::QoS(10), pubOptions);
    } else if(pubConfig.logLatency) {
        // Own raw + compressed + info publishers so PNG encode and the DDS write can be
        // timed separately (image_transport fuses both inside one opaque publish call).
        imgPub = node->create_publisher<sensor_msgs::msg::Image>(pubConfig.topicName + pubConfig.topicSuffix, imgQos, pubOptions);
        if(pubConfig.enableCompressedDepth) {
            compressedImgPub = node->create_publisher<sensor_msgs::msg::CompressedImage>(
                pubConfig.topicName + pubConfig.topicSuffix + "/compressedDepth", imgQos, pubOptions);
        }
        infoPub =
            node->create_publisher<sensor_msgs::msg::CameraInfo>(pubConfig.topicName + pubConfig.infoSuffix + "/camera_info", rclcpp::QoS(10), pubOptions);
    } else if(!pubConfig.enableCompressedDepth) {
        // Compressed transport disabled: advertise raw image + info only, so the driver
        // never encodes or publishes a compressedDepth topic regardless of subscribers.
        imgPub = node->create_publisher<sensor_msgs::msg::Image>(pubConfig.topicName + pubConfig.topicSuffix, imgQos, pubOptions);
        infoPub =
            node->create_publisher<sensor_msgs::msg::CameraInfo>(pubConfig.topicName + pubConfig.infoSuffix + "/camera_info", rclcpp::QoS(10), pubOptions);
    } else {
        imgPubIT = image_transport::create_camera_publisher(node.get(), pubConfig.topicName + pubConfig.topicSuffix);
    }
    if(!synced) {
        if(encConfig.enabled) {
            dataQ = encoder->out.createOutputQueue(pubConf.maxQSize, pubConf.qBlocking);
        } else {
            dataQ = out->createOutputQueue(pubConf.maxQSize, pubConf.qBlocking);
        }
        addQueueCB();
    }
    if(pubConfig.logLatency) {
        RCLCPP_INFO(node->get_logger(),
                    "[lat %s] USB link: %s | output queue max size: %d",
                    qName.c_str(),
                    usbSpeedStr(device->getUsbSpeed()),
                    pubConfig.maxQSize);
    }
}

void ImagePublisher::createImageConverter(std::shared_ptr<dai::Device> device) {
    converter = std::make_shared<depthai_bridge::ImageConverter>(convConfig.tfPrefix, convConfig.interleaved, convConfig.getBaseDeviceTimestamp);
    converter->setUpdateRosBaseTimeOnToRosMsg(convConfig.updateROSBaseTimeOnRosMsg);
    if(convConfig.lowBandwidth) {
        converter->convertFromBitstream(convConfig.encoding);
        if(convConfig.isStereo && !convConfig.outputDisparity) {
            try {
                auto calHandler = device->readCalibration();
                double baseline = calHandler.getBaselineDistance(pubConfig.leftSocket, pubConfig.rightSocket, false);
                if(convConfig.reverseSocketOrder) {
                    baseline = calHandler.getBaselineDistance(pubConfig.rightSocket, pubConfig.leftSocket, false);
                }
                converter->convertDispToDepth(baseline);
            } catch(const std::exception& e) {
                RCLCPP_DEBUG(node->get_logger(), "Failed to convert disparity to depth: %s", e.what());
            }
        }
    }
    if(convConfig.addExposureOffset) {
        converter->addExposureOffset(convConfig.expOffset);
    }
    if(convConfig.reverseSocketOrder) {
        converter->reverseStereoSocketOrder();
    }
    if(convConfig.alphaScalingEnabled) {
        converter->setAlphaScaling(convConfig.alphaScaling);
    }
    if(convConfig.isStereo && !convConfig.outputDisparity) {
        auto calHandler = device->readCalibration();
        double baseline = calHandler.getBaselineDistance(pubConfig.leftSocket, pubConfig.rightSocket, false);
        if(convConfig.reverseSocketOrder) {
            baseline = calHandler.getBaselineDistance(pubConfig.rightSocket, pubConfig.leftSocket, false);
        }
        converter->convertDispToDepth(baseline);
    }
    converter->setFFMPEGEncoding(convConfig.ffmpegEncoder);
}

std::shared_ptr<dai::node::VideoEncoder> ImagePublisher::createEncoder(std::shared_ptr<dai::Pipeline> pipeline,
                                                                       const utils::VideoEncoderConfig& encoderConfig) {
    auto enc = pipeline->create<dai::node::VideoEncoder>();
    enc->setQuality(encoderConfig.quality);
    enc->setProfile(encoderConfig.profile);
    if(encoderConfig.profile != dai::VideoEncoderProperties::Profile::MJPEG) {
        enc->setBitrate(encoderConfig.bitrate);
        enc->setKeyframeFrequency(encoderConfig.frameFreq);
    }
    return enc;
}
void ImagePublisher::createInfoManager(std::shared_ptr<dai::Device> device) {
    infoManager = std::make_shared<camera_info_manager::CameraInfoManager>(
        node->create_sub_node(std::string(node->get_name()) + "/" + pubConfig.daiNodeName).get(), "/" + pubConfig.daiNodeName + pubConfig.infoMgrSuffix);
    if(pubConfig.calibrationFile.empty()) {
        auto calHandler = device->readCalibration();
        auto info = sensor_helpers::getCalibInfo(node->get_logger(), converter, calHandler, pubConfig.socket, pubConfig.width, pubConfig.height);
        if(pubConfig.rectified) {
            std::fill(info.d.begin(), info.d.end(), 0.0);
            info.r[0] = info.r[4] = info.r[8] = 1.0;
        }
        infoManager->setCameraInfo(info);
    } else {
        infoManager->loadCameraInfo(pubConfig.calibrationFile);
    }
};
ImagePublisher::~ImagePublisher() {
    closeQueue();
};

void ImagePublisher::closeQueue() {
    if(dataQ && !dataQ->isClosed()) {
        dataQ->removeCallback(cbID);
        dataQ->close();
    }
}
void ImagePublisher::link(dai::Node::Input& in) {
    out->link(in);
}
std::shared_ptr<dai::MessageQueue> ImagePublisher::getQueue() {
    return dataQ;
}
bool ImagePublisher::isSynced() {
    return synced;
}
void ImagePublisher::addQueueCB() {
    cbID = dataQ->addCallback([this](const std::shared_ptr<dai::ADatatype>& data) { publish(data); });
}

std::string ImagePublisher::getQueueName() {
    return qName;
}
std::shared_ptr<Image> ImagePublisher::convertData(const std::shared_ptr<dai::ADatatype>& data) {
    sensor_msgs::msg::CameraInfo info;
    auto img = std::make_shared<Image>();
    if(encConfig.enabled) {
        auto daiImg = std::dynamic_pointer_cast<dai::EncodedFrame>(data);
        if(pubConfig.calibrationFile.empty()) {
            info = converter->generateCameraInfo(daiImg);
        } else {
            info = infoManager->getCameraInfo();
        }
        if(pubConfig.publishCompressed) {
            auto rawMsg = converter->toRosMsgRawPtr(daiImg, info);
            info.header = rawMsg.header;
            if(encConfig.profile == dai::VideoEncoderProperties::Profile::MJPEG) {
                std::deque<sensor_msgs::msg::CompressedImage> deq;
                converter->toRosCompressedMsg(daiImg, deq);
                img->compressedImg = std::make_unique<sensor_msgs::msg::CompressedImage>(deq.front());
            } else {
                std::deque<ffmpeg_image_transport_msgs::msg::FFMPEGPacket> deq;
                converter->toRosFFMPEGPacket(daiImg, deq);
                img->ffmpegPacket = std::make_unique<ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(deq.front());
            }
        } else {
            auto rawMsg = converter->toRosMsgRawPtr(daiImg, info);
            info.header = rawMsg.header;
            sensor_msgs::msg::Image::UniquePtr msg = std::make_unique<sensor_msgs::msg::Image>(rawMsg);
            img->image = std::move(msg);
        }
    } else {
        auto daiImg = std::dynamic_pointer_cast<dai::ImgFrame>(data);
        if(pubConfig.calibrationFile.empty()) {
            info = converter->generateCameraInfo(daiImg);
        } else {
            info = infoManager->getCameraInfo();
        }
        auto rawMsg = converter->toRosMsgRawPtr(daiImg, info);
        info.header = rawMsg.header;
        sensor_msgs::msg::Image::UniquePtr msg = std::make_unique<sensor_msgs::msg::Image>(rawMsg);
        img->image = std::move(msg);
    }
    if(pubConfig.rectified) {
        info.r[0] = info.r[4] = info.r[8] = 1.0;
    }
    if(pubConfig.undistorted) {
        std::fill(info.d.begin(), info.d.end(), 0.0);
    }
    sensor_msgs::msg::CameraInfo::UniquePtr infoMsg = std::make_unique<sensor_msgs::msg::CameraInfo>(info);
    img->info = std::move(infoMsg);
    return img;
}
void ImagePublisher::publish(std::shared_ptr<Image> img) {
    if(pubConfig.publishCompressed) {
        if(encConfig.profile == dai::VideoEncoderProperties::Profile::MJPEG) {
            compressedImgPub->publish(std::move(img->compressedImg));
        } else {
            ffmpegPub->publish(std::move(img->ffmpegPacket));
        }
        infoPub->publish(std::move(img->info));
    } else if(!pubConfig.enableCompressedDepth) {
        // Compressed transport disabled: raw image + info only (no imgPubIT exists).
        imgPub->publish(std::move(img->image));
        infoPub->publish(std::move(img->info));
    } else {
        if(ipcEnabled && (!pubConfig.lazyPub || detectSubscription(imgPub, infoPub))) {
            imgPub->publish(std::move(img->image));
            infoPub->publish(std::move(img->info));
        } else {
            if(!pubConfig.lazyPub || imgPubIT.getNumSubscribers() > 0) imgPubIT.publish(*img->image, *img->info);
        }
    }
}
void ImagePublisher::publish(std::shared_ptr<Image> img, rclcpp::Time timestamp) {
    img->info->header.stamp = timestamp;
    if(pubConfig.publishCompressed) {
        if(encConfig.profile == dai::VideoEncoderProperties::Profile::MJPEG) {
            img->compressedImg->header.stamp = timestamp;
        } else {
            img->ffmpegPacket->header.stamp = timestamp;
        }
    } else {
        img->image->header.stamp = timestamp;
    }
    publish(img);
}

void ImagePublisher::publish(const std::shared_ptr<dai::ADatatype>& data) {
    if(rclcpp::ok()) {
        if(!pubConfig.logLatency) {
            auto img = convertData(data);
            publish(img);
            return;
        }
        // Timed path: capture->callback (on-device compute + USB), ROS-msg
        // conversion, and publish (incl. compressedDepth PNG encode on the host).
        auto entry = node->now();
        // Backlog still waiting in the host output queue when this callback fired:
        // ~full => host consumer too slow; ~0 but dev+usb high => device/USB upstream.
        double qSize = static_cast<double>(dataQ ? dataQ->getSize() : 0);
        auto t0 = std::chrono::steady_clock::now();
        auto img = convertData(data);
        auto t1 = std::chrono::steady_clock::now();
        rclcpp::Time capture(img->info->header.stamp);
        auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        // Encode the depth ourselves (same codec the image_transport plugin uses) then
        // do the compressed DDS write, so encode vs publish are timed independently.
        // Skipped entirely when compressed depth is disabled (no encode, no topic).
        double encodeMs = 0.0;
        double publishMs = 0.0;
        if(pubConfig.enableCompressedDepth) {
            auto compressed =
                compressed_depth_image_transport::encodeCompressedDepthImage(*img->image, 10.0, 100.0, pubConfig.pngLevel);
            auto t2 = std::chrono::steady_clock::now();
            encodeMs = ms(t1, t2);
            if(compressed) {
                compressed->header = img->image->header;
                compressedImgPub->publish(*compressed);
            }
            publishMs = ms(t2, std::chrono::steady_clock::now());
        }
        imgPub->publish(std::move(img->image));
        infoPub->publish(std::move(img->info));
        recordLatency((entry - capture).seconds() * 1e3, ms(t0, t1), encodeMs, publishMs, qSize);
    }
}

void ImagePublisher::recordLatency(double devUsbMs, double convertMs, double encodeMs, double publishMs, double queueSize) {
    latDevUsb.push_back(devUsbMs);
    latConvert.push_back(convertMs);
    latEncode.push_back(encodeMs);
    latPublish.push_back(publishMs);
    latTotal.push_back(devUsbMs + convertMs + encodeMs + publishMs);
    latQSize.push_back(queueSize);
    constexpr size_t kN = 30;  // ~1s at 30fps
    if(latTotal.size() < kN) {
        return;
    }
    auto stat = [](std::vector<double>& v) {
        double mn = *std::min_element(v.begin(), v.end());
        double mx = *std::max_element(v.begin(), v.end());
        double avg = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        char buf[48];
        std::snprintf(buf, sizeof(buf), "%.1f/%.1f/%.1f", mn, avg, mx);
        return std::string(buf);
    };
    RCLCPP_INFO(node->get_logger(),
                "[lat %s n=%zu ms min/avg/max] dev+usb %s | convert %s | encode %s | publish %s | TOTAL %s | qsize %s",
                qName.c_str(),
                latTotal.size(),
                stat(latDevUsb).c_str(),
                stat(latConvert).c_str(),
                stat(latEncode).c_str(),
                stat(latPublish).c_str(),
                stat(latTotal).c_str(),
                stat(latQSize).c_str());
    latDevUsb.clear();
    latConvert.clear();
    latEncode.clear();
    latPublish.clear();
    latTotal.clear();
    latQSize.clear();
}

bool ImagePublisher::detectSubscription(const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& pub,
                                        const rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr& infoPub) {
    return (pub->get_subscription_count() > 0 || pub->get_intra_process_subscription_count() > 0 || infoPub->get_subscription_count() > 0
            || infoPub->get_intra_process_subscription_count() > 0);
}
}  // namespace sensor_helpers
}  // namespace dai_nodes
}  // namespace depthai_ros_driver
