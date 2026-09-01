#include "depthai_ros_driver_v3/param_handlers/driver_param_handler.hpp"

#include "depthai/common/UsbSpeed.hpp"
#include "depthai_ros_driver_v3/utils.hpp"
#include "rclcpp/logger.hpp"
#include "rclcpp/node.hpp"

namespace depthai_ros_driver {
namespace param_handlers {
DriverParamHandler::DriverParamHandler(std::shared_ptr<rclcpp::Node> node, const std::string& name, const std::string& deviceName, bool rsCompat)
    : BaseParamHandler(node, name, deviceName, rsCompat) {
    usbSpeedMap = {
        {"LOW", dai::UsbSpeed::LOW},
        {"FULL", dai::UsbSpeed::FULL},
        {"HIGH", dai::UsbSpeed::HIGH},
        {"SUPER", dai::UsbSpeed::SUPER},
        {"SUPER_PLUS", dai::UsbSpeed::SUPER_PLUS},
    };
}
DriverParamHandler::~DriverParamHandler() = default;

dai::UsbSpeed DriverParamHandler::getUSBSpeed() {
    return utils::getValFromMap(getParam<std::string>("i_usb_speed"), usbSpeedMap);
}
void DriverParamHandler::declareParams() {
    declareAndLogParam<bool>("i_enable_ir", true);
    declareAndLogParam<std::string>("i_usb_speed", "SUPER");
    declareAndLogParam<std::string>("i_device_id", "");
    declareAndLogParam<std::string>("i_ip", "");
    declareAndLogParam<std::string>("i_usb_port_id", "");
    declareAndLogParam<bool>("i_pipeline_dump", false);
    declareAndLogParam<bool>("i_calibration_dump", false);
    declareAndLogParam<std::string>("i_external_calibration_path", "");
    declareAndLogParam<float>("r_laser_dot_intensity", 0.6, getRangedFloatDescriptor(0.0, 1.0));
    declareAndLogParam<float>("r_floodlight_intensity", 0.6, getRangedFloatDescriptor(0.0, 1.0));
    declareAndLogParam<bool>("i_restart_on_diagnostics_error", false);
    declareAndLogParam<bool>("i_rs_compat", false);

    // Alternate-mode (rgbd_alternate pipeline) per-branch IR config. Inert otherwise.
    // Tolerance (µs) for matching a frame's device timestamp to a Script-emitted
    // tag. Each FSYNC pulse is ~33333µs (30Hz) apart, and we observe ~20µs
    // per-sensor stamping offset, so 200µs is a safe default.
    declareAndLogParam<int>("i_left_right_tolerance_us", 200, getRangedIntDescriptor(0, 30000));
    // How long (ms) a frame waits for a matching tag before being dropped.
    declareAndLogParam<int>("i_frame_tag_timeout_ms", 10, getRangedIntDescriptor(0, 1000));
    declareAndLogParam<std::string>("this.namespace", "primary");
    declareAndLogParam<float>("this.r_laser_dot_intensity", 1.0, getRangedFloatDescriptor(0.0, 1.0));
    declareAndLogParam<float>("this.r_floodlight_intensity", 0.0, getRangedFloatDescriptor(0.0, 1.0));
    // Number of consecutive frames captured in this branch before switching to
    // the other. Default 1 → strict alternation. Use e.g. this=5, other=25 to
    // get 5 dot_on frames per 30-frame cycle (1:5 duty).
    declareAndLogParam<int>("this.i_frames_per_cycle", 1, getRangedIntDescriptor(1, 1000));
    declareAndLogParam<std::string>("other.namespace", "secondary");
    declareAndLogParam<float>("other.r_laser_dot_intensity", 0.0, getRangedFloatDescriptor(0.0, 1.0));
    declareAndLogParam<float>("other.r_floodlight_intensity", 0.0, getRangedFloatDescriptor(0.0, 1.0));
    declareAndLogParam<int>("other.i_frames_per_cycle", 1, getRangedIntDescriptor(1, 1000));

    declareAndLogParam<bool>("i_publish_tf_from_calibration", true);
    declareAndLogParam<std::string>("i_tf_device_name", getROSNode()->get_name());
    declareAndLogParam<std::string>("i_tf_device_model", "");
    declareAndLogParam<std::string>("i_tf_base_frame", "oak");
    declareAndLogParam<std::string>("i_tf_parent_frame", "oak_parent_frame");
    // Which sensor the base frame coincides with, by the same name the frames
    // use ("left"/"right"/"rgb", or "infra1"/"infra2"/"color" under
    // i_rs_compat). Empty keeps the historical behaviour: the base frame lands
    // on whichever socket happens to end the EEPROM extrinsic chain, which on
    // a device rooted at CAM_A means the base frame IS the RGB sensor.
    //
    // Setting it re-expresses every sensor against that one, so a consumer
    // that calibrates against a specific lens can hand its pose straight to
    // i_tf_cam_pos_*/i_tf_cam_* instead of correcting for a body->lens
    // transform it has to discover at runtime.
    declareAndLogParam<std::string>("i_tf_reference_socket", "");
    declareAndLogParam<std::string>("i_tf_cam_pos_x", "0.0");
    declareAndLogParam<std::string>("i_tf_cam_pos_y", "0.0");
    declareAndLogParam<std::string>("i_tf_cam_pos_z", "0.0");
    declareAndLogParam<std::string>("i_tf_cam_roll", "0.0");
    declareAndLogParam<std::string>("i_tf_cam_pitch", "0.0");
    declareAndLogParam<std::string>("i_tf_cam_yaw", "0.0");
    declareAndLogParam<std::string>("i_tf_imu_from_descr", "false");
    declareAndLogParam<std::string>("i_tf_custom_urdf_location", "");
    declareAndLogParam<std::string>("i_tf_custom_xacro_args", "");
}
}  // namespace param_handlers
}  // namespace depthai_ros_driver
