#pragma once

#include <memory>
#include <string>

#include "depthai/pipeline/Node.hpp"
#include "depthai_ros_driver_v3/dai_nodes/base_node.hpp"

namespace dai {
class Pipeline;
class Device;
namespace node {
class Script;
}
}  // namespace dai

namespace rclcpp {
class Node;
}

namespace depthai_ros_driver {
namespace dai_nodes {

/**
 * Device-side Script (LEON_CSS) that toggles IR each FSYNC pulse and emits a
 * sidecar tag per frame containing the IR state in effect during that frame's
 * exposure. The host pairs tags with data frames by device timestamp (set by
 * the sensor ISP at FSYNC edge — identical across sensors of the same OAK
 * for the same pulse).
 */
class IrAlternator : public BaseNode {
   public:
    struct BranchIntensities {
        float laserDot = 0.0f;
        float flood = 0.0f;
    };

    IrAlternator(const std::string& daiNodeName,
                 std::shared_ptr<rclcpp::Node> node,
                 std::shared_ptr<dai::Pipeline> pipeline,
                 const std::string& deviceName,
                 bool rsCompat,
                 const BranchIntensities& thisBranch,
                 const BranchIntensities& otherBranch,
                 int phaseOffset);
    ~IrAlternator();

    void setNames() override;
    void setInOut(std::shared_ptr<dai::Pipeline> pipeline) override;
    void setupQueues(std::shared_ptr<dai::Device> device) override;
    void closeQueues() override;

    dai::Node::Input& getTickInput();
    dai::Node::Output* getTagOutput();

   private:
    std::shared_ptr<dai::node::Script> script;
};

}  // namespace dai_nodes
}  // namespace depthai_ros_driver
