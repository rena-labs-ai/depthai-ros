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
 * Device-side Script (LEON_CSS) that toggles IR each FSYNC pulse.
 *
 * Frame data does NOT flow through the Script (Script outputs are typed Buffer
 * and downstream nodes like StereoDepth choke on them). The Script only takes
 * a tick from one camera output to wake on each frame; demuxing happens on the
 * host using the frame's own seqNum.
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

   private:
    std::shared_ptr<dai::node::Script> script;
};

}  // namespace dai_nodes
}  // namespace depthai_ros_driver
