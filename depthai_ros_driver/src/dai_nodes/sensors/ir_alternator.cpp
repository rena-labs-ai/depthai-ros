#include "depthai_ros_driver_v3/dai_nodes/sensors/ir_alternator.hpp"

#include <sstream>

#include "depthai/common/ProcessorType.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/Script.hpp"
#include "rclcpp/node.hpp"

namespace depthai_ros_driver {
namespace dai_nodes {

IrAlternator::IrAlternator(const std::string& daiNodeName,
                           std::shared_ptr<rclcpp::Node> node,
                           std::shared_ptr<dai::Pipeline> pipeline,
                           const std::string& deviceName,
                           bool rsCompat,
                           const BranchIntensities& thisBranch,
                           const BranchIntensities& otherBranch,
                           int phaseOffset)
    : BaseNode(daiNodeName, node, pipeline, deviceName, rsCompat) {
    script = pipeline->create<dai::node::Script>();
    script->setProcessor(dai::ProcessorType::LEON_CSS);

    std::ostringstream body;
    body.precision(6);
    body << std::fixed;
    body << "thisDot = " << thisBranch.laserDot << "\n"
         << "thisFlood = " << thisBranch.flood << "\n"
         << "otherDot = " << otherBranch.laserDot << "\n"
         << "otherFlood = " << otherBranch.flood << "\n"
         << "phaseOffset = " << phaseOffset << "\n"
         << "node.warn(f'IrAlternator started; IR drivers: {str(Device.getIrDrivers())}')\n"
         << "if ((0 + phaseOffset) % 2) == 0:\n"
         << "    Device.setIrLaserDotProjectorIntensity(thisDot)\n"
         << "    Device.setIrFloodLightIntensity(thisFlood)\n"
         << "else:\n"
         << "    Device.setIrLaserDotProjectorIntensity(otherDot)\n"
         << "    Device.setIrFloodLightIntensity(otherFlood)\n"
         << "while True:\n"
         << "    tick = node.io['tick'].get()\n"
         << "    seq = tick.getSequenceNum()\n"
         << "    isThis = ((seq + phaseOffset) % 2) == 0\n"
         // Set IR for the NEXT frame (its parity will be opposite).
         << "    if isThis:\n"
         << "        Device.setIrLaserDotProjectorIntensity(otherDot)\n"
         << "        Device.setIrFloodLightIntensity(otherFlood)\n"
         << "    else:\n"
         << "        Device.setIrLaserDotProjectorIntensity(thisDot)\n"
         << "        Device.setIrFloodLightIntensity(thisFlood)\n";

    script->setScript(body.str(), getName());
}

IrAlternator::~IrAlternator() = default;

void IrAlternator::setNames() {}
void IrAlternator::setInOut(std::shared_ptr<dai::Pipeline> /*pipeline*/) {}
void IrAlternator::setupQueues(std::shared_ptr<dai::Device> /*device*/) {}
void IrAlternator::closeQueues() {}

dai::Node::Input& IrAlternator::getTickInput() {
    return script->inputs["tick"];
}

}  // namespace dai_nodes
}  // namespace depthai_ros_driver
