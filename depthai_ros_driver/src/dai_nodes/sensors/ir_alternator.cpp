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

    // Always process the most recent tick. Stale ticks would tag frames that
    // already shipped.
    script->inputs["tick"].setBlocking(false);
    script->inputs["tick"].setMaxSize(1);

    // phaseOffset: which branch the FIRST captured frame's IR state should be.
    // 0 -> "this", 1 -> "other". Used only for cross-device IR phase alignment;
    // host routing is purely tag-driven and does not need parity at all.
    bool firstIsThis = (phaseOffset % 2) == 0;

    std::ostringstream body;
    body.precision(6);
    body << std::fixed;
    body << "thisDot = " << thisBranch.laserDot << "\n"
         << "thisFlood = " << thisBranch.flood << "\n"
         << "otherDot = " << otherBranch.laserDot << "\n"
         << "otherFlood = " << otherBranch.flood << "\n"
         << "firstIsThis = " << (firstIsThis ? "True" : "False") << "\n"
         << "node.warn(f'IrAlternator started; IR drivers: {str(Device.getIrDrivers())}')\n"
         // Prime the IR for the first captured frame.
         << "if firstIsThis:\n"
         << "    Device.setIrLaserDotProjectorIntensity(thisDot)\n"
         << "    Device.setIrFloodLightIntensity(thisFlood)\n"
         << "else:\n"
         << "    Device.setIrLaserDotProjectorIntensity(otherDot)\n"
         << "    Device.setIrFloodLightIntensity(otherFlood)\n"
         << "currentIsThis = firstIsThis\n"
         << "while True:\n"
         << "    tick = node.io['tick'].get()\n"
         // Emit tag for this frame: 1 byte IR state + 8 bytes device-clock
         // timestamp (little-endian MICROSECONDS). Microseconds because
         // Python timedelta only has us precision, so we'd lose nanos anyway;
         // host matches by the same us key.
         << "    ts_dev = tick.getTimestampDevice()\n"
         << "    ts_us = int(ts_dev.total_seconds() * 1e6)\n"
         << "    state_byte = 1 if currentIsThis else 0\n"
         << "    data = bytearray(9)\n"
         << "    data[0] = state_byte\n"
         << "    for i in range(8):\n"
         << "        data[1 + i] = (ts_us >> (i*8)) & 0xFF\n"
         << "    tag = Buffer(9)\n"
         << "    tag.setData(bytes(data))\n"
         << "    node.io['tag'].send(tag)\n"
         << "    if (tick.getSequenceNum() % 60) == 0:\n"
         << "        node.warn(f'IrAlt seq={tick.getSequenceNum()} ts_us={ts_us} currentIsThis={currentIsThis}')\n"
         // Toggle for the NEXT pulse.
         << "    currentIsThis = not currentIsThis\n"
         << "    if currentIsThis:\n"
         << "        Device.setIrLaserDotProjectorIntensity(thisDot)\n"
         << "        Device.setIrFloodLightIntensity(thisFlood)\n"
         << "    else:\n"
         << "        Device.setIrLaserDotProjectorIntensity(otherDot)\n"
         << "        Device.setIrFloodLightIntensity(otherFlood)\n";

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

dai::Node::Output* IrAlternator::getTagOutput() {
    return &script->outputs["tag"];
}

}  // namespace dai_nodes
}  // namespace depthai_ros_driver
