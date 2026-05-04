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
                           const BranchIntensities& otherBranch)
    : BaseNode(daiNodeName, node, pipeline, deviceName, rsCompat) {
    script = pipeline->create<dai::node::Script>();
    script->setProcessor(dai::ProcessorType::LEON_CSS);

    // Always process the most recent tick. Stale ticks would tag frames that
    // already shipped.
    script->inputs["tick"].setBlocking(false);
    script->inputs["tick"].setMaxSize(1);

    // First captured frame is "this" by convention. Host routing is purely
    // tag-driven, so the actual phase doesn't matter — every frame is labeled
    // by the IR state that was active during its exposure, independent of
    // which pulse the Script started on.
    std::ostringstream body;
    body.precision(6);
    body << std::fixed;
    body << "thisDot = " << thisBranch.laserDot << "\n"
         << "thisFlood = " << thisBranch.flood << "\n"
         << "otherDot = " << otherBranch.laserDot << "\n"
         << "otherFlood = " << otherBranch.flood << "\n"
         << "thisBudget = " << thisBranch.framesPerCycle << "\n"
         << "otherBudget = " << otherBranch.framesPerCycle << "\n"
         << "node.warn(f'IrAlternator started; thisBudget={thisBudget} otherBudget={otherBudget} drivers={str(Device.getIrDrivers())}')\n"
         << "Device.setIrLaserDotProjectorIntensity(thisDot)\n"
         << "Device.setIrFloodLightIntensity(thisFlood)\n"
         << "currentIsThis = True\n"
         << "remaining = thisBudget\n"
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
         // Maintain a per-branch frame budget. Stay in the current branch
         // until 'remaining' hits zero, then flip and reload from the other
         // branch's budget. budget=1,1 reproduces strict alternation.
         << "    remaining = remaining - 1\n"
         << "    if remaining <= 0:\n"
         << "        currentIsThis = not currentIsThis\n"
         << "        remaining = thisBudget if currentIsThis else otherBudget\n"
         << "        if currentIsThis:\n"
         << "            Device.setIrLaserDotProjectorIntensity(thisDot)\n"
         << "            Device.setIrFloodLightIntensity(thisFlood)\n"
         << "        else:\n"
         << "            Device.setIrLaserDotProjectorIntensity(otherDot)\n"
         << "            Device.setIrFloodLightIntensity(otherFlood)\n";

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
