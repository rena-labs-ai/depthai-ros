#pragma once

#include <cstdint>
#include <memory>

#include "depthai/pipeline/datatype/Buffer.hpp"
#include "depthai/pipeline/datatype/ImgFrame.hpp"
#include "depthai/pipeline/datatype/MessageGroup.hpp"
#include "depthai/pipeline/node/host/HostNode.hpp"

namespace depthai_ros_driver {
namespace dai_nodes {

/**
 * @brief Final-layer confidence mask for stereo depth.
 *
 * Zeroes every depth pixel whose StereoDepth confidence-map value is below
 * `threshold`. Mirrors the standalone diagnostic front_depthai.py --mask, which
 * showed this is the ONLY way to make "conf < threshold => depth == 0" hold
 * exactly (the on-device confidence threshold gates a different, earlier signal).
 *
 * Confidence-map scale: higher = MORE confident (empirically verified on this
 * stack). So we drop the LOW values: depth = 0 where conf < threshold.
 *
 * FRAME REQUIREMENT: the confidence map is always produced in the rectified
 * (unaligned) frame. depth must be in that SAME frame for the mask to line up.
 * Feed the UNALIGNED depth (run the stereo node with i_aligned:false). Masking
 * RGB-aligned depth with the rectified confidence map misregisters by the
 * stereo->RGB warp; as a guard, this node passes depth through untouched if the
 * two frames' dimensions/byte-sizes don't match.
 */
class ConfidenceMask : public dai::NodeCRTP<dai::node::HostNode, ConfidenceMask> {
   public:
    constexpr static const char* NAME = "ConfidenceMask";

    // HostNode auto-syncs its `inputs` map (by sequence-num/timestamp).
    dai::Node::Input& inDepth = inputs["depth"];
    dai::Node::Input& inConf = inputs["conf"];

    ConfidenceMask& setThreshold(int t) {
        threshold = t;
        return *this;
    }

    std::shared_ptr<dai::Buffer> processGroup(std::shared_ptr<dai::MessageGroup> grp) override {
        auto depth = grp->get<dai::ImgFrame>("depth");
        auto conf = grp->get<dai::ImgFrame>("conf");
        if(!depth || !conf) {
            return depth;  // pass through if a frame is missing this cycle
        }
        const size_t dw = depth->getWidth(), dh = depth->getHeight();
        const size_t cw = conf->getWidth(), ch = conf->getHeight();
        auto dData = depth->getData();  // RAW16 depth (mm), little-endian
        auto cData = conf->getData();   // RAW8 confidence (often HALF the depth res)
        // The StereoDepth confidence map is frequently at half the depth
        // resolution, so nearest-sample it per depth pixel rather than requiring
        // equal dims. Guard only against malformed buffers.
        if(dw > 0 && dh > 0 && cw > 0 && ch > 0 && dData.size() == dw * dh * 2 && cData.size() == cw * ch) {
            auto* d16 = reinterpret_cast<uint16_t*>(dData.data());
            for(size_t y = 0; y < dh; ++y) {
                const size_t cy = y * ch / dh;
                for(size_t x = 0; x < dw; ++x) {
                    if(cData[cy * cw + (x * cw / dw)] < threshold) {
                        d16[y * dw + x] = 0;
                    }
                }
            }
        }
        return depth;
    }

   private:
    int threshold = 100;
};

}  // namespace dai_nodes
}  // namespace depthai_ros_driver
