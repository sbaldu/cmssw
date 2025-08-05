
#pragma once

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

namespace cms {

  template <typename V, typename Score>
  class AssociationMapLayout {
    GENERATE_SOA_LAYOUT(ContentBuffersLayout,
                        SOA_COLUMN(uint32_t, indexes),
                        SOA_COLUMN(V, values),
                        SOA_COLUMN(Score, scores))
    GENERATE_SOA_LAYOUT(OffsetBufferLayout, SOA_COLUMN(uint32_t, offsets))

    GENERATE_SOA_BLOCKS(Layout,
                        SOA_BLOCK(content, ContentBuffersLayout),
                        SOA_BLOCK(offsets, OffsetBufferLayout),
                        SOA_VIEW_METHODS(
                            auto indexes(uint32_t id) const {
                              auto offset = offsets[id];
                              auto size = offsets[id + 1] - offset;
                              return std::span<uint32_t>{indexes + offset, size};
                            } auto values(uint32_t id) const {
                              auto offset = offsets[id];
                              auto size = offsets[id + 1] - offset;
                              return std::span<uint32_t>{values + offset, size};
                            } auto scores(uint32_t id) const {
                              auto offset = offsets[id];
                              auto size = offsets[id + 1] - offset;
                              return std::span<uint32_t>{scores + offset, size};
                            }))
  };

  template <typename V>
  class AssociationMapLayout<V, void> {
    GENERATE_SOA_LAYOUT(ContentBuffersLayout, SOA_COLUMN(uint32_t, indexes), SOA_COLUMN(V, values))
    GENERATE_SOA_LAYOUT(OffsetBufferLayout, SOA_COLUMN(uint32_t, offsets))

    GENERATE_SOA_BLOCKS(Layout,
                        SOA_BLOCK(content, ContentBuffersLayout),
                        SOA_BLOCK(offsets, OffsetBufferLayout),
                        SOA_VIEW_METHODS(
                            auto indexes(uint32_t id) const {
                              auto offset = offsets[id];
                              auto size = offsets[id + 1] - offset;
                              return std::span<uint32_t>{indexes + offset, size};
                            } auto values(uint32_t id) const {
                              auto offset = offsets[id];
                              auto size = offsets[id + 1] - offset;
                              return std::span<uint32_t>{values + offset, size};
                            }
                            ))
  };

  template <>
  class AssociationMapLayout<void, void> {
    GENERATE_SOA_LAYOUT(ContentBuffersLayout, SOA_COLUMN(uint32_t, indexes), )
    GENERATE_SOA_LAYOUT(OffsetBufferLayout, SOA_COLUMN(uint32_t, offsets))

    GENERATE_SOA_BLOCKS(Layout,
                        SOA_BLOCK(content, ContentBuffersLayout),
                        SOA_BLOCK(offsets, OffsetBufferLayout),
                        SOA_VIEW_METHODS(
                            auto indexes(uint32_t id) const {
                              auto offset = offsets[id];
                              auto size = offsets[id + 1] - offset;
                              return std::span<uint32_t>{indexes + offset, size};
                            }
                            ))
  };

  using associationMap = AssociationMapLayout<void, void>::Layout<>;
  using mapWithFraction = AssociationMapLayout<FractionEnergy, void>::Layout<>;
  using mapWithShared = AssociationMapLayout<SharedEnergy, void>::Layout<>;
  template <typename Score>
  using mapWithFractionAndScores = AssociationMapLayout<FractionEnergy, Score>::Layout<>;
  template <typename Score>
  using mapWithSharedAndScores = AssociationMapLayout<SharedEnergy, Score>::Layout<>;

}  // namespace cms
