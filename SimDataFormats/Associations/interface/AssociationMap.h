
#include "DataFormats/SoATemplate/interface/SoAAssociatorLayout.h"

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

namespace TICL {

  struct FractionType {
    float value;
    FractionType(float v = 0.0f) : value(v) {}
    FractionType& operator+=(float v) {
      value += v;
      return *this;
    }
  };

  struct SharedEnergyType {
    float value;
    SharedEnergyType(float v = 0.0f) : value(v) {}
    SharedEnergyType& operator+=(float v) {
      value += v;
      return *this;
    }
  };

  template <typename Values, typename Score, typename Collection1, typename Collection2>
  struct AssociationMapLayoutStruct {
    GENERATE_SOA_ASSOCIATOR_LAYOUT(Associator,
                                   SOA_COLUMN(uint32_t, indexes),
                                   SOA_COLUMN(Values, values),
                                   SOA_COLUMN(Score, scores))
  };

  template <typename Values, typename Collection1, typename Collection2>
  struct AssociationMapLayoutStruct<Values, void, Collection1, Collection2> {
    GENERATE_SOA_ASSOCIATOR_LAYOUT(Associator, SOA_COLUMN(uint32_t, indexes), SOA_COLUMN(Values, values))
  };

  template <typename Collection1, typename Collection2>
  struct AssociationMapLayoutStruct<void, void, Collection1, Collection2> {
    GENERATE_SOA_ASSOCIATOR_LAYOUT(Associator, SOA_COLUMN(uint32_t, indexes))
  };

  template <typename Values, typename Score, typename Collection1 = void, typename Collection2 = void>
  using AssociationMap = typename AssociationMapLayoutStruct<Values, Score, Collection1, Collection2>::Associator<>;
  template <typename Values, typename Score, typename Collection1 = void, typename Collection2 = void>
  using AssociationMapView = typename AssociationMap<Values, Score, Collection1, Collection2>::View;
  template <typename Values, typename Score, typename Collection1 = void, typename Collection2 = void>
  using AssociationMapConstView = typename AssociationMap<Values, Score, Collection1, Collection2>::ConstView;

  template <typename Collection1 = void, typename Collection2 = void>
  using mapWithFraction = AssociationMap<FractionType, void, Collection1, Collection2>;
  template <typename Collection1 = void, typename Collection2 = void>
  using mapWithFractionAndScore = AssociationMap<FractionType, float, Collection1, Collection2>;

  template <typename Collection1 = void, typename Collection2 = void>
  using mapWithSharedEnergy = AssociationMap<SharedEnergyType, void, Collection1, Collection2>;
  template <typename Collection1 = void, typename Collection2 = void>
  using mapWithSharedEnergyAndScore = AssociationMap<SharedEnergyType, float, Collection1, Collection2>;

  namespace detail {

    struct KernelComputeAssociationSizes {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc, const uint32_t* associations, uint32_t* sizes, size_t size) const {
        for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
          alpaka::atomicAdd(acc, &sizes[associations[i]], 1u);
        }
      }
    };

    template <typename Values, typename Score, typename Collection1, typename Collection2>
    ALPAKA_FN_ACC void insert(AssociationMapView<Values, Score, Collection1, Collection2> view,
                              size_t size,
                              int offset,
                              int index,
                              Values fraction_or_energy,
                              Score score) {
      assert(static_cast<size_t>(index) < size);
      if constexpr (!std::is_void_v<Values>) {
        view.values()[offset] = fraction_or_energy;
      }
      if constexpr (!std::is_void_v<Score>) {
        view.scores()[offset] = score;
      }
      view.indexes()[offset] = index;
    }

    template <typename Values, typename Score, typename Collection1, typename Collection2>
    struct KernelFillAssociator {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    AssociationMapView<Values, Score, Collection1, Collection2> view,
                                    const uint32_t* bin_buffer,
                                    const Values* values,
                                    const Score* scores,
                                    uint32_t* temp_offsets,
                                    size_t size) const {
        for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
          const auto binId = bin_buffer[i];
          const uint32_t position = alpaka::atomicAdd(acc, &temp_offsets[binId], 1u);
          insert<Values, Score, Collection1, Collection2>(view, size, position, i, values[i], scores[i]);
        }
      }
    };

    template <typename TAcc,
              typename Values,
              typename Score,
              typename Collection1,
              typename Collection2,
              typename TQueue,
              typename = std::enable_if_t<alpaka::isQueue<TQueue>>,
              typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST void fill(AssociationMapView<Values, Score, Collection1, Collection2> map,
                             const uint32_t* indexes,
                             const uint32_t* assoc_ids,
                             const Values* values,
                             const Score* scores,
                             TQueue& queue) {
      using namespace cms::alpakatools;

      const auto nbins = map.metadata().bins();
      const auto size = map.metadata().size();

      auto dev = alpaka::getDev(queue);

      const auto blocksize = 512;
      const auto gridsize = divide_up_by(size, blocksize);
      const auto workdiv = make_workdiv<TAcc>(gridsize, blocksize);

      auto sizes_buffer = make_device_buffer<uint32_t[]>(dev, nbins);
      alpaka::memset(queue, sizes_buffer, 0);
      alpaka::exec<TAcc>(queue, workdiv, KernelComputeAssociationSizes{}, assoc_ids, sizes_buffer.data(), size);

      // prepare for prefix scan
      auto block_counter = make_device_buffer<int32_t>(queue);
      alpaka::memset(queue, block_counter, 0);

      uint32_t* offsets = map.metadata().offsets();
      auto view = make_device_view(dev, offsets, nbins);
      alpaka::memset(queue, view, 0u);

      const auto blocksize_multiblockscan = 1024;
      auto gridsize_multiblockscan = divide_up_by(nbins, blocksize_multiblockscan);
      const auto workdiv_multiblockscan = make_workdiv<TAcc>(gridsize_multiblockscan, blocksize_multiblockscan);
      auto warp_size = alpaka::getPreferredWarpSize(dev);
      alpaka::exec<TAcc>(queue,
                         workdiv_multiblockscan,
                         multiBlockPrefixScan<uint32_t>{},
                         sizes_buffer.data(),
                         offsets + 1,
                         nbins,
                         gridsize_multiblockscan,
                         block_counter.data(),
                         warp_size);

      auto temp_offsets = make_device_buffer<uint32_t[]>(queue, nbins + 1);
      alpaka::memcpy(queue, temp_offsets, make_device_view(dev, offsets, nbins));
      alpaka::exec<TAcc>(queue,
                         workdiv,
                         KernelFillAssociator<Values, Score, Collection1, Collection2>{},
                         map,
                         assoc_ids,
                         values,
                         scores,
                         temp_offsets.data(),
                         size);
    }

  }  // namespace detail

  template <typename TAcc,
            typename Values,
            typename Score,
            typename Collection1,
            typename Collection2,
            typename TQueue,
            typename = std::enable_if_t<alpaka::isQueue<TQueue>>,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_HOST void fill(AssociationMapView<Values, Score, Collection1, Collection2> map,
                           const uint32_t* indexes,
                           const uint32_t* assoc_ids,
                           TQueue& queue) {
    const Values* values = nullptr;
    const Score* scores = nullptr;
    detail::fill<TAcc, Values, Score, Collection1, Collection2, TQueue>(map, indexes, assoc_ids, values, scores, queue);
  }

  template <typename TAcc,
            typename Values,
            typename Score,
            typename Collection1,
            typename Collection2,
            typename TQueue,
            typename = std::enable_if_t<alpaka::isQueue<TQueue>>,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_HOST void fill(AssociationMapView<Values, Score, Collection1, Collection2> map,
                           const uint32_t* indexes,
                           const uint32_t* assoc_ids,
                           const Values* values,
                           TQueue& queue) {
    const Score* scores = nullptr;
    detail::fill<TAcc, Values, Score, Collection1, Collection2, TQueue>(map, indexes, assoc_ids, values, scores, queue);
  }

  template <typename TAcc,
            typename Values,
            typename Score,
            typename Collection1,
            typename Collection2,
            typename TQueue,
            typename = std::enable_if_t<alpaka::isQueue<TQueue>>,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_HOST void fill(AssociationMapView<Values, Score, Collection1, Collection2> map,
                           const uint32_t* indexes,
                           const uint32_t* assoc_ids,
                           const Values* values,
                           const Score* scores,
                           TQueue& queue) {
    detail::fill<TAcc, Values, Score, Collection1, Collection2, TQueue>(map, indexes, assoc_ids, values, scores, queue);
  }

}  // namespace TICL
