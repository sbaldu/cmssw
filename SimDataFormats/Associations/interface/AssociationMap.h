
#pragma once

#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <cassert>

#include <limits>

#include <alpaka/alpaka.hpp>

// CMSSW specific includes
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // Define wrapper types to differentiate between fraction and shared energy
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

  template <typename ValueType, typename Score>
  struct AssociationElementsSoAStruct {
    GENERATE_SOA_LAYOUT(Layout, SOA_COLUMN(ValueType, values), SOA_COLUMN(Score, scores), SOA_COLUMN(int, indexes))
  };

  template <typename ValueType>
  struct AssociationElementsSoAStruct<ValueType, void> {
    GENERATE_SOA_LAYOUT(Layout, SOA_COLUMN(ValueType, values), SOA_COLUMN(int, indexes))
  };

  template <typename ValueType, typename Score>
  using AssociationElementsSoA = typename AssociationElementsSoAStruct<ValueType, Score>::template Layout<>;
  template <typename ValueType, typename Score>
  using AssociationElementsSoAView = typename AssociationElementsSoAStruct<ValueType, Score>::template Layout<>::View;

  template <typename TFunc>
  struct KernelComputeAssociations {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, const int* indexes, size_t size, int* associations, int* nbins, TFunc func) const {
      auto max = 0;
      for (auto i : uniform_elements(acc, size)) {
        associations[i] = func(indexes[i]);
        if (associations[i] > max) {
          max = associations[i];
        }
      }
      *nbins = max + 1;
    }
  };

  // Note: bad name. Find a better one.
  struct KernelComputeAssociationSizes {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, const int* associations, int* sizes /*tempname*/, size_t size) const {
      for (auto i : uniform_elements(acc, size)) {
        alpaka::atomicAdd(acc, &sizes[associations[i]], 1);
      }
    }
  };

  template <typename V, typename Score, std::enable_if_t<!std::is_void_v<Score>, int> = 0>
  ALPAKA_FN_ACC void insert(AssociationElementsSoAView<V, Score>& assoc_soa_view,
                            size_t size,
                            int offset,
                            int index,
                            V fraction_or_energy,
                            Score score = 0.0) {
    assert(static_cast<size_t>(index) < size);
    assoc_soa_view.values()[offset] = fraction_or_energy;
    assoc_soa_view.scores()[offset] = score;
    assoc_soa_view.indexes()[offset] = index;
  }

  template <typename V, typename Score, std::enable_if_t<std::is_void_v<Score>, int> = 0>
  ALPAKA_FN_ACC void insert(
      AssociationElementsSoAView<V, void>& assoc_soa_view, size_t size, int offset, int index, V fraction_or_energy) {
    assert(static_cast<size_t>(index) < size);
    assoc_soa_view.values()[offset] = fraction_or_energy;
    assoc_soa_view.indexes()[offset] = index;
  }

  template <typename TDev, typename V, typename Score, typename Collection1, typename Collection2>
  struct KernelFillAssociator {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  AssociationElementsSoAView<V, Score> assoc_soa_view,
                                  const int* bin_buffer,
                                  const V* values,
                                  const Score* scores,
                                  int* temp_offsets,
                                  size_t size) const {
      for (auto i : uniform_elements(acc, size)) {
        const auto binId = bin_buffer[i];
        const auto position = temp_offsets[binId];
        insert(assoc_soa_view, size, position, i, values[i], scores[i]);
        alpaka::atomicAdd(acc, &temp_offsets[binId], 1);
      }
    }
  };

  template <typename TDev, typename V, typename Collection1, typename Collection2>
  struct KernelFillAssociator<TDev, V, void, Collection1, Collection2> {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  AssociationElementsSoAView<V, void> assoc_soa_view,
                                  const int* bin_buffer,
                                  const V* values,
                                  int* temp_offsets,
                                  size_t size) const {
      for (auto i : uniform_elements(acc, size)) {
        const auto binId = bin_buffer[i];
        const auto position = temp_offsets[binId];
        insert(assoc_soa_view, size, position, i, values[i]);
        alpaka::atomicAdd(acc, &temp_offsets[binId], 1);
      }
    }
  };

  /*
  template <typename TDev, typename Collection1, typename Collection2>
  struct KernelFillAssociator<TDev, void, void, Collection1, Collection2> {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  AssociationElementsSoAView<V, Score> assoc_soa_view,
                                  const int* bin_buffer,
                                  int* temp_offsets,
                                  size_t size) const {
      for (auto i : uniform_elements(acc, size)) {
        const auto binId = bin_buffer[i];
        const auto position = temp_offsets[binId];
        insert(assoc_soa_view, size, position, i);
        alpaka::atomicAdd(acc, &temp_offsets[binId], 1);
      }
    }
  };
  */

  template <typename TDev, typename V, typename Score = void, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class AssociationElements {
  private:
    PortableCollection<AssociationElementsSoA<V, Score>, TDev> m_data;

  public:
    using value_type = V;
    using score_type = Score;
    static constexpr bool has_score = std::is_void_v<Score>;

    AssociationElements(size_t size, const TDev& dev) : m_data(size, dev) {}

    ALPAKA_FN_HOST_ACC bool isValid(size_t i) {
      if constexpr (has_score) {
        return m_data.view().scores(i) >= 0.f;
      } else {
        return m_data.view().values(i) >= 0.f;
      }
    }

    ALPAKA_FN_HOST auto view() { return m_data.view(); }

    // Enable fraction() if ValueType is FractionType
    /*
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, FractionType>, int> = 0>
    ALPAKA_FN_HOST_ACC float fraction(size_t i) const {
        return m_data.view().values[i];
    }

    // Enable sharedEnergy() if ValueType is SharedEnergyType
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, SharedEnergyType>, int> = 0>
    ALPAKA_FN_HOST_ACC float sharedEnergy(size_t i) const {
        return m_data.view().values[i];
    }

    template <typename T = Score, typename std::enable_if_t<std::is_void_v<T>, int> = 0>
    ALPAKA_FN_HOST_ACC float score(size_t i) const {
        return m_data.view().scores[i];
    }
    */
  };

  template <typename TDev,
            typename V,
            typename Score = void,
            typename Collection1 = void,
            typename Collection2 = void,
            typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class AssociationMap {
  private:
    AssociationElements<TDev, V, Score> m_associations;
    device_buffer<TDev, int[]> m_offsets;
    size_t m_size;

    using CollectionRefProdType =
        typename std::conditional_t<std::is_void_v<Collection1> || std::is_void_v<Collection2>,
                                    std::monostate,
                                    std::pair<edm::RefProd<Collection1>, edm::RefProd<Collection2>>>;

    CollectionRefProdType collectionRefProds;

    using value_type = V;
    static constexpr bool has_score = AssociationElements<V, Score>::has_score;

    template <typename T>
    struct Span {
      T* buf;
      int m_size;

      ALPAKA_FN_ACC T* data() { return buf; }
      ALPAKA_FN_ACC const T* data() const { return buf; }
      ALPAKA_FN_ACC int size() const { return m_size; }
    };

  public:
    // Constructors for generic use
    AssociationMap(size_t size, size_t nbins, const TDev& dev)
        : m_associations(size, dev),
          m_offsets{make_device_buffer<int[]>(dev, nbins)},
          m_size{nbins},
          collectionRefProds() {}

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    AssociationMap(size_t size, size_t nbins, const TQueue& queue)
        : m_associations(size, queue),
          m_offsets{make_device_buffer<int[]>(queue, nbins)},
          m_size{nbins},
          collectionRefProds() {}

    // Constructor for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    AssociationMap(size_t size,
                   size_t nbins,
                   const TDev& dev,
                   const edm::RefProd<C1>& id1,
                   const edm::RefProd<C2>& id2,
                   const edm::Event& event)
        : m_associations(size, dev),
          m_offsets{make_device_buffer<int[]>(dev, nbins)},
          m_size{nbins},
          collectionRefProds(std::make_pair(id1, id2)) {
      //resize(event);
    }

    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename TQueue,
              std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0,
              typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    AssociationMap(size_t size,
                   size_t nbins,
                   const TQueue& queue,
                   const edm::RefProd<C1>& id1,
                   const edm::RefProd<C2>& id2,
                   const edm::Event& event)
        : m_associations(size, queue),
          m_offsets{make_device_buffer<int[]>(queue, nbins)},
          m_size{nbins},
          collectionRefProds(std::make_pair(id1, id2)) {
      //resize(event);
    }

    // Constructor for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename TQueue,
              std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0,
              typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    AssociationMap(size_t size,
                   size_t nbins,
                   const TQueue& queue,
                   const edm::Handle<C1>& handle1,
                   const edm::Handle<C2>& handle2,
                   const edm::Event& event)
        : m_associations(size, queue),
          m_offsets{make_device_buffer<int[]>(queue, nbins)},
          m_size{nbins},
          collectionRefProds(std::make_pair(edm::RefProd<C1>(handle1), edm::RefProd<C2>(handle2))) {
      //resize(event);
    }

    auto size() const { return m_size; }

    auto view() { return m_associations.view(); }

    ALPAKA_FN_ACC Span<V> values(size_t assoc_id) {
      auto size = m_offsets[assoc_id + 1] - m_offsets[assoc_id];
      auto* buf_ptr = m_associations.view().values() + m_offsets[assoc_id];
      return Span<V>{buf_ptr, size};
    }
    ALPAKA_FN_HOST device_view<TDev, V[]> values(const TDev& dev, size_t assoc_id) {
      auto size = m_offsets[assoc_id + 1] - m_offsets[assoc_id];
      auto* buf_ptr = m_associations.view().values() + m_offsets[assoc_id];
      return make_device_view<V[], TDev>(dev, buf_ptr, size);
    }

    ALPAKA_FN_ACC Span<Score> scores(size_t assoc_id) {
      auto size = m_offsets[assoc_id + 1] - m_offsets[assoc_id];
      auto* buf_ptr = m_associations.view().scores() + m_offsets[assoc_id];
      return Span<Score>{buf_ptr, size};
    }
    ALPAKA_FN_HOST device_view<TDev, Score[]> scores(const TDev& dev, size_t assoc_id) {
      auto size = m_offsets[assoc_id + 1] - m_offsets[assoc_id];
      auto* buf_ptr = m_associations.view().scores() + m_offsets[assoc_id];
      return make_device_view<Score[], TDev>(dev, buf_ptr, size);
    }

    ALPAKA_FN_ACC Span<int> indexes(size_t assoc_id) {
      auto size = m_offsets[assoc_id + 1] - m_offsets[assoc_id];
      auto* buf_ptr = m_associations.view().indexes() + m_offsets[assoc_id];
      return Span<int>{buf_ptr, size};
    }
    ALPAKA_FN_HOST device_view<TDev, int[]> indexes(const TDev& dev, size_t assoc_id) {
      auto size = m_offsets[assoc_id + 1] - m_offsets[assoc_id];
      auto* buf_ptr = m_associations.view().indexes() + m_offsets[assoc_id];
      return make_device_view<int[], TDev>(dev, buf_ptr, size);
    }

    ALPAKA_FN_HOST device_buffer<TDev, int[]>& offsets() { return m_offsets; }
    ALPAKA_FN_ACC int offsets(size_t assoc_id) const { return m_offsets[assoc_id]; }

    // CMSSW-specific method to get references
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    edm::Ref<C1> getRefFirst(unsigned int index) const {
      return edm::Ref<C1>(collectionRefProds.first, index);
    }

    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    edm::Ref<C2> getRefSecond(unsigned int index) const {
      return edm::Ref<C2>(collectionRefProds.second, index);
    }

    // Method to get collection IDs for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    std::pair<const edm::RefProd<C1>, const edm::RefProd<C2>> getCollectionIDs() const {
      return collectionRefProds;
    }

    template <typename TFunc>
    ALPAKA_FN_HOST void fill(
        const int* indexes, const V* values, const Score* scores, size_t size, TFunc func, const TDev& dev) {
      auto nbins_buffer = make_device_buffer<int>(dev);
      auto bin_buffer = make_device_buffer<int[]>(dev, size);

      Queue queue(dev);
      const auto blocksize = 512;
      const auto gridsize = divide_up_by(size, blocksize);
      const auto workdiv = make_workdiv<Acc1D>(gridsize, blocksize);
      alpaka::exec<Acc1D>(queue,
                          workdiv,
                          KernelComputeAssociations<TFunc>{},
                          indexes,
                          size,
                          bin_buffer.data(),
                          nbins_buffer.data(),
                          func);

      int nbins = 0;
      alpaka::memcpy(queue, make_host_view<int>(nbins), nbins_buffer);
      m_size = nbins;
      m_offsets = make_device_buffer<int[]>(dev, nbins + 1);
      auto sizes_buffer = make_device_buffer<int[]>(dev, nbins);
      alpaka::memset(queue, sizes_buffer, 0);
      alpaka::exec<Acc1D>(
          queue, workdiv, KernelComputeAssociationSizes{}, bin_buffer.data(), sizes_buffer.data(), size);

      // prepare for prefix scan
      auto block_counter = make_device_buffer<int32_t>(queue);
      alpaka::memset(queue, block_counter, 0);

      alpaka::memset(queue, m_offsets, 0);

      const auto blocksize_multiblockscan = 1024;
      auto gridsize_multiblockscan = divide_up_by(nbins, blocksize_multiblockscan);  // think about the size
      const auto workdiv_multiblockscan = make_workdiv<Acc1D>(gridsize_multiblockscan, blocksize_multiblockscan);
      auto warp_size = alpaka::getPreferredWarpSize(dev);
      alpaka::exec<Acc1D>(queue,
                          workdiv_multiblockscan,
                          multiBlockPrefixScan<int>{},
                          sizes_buffer.data(),
                          m_offsets.data() + 1,
                          nbins,
                          gridsize_multiblockscan,
                          block_counter.data(),
                          warp_size);

      auto temp_offsets = make_device_buffer<int[]>(queue, nbins + 1);
      alpaka::memcpy(queue, temp_offsets, m_offsets);
      alpaka::exec<Acc1D>(queue,
                          workdiv,
                          KernelFillAssociator<TDev, V, Score, void, void>{},
                          this->view(),
                          bin_buffer.data(),
                          values,
                          scores,
                          temp_offsets.data(),
                          size);
    }

    /*
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    void fill(const edm::Ref<C1>& ref1, const edm::Ref<C2>& ref2, float fraction_or_energy, float score = 0.0f) {
      auto assoc_id = ref1.key();
      auto index = ref2.key();
      insert(ref1.key(), ref2.key(), fraction_or_energy, score);
    }
    */
  };

  template <typename V,
            typename Score,
            typename TFunc,
            typename TDev,
            typename = std::enable_if_t<alpaka::isDevice<TDev>>,
            std::enable_if_t<!std::is_void_v<Score>, int> = 0>
  ALPAKA_FN_HOST AssociationMap<TDev, V, Score> CreateAssociationMap(
      const int* indexes, const V* values, const Score* scores, size_t size, TFunc func, const TDev& dev) {
    auto nbins_buffer = make_device_buffer<int>(dev);
    auto bin_buffer = make_device_buffer<int[]>(dev, size);

    Queue queue(dev);
    const auto blocksize = 512;
    const auto gridsize = divide_up_by(size, blocksize);
    const auto workdiv = make_workdiv<Acc1D>(gridsize, blocksize);
    alpaka::exec<Acc1D>(
        queue, workdiv, KernelComputeAssociations<TFunc>{}, indexes, size, bin_buffer.data(), nbins_buffer.data(), func);

    auto nbins = *nbins_buffer.data();
    auto sizes_buffer = make_device_buffer<int[]>(dev, nbins);
    alpaka::memset(queue, sizes_buffer, 0);
    alpaka::exec<Acc1D>(queue, workdiv, KernelComputeAssociationSizes{}, bin_buffer.data(), sizes_buffer.data(), size);

    AssociationMap<TDev, V, Score> assoc_map(size, nbins + 1, dev);

    // prepare for prefix scan
    auto block_counter = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, block_counter, 0);

    alpaka::memset(queue, assoc_map.offsets(), 0);

    const auto blocksize_multiblockscan = 1;
    auto gridsize_multiblockscan = divide_up_by(nbins, blocksize_multiblockscan);  // think about the size
    const auto workdiv_multiblockscan = make_workdiv<Acc1D>(gridsize_multiblockscan, blocksize_multiblockscan);
    auto warp_size = alpaka::getPreferredWarpSize(dev);
    alpaka::exec<Acc1D>(queue,
                        workdiv_multiblockscan,
                        multiBlockPrefixScan<int>{},
                        sizes_buffer.data(),
                        assoc_map.offsets().data() + 1,
                        nbins,
                        gridsize_multiblockscan,
                        block_counter.data(),
                        warp_size);

    auto temp_offsets = make_device_buffer<int[]>(queue, nbins + 1);
    alpaka::memcpy(queue, temp_offsets, assoc_map.offsets());
    alpaka::exec<Acc1D>(queue,
                        workdiv,
                        KernelFillAssociator<TDev, V, Score, void, void>{},
                        assoc_map.view(),
                        bin_buffer.data(),
                        values,
                        scores,
                        temp_offsets.data(),
                        size);

    return assoc_map;
  }

  template <typename V,
            typename Score,
            typename TFunc,
            typename TDev,
            typename = std::enable_if_t<alpaka::isDevice<TDev>>,
            std::enable_if_t<std::is_void_v<Score>, int> = 0>
  ALPAKA_FN_HOST AssociationMap<TDev, V, Score> CreateAssociationMap(
      const int* indexes, const V* values, size_t size, TFunc func, const TDev& dev) {
    auto nbins_buffer = make_device_buffer<int>(dev);
    auto bin_buffer = make_device_buffer<int[]>(dev, size);

    Queue queue(dev);
    const auto blocksize = 512;
    const auto gridsize = divide_up_by(size, blocksize);
    const auto workdiv = make_workdiv<Acc1D>(gridsize, blocksize);
    alpaka::exec<Acc1D>(
        queue, workdiv, KernelComputeAssociations<TFunc>{}, indexes, size, bin_buffer.data(), nbins_buffer.data(), func);

    auto nbins = *nbins_buffer.data();
    auto sizes_buffer = make_device_buffer<int[]>(dev, nbins);
    alpaka::memset(queue, sizes_buffer, 0);
    alpaka::exec<Acc1D>(queue, workdiv, KernelComputeAssociationSizes{}, bin_buffer.data(), sizes_buffer.data(), size);

    AssociationMap<TDev, V, Score> assoc_map(size, nbins + 1, dev);

    // prepare for prefix scan
    auto block_counter = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, block_counter, 0);

    const auto blocksize_multiblockscan = 1024;
    auto gridsize_multiblockscan = divide_up_by(size, blocksize_multiblockscan);  // think about the size
    const auto workdiv_multiblockscan = make_workdiv<Acc1D>(gridsize_multiblockscan, blocksize_multiblockscan);
    auto warp_size = alpaka::getPreferredWarpSize(dev);
    alpaka::exec<Acc1D>(queue,
                        workdiv_multiblockscan,
                        multiBlockPrefixScan<int>{},
                        sizes_buffer.data(),
                        assoc_map.offsets().data() + 1,
                        size,
                        gridsize_multiblockscan,
                        block_counter.data(),
                        warp_size);

    auto temp_offsets = make_device_buffer<int[]>(queue, nbins + 1);
    alpaka::memcpy(queue, temp_offsets, assoc_map.offsets());
    alpaka::exec<Acc1D>(queue,
                        workdiv,
                        KernelFillAssociator<TDev, V, Score, void, void>{},
                        assoc_map.view(),
                        bin_buffer.data(),
                        values,
                        temp_offsets.data(),
                        size);

    return assoc_map;
  }

  /*
  template <typename V, typename TFunc, typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  ALPAKA_FN_HOST AssociationMap<TDev, V> CreateAssociationMap(const int* indexes,
                                                              const V* values,
                                                              size_t size,
                                                              const TFunc* func,
                                                              const TQueue& queue) {
      auto device = alpaka::getDevs(queue);
      return CreateAssociationMap(indexes, values, size, func, device);
  }
  */

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
