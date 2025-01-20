
#pragma once

#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <cassert>

#include <limits>

// CMSSW specific includes
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace ticl {

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

  template <typename TDev, typename V, typename Score = void, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class AssociationElements {
  private:
    PortableCollection<AssociationElementsSoA<V, Score>, TDev> m_data;

  public:
    using value_type = V;
    using score_type = std::enable_if_t<!std::is_void_v<Score>, Score>;
    static constexpr bool has_score = std::is_void_v<Score>;

    AssociationElements(size_t size, const TDev& dev) : m_data(size, dev) {}

    ALPAKA_FN_HOST_ACC bool isValid(size_t i) {
      if constexpr (has_score) {
        return m_data.view()->scores(i) >= 0.f;
      } else {
        return m_data.view()->values(i) >= 0.f;
      }
    }

    // Enable fraction() if ValueType is FractionType
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, FractionType>, int> = 0>
    ALPAKA_FN_HOST_ACC float fraction(size_t i) const {}

    // Enable sharedEnergy() if ValueType is SharedEnergyType
    template <typename T = V, typename std::enable_if_t<std::is_same_v<T, SharedEnergyType>, int> = 0>
    ALPAKA_FN_HOST_ACC float sharedEnergy(size_t i) const {}

    template <typename T = Score, typename std::enable_if_t<std::is_void_v<T>, int> = 0>
    ALPAKA_FN_HOST_ACC float score(size_t i) const {}

    // Method to accumulate values
    /*
    void accumulate(const V& other_value) {
      if constexpr (std::is_same_v<V, FractionType> || std::is_same_v<V, SharedEnergyType>) {
        value_.value += other_value.value;
      } else if constexpr (std::is_same_v<V, std::pair<FractionType, float>> ||
                           std::is_same_v<V, std::pair<SharedEnergyType, float>>) {
        value_.first.value += other_value.first.value;
        value_.second += other_value.second;
      }
    }
    bool operator==(const AssociationElement& other) const {
      return index_ == other.index_ && value_.value == other.value_.value;
    }

    bool operator!=(const AssociationElement& other) const { return !(*this == other); }
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
    AssociationElements<V, Score> m_associations;
    cms::alpakatools::device_buffer<TDev, int> m_offsets;
    size_t m_size;  // std::span?

    using CollectionRefProdType =
        typename std::conditional_t<std::is_void_v<Collection1> || std::is_void_v<Collection2>,
                                    std::monostate,
                                    std::pair<edm::RefProd<Collection1>, edm::RefProd<Collection2>>>;

    CollectionRefProdType collectionRefProds;

    using value_type = V;
    using has_score = AssociationElements<V, Score>::has_score;

    // TODO
    //using Traits = MapTraits<MapType>;
    //using AssociationElementType = typename Traits::AssociationElementType;
    //static constexpr bool is_one_to_one = Traits::is_one_to_one;

  public:
    // Constructors for generic use
    AssociationMap(size_t size, const TDev& dev)
        : m_associations(size, dev),
          m_keys{cms::alpakatools::make_device_buffer<int[]>(dev, size)},
          m_size{size},
          collectionRefProds() {}

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    AssociationMap(size_t size, const TQueue& queue)
        : m_associations(size, queue),
          m_keys{cms::alpakatools::make_device_buffer<int[]>(queue, size)},
          m_size{size},
          collectionRefProds() {}

    // Constructor for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0>
    AssociationMap(
        size_t size, const TDev& dev, const edm::RefProd<C1>& id1, const edm::RefProd<C2>& id2, const edm::Event& event)
        : m_associations(size, dev), m_size{size}, collectionRefProds(std::make_pair(id1, id2)) {
      resize(event);
    }

    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename TQueue,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0,
              typename std::enable_if_t<alpaka::isQueue<TQueue>>>
    AssociationMap(
        size_t size, const TQueue& queue, const edm::RefProd<C1>& id1, const edm::RefProd<C2>& id2, const edm::Event& event)
        : m_associations(size, queue), m_size{size}, collectionRefProds(std::make_pair(id1, id2)) {
      resize(event);
    }


    // Constructor for CMSSW-specific use
    template <typename C1 = Collection1,
              typename C2 = Collection2,
              typename TQueue,
              typename std::enable_if_t<!std::is_void_v<C1> && !std::is_void_v<C2>, int> = 0,
              typename std::enable_if_t<alpaka::isQueue<TQueue>>>
    AssociationMap(size_t size,
                   const TQueue& queue,
                   const edm::Handle<C1>& handle1,
                   const edm::Handle<C2>& handle2,
                   const edm::Event& event)
        : m_associations(size, queue),
          m_size{size},
          collectionRefProds(std::make_pair(edm::RefProd<C1>(handle1), edm::RefProd<C2>(handle2))) {
      resize(event);
    }

    auto size() const { return m_size; }

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

    ALPAKA_FN_ACC void insert(int index1, int index2, float fraction_or_energy, float score = 0.0) {
      assert(index1 < m_size);
      if constexpr (has_score) {
      } else {
      }
    }
  };

  struct KernelFillAssociator {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()() const {}
  };


  // TODO
  /*
  template <typename TFunc>
  struct KernelComputeAssociations {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, const int* indexes, size_t size, int* associations, TFunc func) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
        associations[i] = func(indexes[i]);
      }
    }
  };

  template <typename V, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  ALPAKA_FN_HOST AssociationMap CreateAssociationMap(const int* indexes, const V* values, size_t size, const TDev& dev) {
      auto bin_buffer = cms::alpakatools::make_device_buffer<int[]>(dev, size);

      const auto blocksize = 256;
      const auto gridsize = cms::alpakatools::divide_up_by(size, blocksize);
      const auto workdiv = cms::alpakatools::make_workdiv(gridsize, blocksize);
      alpaka::enqueue(queue, alpaka::CreateTaskKernel(workdiv, KernelComputeAssociations{}));
  }

  template <typename V, typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  ALPAKA_FN_HOST AssociationMap CreateAssociationMap(const int* indexes, const V* values, size_t size, const TQueue& queue) {
      auto bin_buffer = cms::alpakatools::make_device_buffer<>();
  }
  */

}  // namespace ticl

