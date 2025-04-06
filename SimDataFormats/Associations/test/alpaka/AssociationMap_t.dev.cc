

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "SimDataFormats/Associations/interface/AssociationMap.h"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <iostream>
#include <numeric>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

TEST_CASE("Test Association map diving range of numbers in even and odd") {
  const auto nPoints = 1024u;
  const auto nAssociations = 1024u;
  const auto nbins = 2u;
  auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
  auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);

  std::vector<uint32_t> indexes(nPoints);
  std::vector<uint32_t> associations(nPoints);
  std::iota(indexes.begin(), indexes.end(), 0);
  std::transform(indexes.begin(), indexes.end(), associations.begin(), [](auto idx) -> int32_t {
    return static_cast<uint32_t>(idx % 2);
  });
  std::vector<TICL::FractionType> values(nPoints, TICL::FractionType(1.f));
  std::vector<float> scores(nPoints, 1.f);

  Queue queue(device);
  auto d_indexes = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nPoints);
  auto d_associations = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nPoints);
  auto d_values = cms::alpakatools::make_device_buffer<TICL::FractionType[]>(queue, nPoints);
  auto d_scores = cms::alpakatools::make_device_buffer<float[]>(queue, nPoints);
  alpaka::memcpy(queue, d_indexes, indexes, nPoints);
  alpaka::memcpy(queue, d_associations, associations, nPoints);
  alpaka::memcpy(queue, d_values, values, nPoints);
  alpaka::memcpy(queue, d_scores, scores, nPoints);

  PortableCollection<TICL::mapWithFractionAndScore<void, void>, decltype(devHost)> associationMaph(
      devHost, (int)nAssociations, (int)nbins);
  PortableCollection<TICL::mapWithFractionAndScore<void, void>, Device> associationMap(device, (int)nAssociations, (int)nbins);
  TICL::fill<Acc1D, TICL::FractionType, float, void, void, Queue>(
      associationMap.view(), d_indexes.data(), d_associations.data(), d_values.data(), d_scores.data(), queue);
  alpaka::memcpy(queue, associationMaph.buffer(), associationMap.buffer());
  for(auto i = 0u; i < nPoints; ++i) {
	if (associationMaph.view().indexes()[i] % 2 == 0) {
	  REQUIRE(i < (nPoints / 2));
	} else {
	  REQUIRE(i >= (nPoints / 2));
	}
  }

  auto* h_offsets = associationMaph.view().metadata().offsets();
  const auto* d_offsets = associationMap.view().metadata().offsets();
  alpaka::memcpy(queue,
                 cms::alpakatools::make_host_view(h_offsets, nbins),
                 cms::alpakatools::make_device_view(device, d_offsets, nbins));
  REQUIRE(h_offsets[0] == 0);
  REQUIRE(h_offsets[1] == 512);
}
