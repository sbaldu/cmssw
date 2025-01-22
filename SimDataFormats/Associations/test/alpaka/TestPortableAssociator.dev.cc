
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "SimDataFormats/Associations/interface/AssociationMap.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

TEST_CASE("SimDataFormats/Associations test") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
                                "the test will be skipped.");
  }

  // run the test on all available devices
  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    SECTION("Test Association_Elements_constructor on " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend") {
      REQUIRE_NOTHROW([&]() {
        size_t size_points = 10;
        AssociationElements<Device, FractionType> points(size_points, device);
      }());
    }
  }

  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    SECTION("Test Association_Elements_constructor on " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend") {
      REQUIRE_NOTHROW([&]() {
        const auto size = 100;
        Queue queue(device);
        auto h_points = cms::alpakatools::make_host_buffer<int[]>(size);
        auto h_values = cms::alpakatools::make_host_buffer<FractionType[]>(size);
        auto h_scores = cms::alpakatools::make_host_buffer<float[]>(size);
        std::iota(h_points.data(), h_points.data() + size, 0);
        std::fill(h_scores.data(), h_scores.data() + size, .5f);

        auto d_points = cms::alpakatools::make_device_buffer<int[]>(queue, size);
        auto d_values = cms::alpakatools::make_device_buffer<FractionType[]>(queue, size);
        auto d_scores = cms::alpakatools::make_device_buffer<float[]>(queue, size);
        alpaka::memcpy(queue, d_points, h_points);
        alpaka::memcpy(queue, d_values, h_values);
        alpaka::memcpy(queue, d_scores, h_scores);

        auto func = [] ALPAKA_FN_ACC (int value) -> int { return (value % 2 == 0) ? 1 : 0; };

        std::cout << __LINE__ << std::endl;
        auto map =
            CreateAssociationMap<FractionType, float, decltype(func), Device>(d_points.data(), d_values.data(), d_scores.data(), size, func, device);
        std::cout << __LINE__ << std::endl;
      }());
    }
  }
}
