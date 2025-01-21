
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <alpaka/alpaka.hpp>
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
        const int points[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        const FractionType fractions[10] = {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f};

        auto func = [](int value) -> int { return (value % 2 == 0) ? 1 : 0; };

        AssociationMap<FractionType, int, Device> map =
            CreateAssociationMap(points, fractions, points.size(), device);
      }());
    }
  }
}
