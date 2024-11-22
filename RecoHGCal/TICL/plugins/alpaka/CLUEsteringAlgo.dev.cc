// // Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include <iostream>
#include "CLUEsteringAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;

  // template <uint8_t Ndim>
  // void CLUEsteringAlgo<Ndim>::make_clusters(Points<Ndim>& h_points,
  //                                                 PointsAlpaka<Ndim>& d_points,
  //                                                 Queue queue_,
  //                                                 std::size_t block_size) {
  //   std::cout << "************Inside make_clusters***************" << std::endl;

  //   alpaka::memcpy(queue_, d_points.coords, cms::alpakatools::make_host_view(h_points.m_coords.data(), h_points.n));
  //   alpaka::memcpy(queue_, d_points.weight, cms::alpakatools::make_host_view(h_points.m_weight.data(), h_points.n));
  // }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
