#ifndef __RecoHGCal_TICL_CLUEsteringAlgo_H__
#define __RecoHGCal_TICL_CLUEsteringAlgo_H__
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "Points.h"
#include "PointsAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <uint8_t Ndim>
  class CLUEsteringAlgo {
  public:
    void make_clusters(Points<Ndim>& h_points, PointsAlpaka<Ndim>& d_points, Queue queue_, std::size_t block_size) {
    std::cout << "************Inside make_clusters***************" << std::endl;

    alpaka::memcpy(queue_, d_points.coords, cms::alpakatools::make_host_view(h_points.m_coords.data(), h_points.n));
    alpaka::memcpy(queue_, d_points.weight, cms::alpakatools::make_host_view(h_points.m_weight.data(), h_points.n));
  }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif /*CLUEsteringAlgo_H_ */
