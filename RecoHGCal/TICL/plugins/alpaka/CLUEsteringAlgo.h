#ifndef __RecoHGCal_TICL_CLUEsteringAlgo_H__
#define __RecoHGCal_TICL_CLUEsteringAlgo_H__

#include <alpaka/core/Common.hpp>
#include <chrono>
#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Points.h"
#include "PointsAlpaka.h"
#include "TilesAlpaka.h"

using namespace cms::alpakatools;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  constexpr int32_t max_followers{100};
  constexpr int32_t max_seeds{100};
  constexpr int32_t reserve{1000000};

  template <uint8_t Ndim>
  using PointsView = typename PointsAlpaka<Ndim>::PointsAlpakaView;

  class KernelResetTiles {
  public:
    template <typename TAcc, uint8_t Ndim>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  TilesAlpaka<Ndim>* tiles,
                                  uint32_t nTiles,
                                  uint32_t nPerDim) const {                                    
      std::cout<<"////////////////KernelResetTiles LAUNCHED////////////////"<<std::endl;
      if (once_per_grid(acc)) {
        tiles->resizeTiles(nTiles, nPerDim);
      }

      for (uint32_t i : uniform_elements(acc, nTiles)) {
        tiles->clear(i);
      }
    }
  };

  class KernelResetFollowers {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  VecArray<int, max_followers>* d_followers,
                                  uint32_t n_points) const {      
      std::cout<<"////////////////KernelResetFollowers LAUNCHED////////////////"<<std::endl;    
      for (uint32_t i : uniform_elements(acc, n_points)) {
        d_followers[i].reset();
      }      
    }
  };

  class KernelFillTiles {
  public:
    template <typename TAcc, uint8_t Ndim>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  PointsView<Ndim>* points,
                                  TilesAlpaka<Ndim>* tiles,
                                  uint32_t n_points) const {
      std::cout<<"////////////////KernelFillTiles LAUNCHED////////////////"<<std::endl;
      for (uint32_t i : uniform_elements(acc, n_points)) {
        tiles->fill(acc, points->coords[i], i);
      }           
    }
  };

  template <typename TAcc, uint8_t Ndim>
  class CLUEsteringAlgo {
  public:
    CLUEsteringAlgo() = delete;
    explicit CLUEsteringAlgo(float dc, float rhoc, float dm, int pPBin, Queue queue_)
        : dc_{dc}, rhoc_{rhoc}, dm_{dm}, pointsPerTile_{pPBin} {
      init_device(queue_);
    }

    TilesAlpaka<Ndim>* m_tiles;
    VecArray<int32_t, max_seeds>* m_seeds;
    VecArray<int32_t, max_followers>* m_followers;

    void make_clusters(Points<Ndim>& h_points, PointsAlpaka<Ndim>& d_points, Queue queue_, std::size_t block_size) {
    std::cout<<"*****************MAKING CLUSTERS IN ALPAKA***************"<<std::endl;
    setup(h_points, d_points, queue_, block_size);

    const Idx grid_size = cms::alpakatools::divide_up_by(h_points.n, block_size);
    auto working_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(
        queue_,
        alpaka::createTaskKernel<Acc1D>(
            working_div, KernelFillTiles{}, d_points.view(), m_tiles, h_points.n));

    }

  private:
    float dc_;
    float rhoc_;
    float dm_;
    // average number of points found in a tile
    int pointsPerTile_;      

    // Buffers
    std::optional<cms::alpakatools::device_buffer<Device, TilesAlpaka<Ndim>>> d_tiles;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int32_t, max_seeds>>> d_seeds;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int32_t, max_followers>[]>> d_followers;

    // Private methods
    void init_device(Queue queue_) {
      d_tiles = cms::alpakatools::make_device_buffer<TilesAlpaka<Ndim>>(queue_);
      d_seeds = cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int32_t, max_seeds>>(queue_);
      d_followers = cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int32_t, max_followers>[]>(queue_, reserve);

      // Copy to the public pointers
      m_tiles = (*d_tiles).data();
      m_seeds = (*d_seeds).data();
      m_followers = (*d_followers).data();
    }

    void setup(const Points<Ndim>& h_points,
               PointsAlpaka<Ndim>& d_points,
               Queue queue_,
               std::size_t block_size) {
      // calculate the number of tiles and their size
      const auto nTiles{std::ceil(h_points.n / static_cast<float>(pointsPerTile_))};
      const auto nPerDim{std::ceil(std::pow(nTiles, 1. / Ndim))};

      CoordinateExtremes<Ndim> min_max;
      float tile_size[Ndim];
      calculate_tile_size(min_max, tile_size, h_points, nPerDim);

      const auto device = alpaka::getDev(queue_);
      alpaka::memcpy(
          queue_,
          cms::alpakatools::make_device_view(device, (*d_tiles)->minMax(), 2 * Ndim),
          cms::alpakatools::make_host_view(min_max.data(), 2 * Ndim));
      alpaka::memcpy(
          queue_,
          cms::alpakatools::make_device_view(device, (*d_tiles)->tileSize(), Ndim),
          cms::alpakatools::make_host_view(tile_size, Ndim));
      alpaka::wait(queue_);

      const Idx tiles_grid_size = cms::alpakatools::divide_up_by(nTiles, block_size);
      const auto tiles_working_div =
          cms::alpakatools::make_workdiv<Acc1D>(tiles_grid_size, block_size);
      alpaka::enqueue(
          queue_,
          alpaka::createTaskKernel<Acc1D>(
              tiles_working_div, KernelResetTiles{}, m_tiles, nTiles, nPerDim));

      alpaka::memcpy(
          queue_,
          d_points.coords,
          cms::alpakatools::make_host_view(h_points.m_coords.data(), h_points.n));
      alpaka::memcpy(
          queue_,
          d_points.weight,
          cms::alpakatools::make_host_view(h_points.m_weight.data(), h_points.n));
      alpaka::memset(queue_, (*d_seeds), 0x00);

      // Define the working division
      const Idx grid_size = cms::alpakatools::divide_up_by(h_points.n, block_size);
      const auto working_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
      alpaka::enqueue(queue_,
                      alpaka::createTaskKernel<Acc1D>(
                          working_div, KernelResetFollowers{}, m_followers, h_points.n));
    }

    // Construction of the tiles
    void calculate_tile_size(CoordinateExtremes<Ndim>& min_max,
                             float* tile_sizes,
                             const Points<Ndim>& h_points,
                             uint32_t nPerDim) {
      for (size_t dim{}; dim != Ndim; ++dim) {
        const float dimMax{
            (*std::max_element(h_points.m_coords.begin(),
                               h_points.m_coords.end(),
                               [dim](const auto& vec1, const auto& vec2) -> bool {
                                 return vec1[dim] < vec2[dim];
                               }))[dim]};
        const float dimMin{
            (*std::min_element(h_points.m_coords.begin(),
                               h_points.m_coords.end(),
                               [dim](const auto& vec1, const auto& vec2) -> bool {
                                 return vec1[dim] < vec2[dim];
                               }))[dim]};

        min_max.min(dim) = dimMin;
        min_max.max(dim) = dimMax;

        const float tileSize{(dimMax - dimMin) / nPerDim};
        tile_sizes[dim] = tileSize;
      }
    }

  //   void make_clusters(Points<Ndim>& h_points, PointsAlpaka<Ndim>& d_points, Queue queue_, std::size_t block_size) {
  //   std::cout << "************Inside make_clusters***************" << std::endl;

  //   alpaka::memcpy(queue_, d_points.coords, cms::alpakatools::make_host_view(h_points.m_coords.data(), h_points.n));
  //   alpaka::memcpy(queue_, d_points.weight, cms::alpakatools::make_host_view(h_points.m_weight.data(), h_points.n));
  // }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif /*CLUEsteringAlgo_H_ */
