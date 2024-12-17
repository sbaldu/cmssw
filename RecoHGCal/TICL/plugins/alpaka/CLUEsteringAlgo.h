#ifndef __RecoHGCal_TICL_CLUEsteringAlgo_H__
#define __RecoHGCal_TICL_CLUEsteringAlgo_H__

#include <alpaka/core/Common.hpp>
#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Points.h"
#include "PointsAlpaka.h"
#include "TilesAlpaka.h"

using namespace cms::alpakatools;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  constexpr int32_t max_followers{100};
  constexpr int32_t reserve{1000000};

  template <uint8_t Ndim>
  using PointsView = typename PointsAlpaka<Ndim>::PointsAlpakaView;

  class KernelResetTiles {
  public:
    template <typename TAcc, uint8_t Ndim>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TilesAlpaka<Ndim>* tiles, uint32_t nTiles, uint32_t nPerDim) const {
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
    ALPAKA_FN_ACC void operator()(const TAcc& acc, VecArray<int, max_followers>* d_followers, uint32_t n_points) const {
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
      for (uint32_t i : uniform_elements(acc, n_points)) {
        tiles->fill(acc, points->coords[i], i);
      }
    }
  };

  template <typename TAcc, uint8_t Ndim, uint8_t N_, typename KernelType>
  ALPAKA_FN_HOST_ACC void for_recursion(const TAcc& acc,
                                        VecArray<uint32_t, Ndim>& base_vec,
                                        const VecArray<VecArray<uint32_t, 2>, Ndim>& search_box,
                                        TilesAlpaka<Ndim>* tiles,
                                        PointsView<Ndim>* dev_points,
                                        const KernelType& kernel,
                                        const VecArray<float, Ndim>& coords_i,
                                        float* rho_i,
                                        float dc,
                                        uint32_t point_id) {
    if constexpr (N_ == 0) {
      int binId{tiles->getGlobalBinByBin(acc, base_vec)};
      // get the size of this bin
      int binSize{static_cast<int>((*tiles)[binId].size())};

      // iterate inside this bin
      for (int binIter{}; binIter < binSize; ++binIter) {
        uint32_t j{(*tiles)[binId][binIter]};
        // query N_{dc_}(i)

        VecArray<float, Ndim> coords_j{dev_points->coords[j]};

        float dist_ij_sq{0.f};
        for (int dim{}; dim != Ndim; ++dim) {
          dist_ij_sq += (coords_j[dim] - coords_i[dim]) * (coords_j[dim] - coords_i[dim]);
        }

        if (dist_ij_sq <= dc * dc) {
          *rho_i += kernel(acc, alpaka::math::sqrt(acc, dist_ij_sq), point_id, j) * dev_points->weight[j];
        }

      }  // end of interate inside this bin

      return;
    } else {
      for (unsigned int i{search_box[search_box.capacity() - N_][0]}; i <= search_box[search_box.capacity() - N_][1];
           ++i) {
        base_vec[base_vec.capacity() - N_] = i;
        for_recursion<TAcc, Ndim, N_ - 1>(
            acc, base_vec, search_box, tiles, dev_points, kernel, coords_i, rho_i, dc, point_id);
      }
    }
  }

  struct KernelCalculateLocalDensity {
    template <typename TAcc, uint8_t Ndim, typename KernelType>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TilesAlpaka<Ndim>* dev_tiles,
                                  PointsView<Ndim>* dev_points,
                                  const KernelType& kernel,
                                  float dc,
                                  uint32_t n_points) const {
      for (uint32_t i : uniform_elements(acc, n_points)) {
        float rho_i{0.f};
        VecArray<float, Ndim> coords_i{dev_points->coords[i]};

        // Get the extremes of the search box
        VecArray<VecArray<float, 2>, Ndim> searchbox_extremes;
        for (int dim{}; dim != Ndim; ++dim) {
          VecArray<float, 2> dim_extremes;
          dim_extremes.push_back_unsafe(coords_i[dim] - dc);
          dim_extremes.push_back_unsafe(coords_i[dim] + dc);

          searchbox_extremes.push_back_unsafe(dim_extremes);
        }

        // Calculate the search box
        VecArray<VecArray<uint32_t, 2>, Ndim> search_box;
        dev_tiles->searchBox(acc, searchbox_extremes, &search_box);

        VecArray<uint32_t, Ndim> base_vec;
        for_recursion<TAcc, Ndim, Ndim>(
            acc, base_vec, search_box, dev_tiles, dev_points, kernel, coords_i, &rho_i, dc, i);

        dev_points->rho[i] = rho_i;
      }
    }
  };

  template <typename TAcc, uint8_t Ndim, uint8_t N_>
  ALPAKA_FN_HOST_ACC void for_recursion_nearest_higher(const TAcc& acc,
                                                       VecArray<uint32_t, Ndim>& base_vec,
                                                       const VecArray<VecArray<uint32_t, 2>, Ndim>& s_box,
                                                       TilesAlpaka<Ndim>* tiles,
                                                       PointsView<Ndim>* dev_points,
                                                       const VecArray<float, Ndim>& coords_i,
                                                       float rho_i,
                                                       float* delta_i,
                                                       int* nh_i,
                                                       float dm_sq,
                                                       uint32_t point_id) {
    if constexpr (N_ == 0) {
      int binId{tiles->getGlobalBinByBin(acc, base_vec)};
      // get the size of this bin
      int binSize{(*tiles)[binId].size()};

      // iterate inside this bin
      for (int binIter{}; binIter < binSize; ++binIter) {
        unsigned int j{(*tiles)[binId][binIter]};
        // query N'_{dm}(i)
        float rho_j{dev_points->rho[j]};
        bool found_higher{(rho_j > rho_i)};
        // in the rare case where rho is the same, use detid
        found_higher = found_higher || ((rho_j == rho_i) && (rho_j > 0.f) && (j > point_id));

        // Calculate the distance between the two points
        VecArray<float, Ndim> coords_j{dev_points->coords[j]};
        float dist_ij_sq{0.f};
        for (int dim{}; dim != Ndim; ++dim) {
          dist_ij_sq += (coords_j[dim] - coords_i[dim]) * (coords_j[dim] - coords_i[dim]);
        }

        if (found_higher && dist_ij_sq <= dm_sq) {
          // find the nearest point within N'_{dm}(i)
          if (dist_ij_sq < *delta_i) {
            // update delta_i and nearestHigher_i
            *delta_i = dist_ij_sq;
            *nh_i = j;
          }
        }
      }  // end of interate inside this bin

      return;
    } else {
      for (unsigned int i{s_box[s_box.capacity() - N_][0]}; i <= s_box[s_box.capacity() - N_][1]; ++i) {
        base_vec[base_vec.capacity() - N_] = i;
        for_recursion_nearest_higher<TAcc, Ndim, N_ - 1>(
            acc, base_vec, s_box, tiles, dev_points, coords_i, rho_i, delta_i, nh_i, dm_sq, point_id);
      }
    }
  }

  struct KernelCalculateNearestHigher {
    template <typename TAcc, uint8_t Ndim>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TilesAlpaka<Ndim>* dev_tiles,
                                  PointsView<Ndim>* dev_points,
                                  float dm,
                                  float dc,
                                  uint32_t n_points) const {
      float dm_squared{dm * dm};
      for (uint32_t i : uniform_elements(acc, n_points)) {
        float delta_i{std::numeric_limits<float>::max()};
        int nh_i{-1};
        VecArray<float, Ndim> coords_i{dev_points->coords[i]};
        float rho_i{dev_points->rho[i]};

        // Get the extremes of the search box
        VecArray<VecArray<float, 2>, Ndim> searchbox_extremes;
        for (int dim{}; dim != Ndim; ++dim) {
          VecArray<float, 2> dim_extremes;
          dim_extremes.push_back_unsafe(coords_i[dim] - dm);
          dim_extremes.push_back_unsafe(coords_i[dim] + dm);

          searchbox_extremes.push_back_unsafe(dim_extremes);
        }

        // Calculate the search box
        VecArray<VecArray<uint32_t, 2>, Ndim> search_box;
        dev_tiles->searchBox(acc, searchbox_extremes, &search_box);

        VecArray<uint32_t, Ndim> base_vec{};
        for_recursion_nearest_higher<TAcc, Ndim, Ndim>(
            acc, base_vec, search_box, dev_tiles, dev_points, coords_i, rho_i, &delta_i, &nh_i, dm_squared, i);

        dev_points->delta[i] = alpaka::math::sqrt(acc, delta_i);
        dev_points->nearest_higher[i] = nh_i;
      }
    }
  };

  template <uint8_t Ndim>
  struct KernelFindClusters {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  VecArray<int32_t, reserve>* seeds,
                                  VecArray<int32_t, max_followers>* followers,
                                  PointsView<Ndim>* dev_points,
                                  float dm,
                                  float d_c,
                                  float rho_c,
                                  uint32_t n_points) const {
      for (uint32_t i : uniform_elements(acc, n_points)) {
        // initialize cluster_index
        dev_points->cluster_index[i] = -1;

        float delta_i{dev_points->delta[i]};
        float rho_i{dev_points->rho[i]};

        // Determine whether the point is a seed or an outlier
        bool is_seed{(delta_i > d_c) && (rho_i >= rho_c)};
        bool is_outlier{(delta_i > dm) && (rho_i < rho_c)};

        if (is_seed) {
          dev_points->is_seed[i] = 1;
          seeds->push_back(acc, i);
        } else {
          if (!is_outlier) {
            followers[dev_points->nearest_higher[i]].push_back(acc, i);
          }
          dev_points->is_seed[i] = 0;
        }
      }
    }
  };

  template <uint8_t Ndim>
  struct KernelAssignClusters {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  VecArray<int32_t, reserve>* seeds,
                                  VecArray<int, max_followers>* followers,
                                  PointsView<Ndim>* dev_points) const {
      const auto& seeds_0{*seeds};
      const auto n_seeds{seeds_0.size()};
      for (uint32_t idx_cls : uniform_elements(acc, n_seeds)) {
        int local_stack[256] = {-1};
        int local_stack_size{};

        int idx_this_seed{seeds_0[idx_cls]};
        dev_points->cluster_index[idx_this_seed] = idx_cls;
        // push_back idThisSeed to localStack
        local_stack[local_stack_size] = idx_this_seed;
        ++local_stack_size;
        // process all elements in localStack
        while (local_stack_size > 0) {
          // get last element of localStack
          int idx_end_of_local_stack{local_stack[local_stack_size - 1]};
          int temp_cluster_index{dev_points->cluster_index[idx_end_of_local_stack]};
          // pop_back last element of localStack
          local_stack[local_stack_size - 1] = -1;
          --local_stack_size;
          const auto& followers_ies{followers[idx_end_of_local_stack]};
          const auto followers_size{followers[idx_end_of_local_stack].size()};
          // loop over followers of last element of localStack
          for (int j{}; j != followers_size; ++j) {
            // pass id to follower
            int follower{followers_ies[j]};
            dev_points->cluster_index[follower] = temp_cluster_index;
            // push_back follower to localStack
            local_stack[local_stack_size] = follower;
            ++local_stack_size;
          }
        }
      }
    }
  };

  template <typename TAcc, uint8_t Ndim>
  class CLUEsteringAlgo {
  public:
    CLUEsteringAlgo() = delete;
    explicit CLUEsteringAlgo(float dc, float rhoc, float dm, int pPBin, Queue queue_)
        : dc_{dc}, rhoc_{rhoc}, dm_{dm}, pointsPerTile_{pPBin} {
      init_device_buffers(queue_);
    }

    TilesAlpaka<Ndim>* m_tiles;
    VecArray<int32_t, reserve>* m_seeds;
    VecArray<int32_t, max_followers>* m_followers;

    template <typename TKernel>
    void make_clusters(Points<Ndim>& h_points,
                       PointsAlpaka<Ndim>& d_points,
                       const TKernel& kernel,
                       Queue queue_,
                       std::size_t block_size) {
      setup(h_points, d_points, queue_, block_size);

      const Idx grid_size = cms::alpakatools::divide_up_by(h_points.n, block_size);
      auto working_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
      alpaka::enqueue(
          queue_,
          alpaka::createTaskKernel<Acc1D>(working_div, KernelFillTiles{}, d_points.view(), m_tiles, h_points.n));

      alpaka::enqueue(
          queue_,
          alpaka::createTaskKernel<Acc1D>(
              working_div, KernelCalculateLocalDensity{}, m_tiles, d_points.view(), kernel, dc_, h_points.n));
      alpaka::enqueue(queue_,
                      alpaka::createTaskKernel<Acc1D>(
                          working_div, KernelCalculateNearestHigher{}, m_tiles, d_points.view(), dm_, dc_, h_points.n));
      alpaka::enqueue(queue_,
                      alpaka::createTaskKernel<Acc1D>(working_div,
                                                      KernelFindClusters<Ndim>{},
                                                      m_seeds,
                                                      m_followers,
                                                      d_points.view(),
                                                      dm_,
                                                      dc_,
                                                      rhoc_,
                                                      h_points.n));

      // We change the working division when assigning the clusters
      const Idx grid_size_seeds = cms::alpakatools::divide_up_by(reserve, block_size);
      auto working_div_seeds = cms::alpakatools::make_workdiv<Acc1D>(grid_size_seeds, block_size);

      alpaka::enqueue(queue_,
                      alpaka::createTaskKernel<Acc1D>(
                          working_div_seeds, KernelAssignClusters<Ndim>{}, m_seeds, m_followers, d_points.view()));

#ifdef DEBUG
      alpaka::memcpy(queue_,
                     cms::alpakatools::make_host_view(h_points.m_rho.data(), h_points.n),
                     d_points.rho,
                     static_cast<uint32_t>(h_points.n));
      alpaka::memcpy(queue_,
                     cms::alpakatools::make_host_view(h_points.m_delta.data(), h_points.n),
                     d_points.delta,
                     static_cast<uint32_t>(h_points.n));
      alpaka::memcpy(queue_,
                     cms::alpakatools::make_host_view(h_points.m_nearestHigher.data(), h_points.n),
                     d_points.nearest_higher,
                     static_cast<uint32_t>(h_points.n));
#endif

      alpaka::memcpy(queue_,
                     cms::alpakatools::make_host_view(h_points.m_clusterIndex.data(), h_points.n),
                     d_points.cluster_index,
                     static_cast<uint32_t>(h_points.n));
      alpaka::memcpy(queue_,
                     cms::alpakatools::make_host_view(h_points.m_isSeed.data(), h_points.n),
                     d_points.is_seed,
                     static_cast<uint32_t>(h_points.n));

      // Wait for all the operations in the queue to finish
      alpaka::wait(queue_);
    }

  private:
    float dc_;
    float rhoc_;
    float dm_;
    // average number of points found in a tile
    int pointsPerTile_;

    std::optional<cms::alpakatools::device_buffer<Device, TilesAlpaka<Ndim>>> d_tiles;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int32_t, reserve>>> d_seeds;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int32_t, max_followers>[]>>
        d_followers;

    void init_device_buffers(Queue queue_) {
      d_tiles = cms::alpakatools::make_device_buffer<TilesAlpaka<Ndim>>(queue_);
      d_seeds = cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int32_t, reserve>>(queue_);
      d_followers =
          cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int32_t, max_followers>[]>(queue_, reserve);

      // Copy to the public pointers
      m_tiles = (*d_tiles).data();
      m_seeds = (*d_seeds).data();
      m_followers = (*d_followers).data();
    }

    void setup(const Points<Ndim>& h_points, PointsAlpaka<Ndim>& d_points, Queue queue_, std::size_t block_size) {
      // calculate the number of tiles and their size
      const auto nTiles{std::ceil(h_points.n / static_cast<float>(pointsPerTile_))};
      const auto nPerDim{std::ceil(std::pow(nTiles, 1. / Ndim))};

      CoordinateExtremes<Ndim> min_max;
      float tile_size[Ndim];
      calculate_tile_size(min_max, tile_size, h_points, nPerDim);

      const auto device = alpaka::getDev(queue_);
      alpaka::memcpy(queue_,
                     cms::alpakatools::make_device_view(device, (*d_tiles)->minMax(), 2 * Ndim),
                     cms::alpakatools::make_host_view(min_max.data(), 2 * Ndim));
      alpaka::memcpy(queue_,
                     cms::alpakatools::make_device_view(device, (*d_tiles)->tileSize(), Ndim),
                     cms::alpakatools::make_host_view(tile_size, Ndim));
      alpaka::wait(queue_);

      const Idx tiles_grid_size = cms::alpakatools::divide_up_by(nTiles, block_size);
      const auto tiles_working_div = cms::alpakatools::make_workdiv<Acc1D>(tiles_grid_size, block_size);
      alpaka::enqueue(queue_,
                      alpaka::createTaskKernel<Acc1D>(tiles_working_div, KernelResetTiles{}, m_tiles, nTiles, nPerDim));

      alpaka::memcpy(queue_, d_points.coords, cms::alpakatools::make_host_view(h_points.m_coords.data(), h_points.n));
      alpaka::memcpy(queue_, d_points.weight, cms::alpakatools::make_host_view(h_points.m_weight.data(), h_points.n));
      alpaka::memset(queue_, (*d_seeds), 0x00);

      // Define the working division
      const Idx grid_size = cms::alpakatools::divide_up_by(h_points.n, block_size);
      const auto working_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
      alpaka::enqueue(queue_,
                      alpaka::createTaskKernel<Acc1D>(working_div, KernelResetFollowers{}, m_followers, h_points.n));
    }

    // Construction of the tiles
    void calculate_tile_size(CoordinateExtremes<Ndim>& min_max,
                             float* tile_sizes,
                             const Points<Ndim>& h_points,
                             uint32_t nPerDim) {
      for (size_t dim{}; dim != Ndim; ++dim) {
        const float dimMax{(*std::max_element(
            h_points.m_coords.begin(), h_points.m_coords.end(), [dim](const auto& vec1, const auto& vec2) -> bool {
              return vec1[dim] < vec2[dim];
            }))[dim]};
        const float dimMin{(*std::min_element(
            h_points.m_coords.begin(), h_points.m_coords.end(), [dim](const auto& vec1, const auto& vec2) -> bool {
              return vec1[dim] < vec2[dim];
            }))[dim]};

        min_max.min(dim) = dimMin;
        min_max.max(dim) = dimMax;

        const float tileSize{(dimMax - dimMin) / nPerDim};
        tile_sizes[dim] = tileSize;
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif /*CLUEsteringAlgo_H_ */
