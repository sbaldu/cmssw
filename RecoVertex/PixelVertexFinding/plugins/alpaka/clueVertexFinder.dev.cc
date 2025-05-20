#include "./clueVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace clueVertexFinder {

    // 
    void Producer::makeClusters(std::vector<float>& coords, std::vector<int>& results, Queue& queue) {
      int nTracks = results.size();
      clue::PointsHost<1> h_points(queue, nTracks, coords, results);
      clue::PointsDevice<1, Device> d_points(queue, nTracks);

      clue::Clusterer<1> algo(queue, m_dc, m_rhoc, m_dm);
      const std::size_t block_size{256};
      algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
    }

    // Kernel to compute parameters of the verteces and the tracks
    template <typename TAcc>
    ALPAKA_FN_ACC void ComputeParams<TAcc>::operator()(TAcc const& acc,
                                                 int* myClusters,
                                                 int* isSeed,
                                                 float* coords,
                                                 reco::ZVertexSoAView vrtxdata,
                                                 reco::ZVertexTracksSoAView trkdata,
                                                 int nTracks,
                                                 int nClusters) const {
      int idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (isSeed[idx]) {
        vrtxdata[myClusters[idx]].zv() = coords[idx];
      }

      trkdata[idx].idv() = myClusters[idx];
    }
  }  // namespace clueVertexFinder
}  //namespace ALPAKA_ACCELERATOR_NAMESPACE
