#include "./clueVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace clueVertexFinder {

    //
    void Producer::makeClusters(std::vector<float>& coords, std::vector<int>& results, Queue& queue) {
      std::cout << "clueVertexFinder line: " << __LINE__ << std::endl;
      int nTracks = results.size();
      std::cout << "clueVertexFinder line: " << __LINE__ << std::endl;
      clue::PointsHost<1> h_points(queue, nTracks, coords, results); // zv pt clidx isSeed, need to use another overload, not the one
                                                                     // I used here:
                                                                     // I need to use the one that takes:
                                                                     // pointer to z coords (input)
                                                                     // pointer to pt (weight) (input)
                                                                     // pointer to cluster indexes (output)  dv of the layout
                                                                     // pointer to isSeed (output)
      std::cout << "clueVertexFinder line: " << __LINE__ << std::endl;
      clue::PointsDevice<1, Device> d_points(queue, nTracks);
      std::cout << "clueVertexFinder line: " << __LINE__ << std::endl;

      clue::Clusterer<1> algo(queue, m_dc, m_rhoc, m_dm);
      std::cout << "clueVertexFinder line: " << __LINE__ << std::endl;
      const std::size_t block_size{256};
      algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
      std::cout << "clueVertexFinder line: " << __LINE__ << std::endl;
    }

    // Kernel to compute parameters of the verteces and the tracks
    template <typename TAcc>
    ALPAKA_FN_ACC void ComputeParams<TAcc>::operator()(TAcc const& acc,
                                                       int* myClusters,
                                                       int* isSeed,
                                                       float* coords,
                                                       int* clusterCounter,
                                                       reco::ZVertexSoAView vrtxdata,
                                                       reco::ZVertexTracksSoAView trkdata,
                                                       int nTracks,
                                                       int nClusters) const {
      int gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      int dimThread = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0];
      int firstElemIdx = gridThreadIdx * dimThread;
      if (firstElemIdx < nTracks) {
        int lastElemIdx = (nTracks > firstElemIdx + dimThread ? firstElemIdx + dimThread : nTracks);
        for (int idx = firstElemIdx; idx < lastElemIdx; ++idx) {
          if (isSeed[idx]) {
            vrtxdata[myClusters[idx]].zv() = coords[idx];
          }
          clusterCounter[myClusters[idx]]++;
          trkdata[idx].idv() = myClusters[idx];
        }
      }
    }
  }  // namespace clueVertexFinder
}  //namespace ALPAKA_ACCELERATOR_NAMESPACE
