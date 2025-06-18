#include <alpaka/alpaka.hpp>
#include <cstdio>

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoVertex/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"
#include "RecoVertex/PixelVertexFinding/plugins/alpaka/PixelVertexWorkSpaceSoADeviceAlpaka.h"

#include "./clueVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace clueVertexFinder {

    //
    void Producer::makeClusters(std::vector<float>& coords, std::vector<int>& results, Queue& queue, size_t& nTracks) {
      std::cout << "clueVertexFinder line: " << __LINE__ << std::endl;
      clue::PointsHost<1> h_points(
          queue, nTracks, coords, results);  // zv pt clidx isSeed, need to use another overload, not the one
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

    class LoadTracks {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    TracksSoACollection<pixelTopology::Phase2>::ConstView tracks_view,
                                    ::vertexFinder::PixelVertexWorkSpaceSoAView ws,
                                    float ptMin) const {
        //printf("clueVertexFinder.dev.cc: Before the for loop in the LoadTracks kernel \n");
        for (auto idx : cms::alpakatools::uniform_elements(acc, tracks_view.nTracks())) {
          auto pt = tracks_view[idx].pt();
          if (pt < ptMin)
            continue;
          auto it = alpaka::atomicAdd(acc, &ws.ntrks(), 1u, alpaka::hierarchy::Blocks{});
          ws[it].itrk() = idx;
          ws[it].zt() = reco::zip(tracks_view, idx);
          ws[it].ptt2() = pt * pt;
        }
        //printf("clueVertexFinder.dev.cc: After the for loop in the LoadTracks kernel \n");
      }
    };  // LoadTracks

    // void for now, since I'm not returning anything yet
    /*ZVertexSoACollection*/ void Producer::makeAsync(
        Queue& queue,
        TracksSoACollection<pixelTopology::Phase2>::ConstView const& tracks_view,
        int maxVertices,
        float ptMin) const {
      const auto maxTracks = tracks_view.metadata().size();
      vertexFinder::PixelVertexWorkSpaceSoADevice workspace(maxTracks, queue);
      std::cout << "clueVertexFinder.dev.cc: Created Workspace \n";
      auto ws = workspace.view();

      //TO DO: Initialize?

      //Load Tracks
      const uint32_t blockSize = 128;
      const uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(maxTracks + blockSize - 1, blockSize);
      const auto loadTracksWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue, loadTracksWorkDiv, LoadTracks{}, tracks_view, ws, ptMin);
      std::cout << "clueVertexFinder.dev.cc: Loaded the tracks into the workspace \n";
    }

    // Kernel to compute parameters of the verteces and the tracks
    /*template <typename TAcc>
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
    }*/
  }  // namespace clueVertexFinder
}  //namespace ALPAKA_ACCELERATOR_NAMESPACE
