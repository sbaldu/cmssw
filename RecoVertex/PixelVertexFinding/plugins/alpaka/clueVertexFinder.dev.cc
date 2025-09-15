#include <alpaka/alpaka.hpp>
#include <cstdio>

#include "RecoVertex/PixelVertexFinding/plugins/alpaka/vertexFinder.h"
#include "RecoVertex/PixelVertexFinding/plugins/alpaka/clueVertexFinder.h"
#include "RecoVertex/PixelVertexFinding/plugins/alpaka/fitVertices.h"
#include "RecoVertex/PixelVertexFinding/plugins/alpaka/splitVertices.h"
#include "RecoVertex/PixelVertexFinding/plugins/alpaka/sortByPt2.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace clueVertexFinder {
    constexpr float maxChi2ForFirstFit = 50.f;
    constexpr float maxChi2ForFinalFit = 5000.f;
    constexpr float maxChi2ForSplit = 9.f;

    class LoadTracks {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::reco::TrackSoAConstView tracks_view,
                                    ::reco::ZVertexSoAView data,
                                    ::reco::ZVertexTracksSoAView trkdata,
                                    ::vertexFinder::PixelVertexWorkSpaceSoAView ws,
                                    float ptMin,
                                    float ptMax) const {
        auto const* quality = tracks_view.quality();

        for (auto idx : cms::alpakatools::uniform_elements(acc, tracks_view.nTracks())) {
          [[maybe_unused]] auto nHits = ::reco::nHits(tracks_view, idx);
          ALPAKA_ASSERT_ACC(nHits >= 3);

          // initialize the track data
          trkdata[idx].idv() = -1;

          // do not use triplets
          if (::reco::isTriplet(tracks_view, idx))
            continue;

          // use only "high purity" track
          if (quality[idx] < ::pixelTrack::Quality::highPurity)
            continue;

          auto pt = tracks_view[idx].pt();
          // pT min cut
          if (pt < ptMin)
            continue;

          // clamp pT to the pTmax
          pt = std::min<float>(pt, ptMax);

          // load the track data into the workspace
          auto it = alpaka::atomicAdd(acc, &ws.ntrks(), 1u, alpaka::hierarchy::Blocks{});
          ws[it].itrk() = idx;
          ws[it].zt() = ::reco::zip(tracks_view, idx);
          ws[it].ezt2() = tracks_view[idx].covariance()(14);
          ws[it].ptt2() = pt * pt;
        }
      }
    };

    ZVertexSoACollection Producer::makeAsync(
        Queue& queue, ::reco::TrackSoAConstView const& tracks_view, int maxVertices, float ptMin, float ptMax) {
      const auto maxTracks = tracks_view.metadata().size();
      std::cout << "max tracks = " << maxTracks << std::endl;
      ZVertexSoACollection vertices({{maxVertices, maxTracks}}, queue);
      auto verticesView = vertices.view();
      auto vertexTrackDataView = vertices.view<::reco::ZVertexTracksSoA>();

      // Initialize the workspace
      vertexFinder::PixelVertexWorkSpaceSoADevice workspace(maxTracks, queue);
      auto workspaceView = workspace.view();
      const auto initWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(
          queue, initWorkDiv, ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::Init{}, verticesView, workspaceView);

      //Load Tracks
      const uint32_t blockSize = 128;
      const uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(maxTracks + blockSize - 1, blockSize);
      const auto loadTracksWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          loadTracksWorkDiv,
                          LoadTracks{},
                          tracks_view,
                          verticesView,
                          vertexTrackDataView,
                          workspaceView,
                          ptMin,
                          ptMax);

      // Copy number of tracks to host
      auto nTracksBuf = cms::alpakatools::make_host_buffer<uint32_t>(queue);
      alpaka::memcpy(queue, nTracksBuf, cms::alpakatools::make_device_view<uint32_t>(queue, workspaceView.ntrks()));
      alpaka::wait(queue);
      const auto nTracks = *nTracksBuf;

      // Run CLUEstering
      if (nTracks > 0) {
        auto isSeed =
            cms::alpakatools::make_device_buffer<int[]>(queue, nTracks);  // temporary buffer needed by CLUEstering
        clue::Clusterer<1> clusterer(queue, dc_, rhoc_, dm_, pPBin_);
        clue::PointsDevice<1, Device> d_points(
            queue, nTracks, workspaceView.zt(), workspaceView.ptt2(), workspaceView.iv(), isSeed.data());
        clusterer.make_clusters(queue, d_points);
        clue::PointsHost<1> h_points(queue, nTracks);
        clue::copyToHost(queue, h_points, d_points);
        alpaka::wait(queue);
        uint32_t nVertices = std::accumulate(h_points.isSeed().data(), h_points.isSeed().data() + h_points.size(), 0u);
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view<uint32_t>(queue, verticesView.nvFinal()),
                       cms::alpakatools::make_host_view<uint32_t>(nVertices));
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view<uint32_t>(queue, workspaceView.nvIntermediate()),
                       cms::alpakatools::make_host_view<uint32_t>(nVertices));
      }
      const auto finderSorterWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024 - 128);
      alpaka::exec<Acc1D>(queue,
                          finderSorterWorkDiv,
                          ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::FitVerticesKernel{},
                          verticesView,
                          vertexTrackDataView,
                          workspaceView,
                          maxChi2ForFirstFit);
      const auto splitterFitterWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1024, 128);
      alpaka::exec<Acc1D>(queue,
                          splitterFitterWorkDiv,
                          ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::SplitVerticesKernel{},
                          verticesView,
                          vertexTrackDataView,
                          workspaceView,
                          maxChi2ForSplit);
      alpaka::exec<Acc1D>(queue,
                          finderSorterWorkDiv,
                          ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::FitVerticesKernel{},
                          verticesView,
                          vertexTrackDataView,
                          workspaceView,
                          maxChi2ForFinalFit);
      alpaka::exec<Acc1D>(queue,
                          finderSorterWorkDiv,
                          ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::SortByPt2Kernel{},
                          verticesView,
                          vertexTrackDataView,
                          workspaceView);

      return vertices;
    }

  }  // namespace clueVertexFinder
}  //namespace ALPAKA_ACCELERATOR_NAMESPACE
