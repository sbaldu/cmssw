//#pragma once
#ifndef RecoVertex_PixleVertexFinding_plugins_alpaka_clueVertexFinder_h
#define RecoVertex_PixleVertexFinding_plugins_alpaka_clueVertexFinder_h

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <vector>
#include <numeric>

#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/Error.h"

#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoVertex/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"
#include "RecoVertex/PixelVertexFinding/plugins/alpaka/PixelVertexWorkSpaceSoADeviceAlpaka.h"

#include "./CLUE/include/CLUEstering/CLUEstering.hpp"

/*test*/

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace clueVertexFinder {
    class Producer {
    public:
      Producer(float dc, float rhoc, float dm, int pPBin, bool wtAvg)
          : m_dc(dc), m_rhoc(rhoc), m_dm(dm), m_pPBin(pPBin), m_wtAvg(wtAvg) {}

      ~Producer() = default;

      void makeClusters(Queue& queue, std::vector<float>& coords, std::vector<int>& results, size_t& nTracks);
      void makeClusters(Queue& queue, ::vertexFinder::PixelVertexWorkSpaceSoAView ws);
      /*ZVertexSoACollection*/ void makeAsync(Queue& queue,
                                              TracksSoACollection<pixelTopology::Phase2>::ConstView const& tracks_view,
                                              int maxVertices,
                                              float ptMin);

    private:
      float m_dc;
      float m_rhoc;
      float m_dm;
      int m_pPBin;
      bool m_wtAvg;
    };
    template <typename TAcc>
    struct ComputeParams {
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    int* myClusters,
                                    int* isSeed,
                                    float* coords,
                                    int* clusterCounter,
                                    reco::ZVertexSoAView vrtxdata,
                                    reco::ZVertexTracksSoAView trkdata,
                                    int nTracks,
                                    int nClusters) const;
    };
  }  // namespace clueVertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACe
#endif
