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

#include "CLUEstering/CLUEstering.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace clueVertexFinder {

    class Producer {
    public:
      Producer(float dc, float rhoc, float dm, int pPBin, bool wtAvg)
          : dc_(dc), rhoc_(rhoc), dm_(dm), pPBin_(pPBin), wtAvg_(wtAvg) {}

      ~Producer() = default;

      reco::ZVertexSoACollection makeAsync(
          Queue& queue, ::reco::TrackSoAConstView const& tracks_view, int maxVertices, float ptMin, float ptMax);

    private:
      float dc_;
      float rhoc_;
      float dm_;
      int pPBin_;
      bool wtAvg_;
    };

  }  // namespace clueVertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
