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

#include "./clueVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CLUEVertexProducer : public global::EDProducer<> {
    using TkSoADevice = reco::TracksSoACollection;

  public:
    CLUEVertexProducer(edm::ParameterSet const& conf)
        : EDProducer(conf),
          verbose_(conf.getParameter<int>("Verbosity")),
          // 1.0 GeV
          maxVertices_(conf.getParameter<int>("maxVertices")),
          ptMin_(conf.getParameter<double>("PtMin")),
          ptMax_(conf.getParameter<double>("PtMax")),
          method2(conf.getParameter<bool>("Method2")),
          trackCollName(conf.getParameter<edm::InputTag>("TrackCollection")),
          token_Tracks(consumes(trackCollName)),
          //token_BeamSpot(consumes(conf.getParameter<edm::InputTag>("beamSpot"))),
          token_RecoVertex(produces()) {
      // Register my product

      // Setup shop
      // std::string finder = conf.getParameter<std::string>("Finder");  // DivisiveVertexFinder
      bool useError = conf.getParameter<bool>("UseError");            // true
      bool wtAverage = conf.getParameter<bool>("WtAverage");          // true
      double zOffset = conf.getParameter<double>("ZOffset");          // 5.0 sigma
      double zSeparation = conf.getParameter<double>("ZSeparation");  // 0.05 cm
      int ntrkMin = conf.getParameter<int>("NTrkMin");                // 3
      // Tracking requirements before sending a track to be considered for vtx

      double track_pt_min = ptMin_;
      double track_pt_max = 10.;
      double track_chi2_max = 9999999.;
      double track_prob_min = -1.;
      if (conf.exists("PVcomparer")) {
        edm::ParameterSet PVcomparerPSet = conf.getParameter<edm::ParameterSet>("PVcomparer");
        track_pt_min = PVcomparerPSet.getParameter<double>("track_pt_min");
        track_pt_max = PVcomparerPSet.getParameter<double>("track_pt_max");
        track_chi2_max = PVcomparerPSet.getParameter<double>("track_chi2_max");
        track_prob_min = PVcomparerPSet.getParameter<double>("track_prob_min");
      }
    }

    ~CLUEVertexProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("Verbosity", 0);
      desc.add<int>("maxVertices", 256);
      desc.add<double>("PtMin", 0.5);
      desc.add<double>("PtMax", 75.);
      desc.add<bool>("Method2", true);
      desc.add<edm::InputTag>("TrackCollection", edm::InputTag("pixelTracks"));
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
      desc.add<std::string>("Finder", "DivisiveVertexFinder");
      desc.add<bool>("UseError", true);
      desc.add<bool>("WtAverage", true);
      desc.add<double>("ZOffset", 5.0);
      desc.add<double>("ZSeparation", 0.05);
      desc.add<int>("NTrkMin", 2);
      {
        edm::ParameterSetDescription PVComparerPSet;
        PVComparerPSet.add<double>("track_pt_min", 1.0);
        PVComparerPSet.add<double>("track_pt_max", 10.0);
        PVComparerPSet.add<double>("track_chi2_max", 999999.);
        PVComparerPSet.add<double>("track_prob_min", -1.);
        desc.addOptional<edm::ParameterSetDescription>("PVcomparer", PVComparerPSet);
      }

      // check label

      descriptions.addWithDefaultLabel(desc);

      //descriptions.add("CLUEVertex", desc);
    }

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      auto const& tracks_d = event.get(token_Tracks);

      clueVertexFinder::Producer vertexProducer(dc_, rhoc_, dm_, pPBin_, wtAvg_);
      event.emplace(token_RecoVertex,
                    std::move(vertexProducer.makeAsync(event.queue(), tracks_d.view(), maxVertices_, ptMin_, ptMax_)));
    }

  private:
    // ----------member data ---------------------------
    // Turn on debug printing if verbose_ > 0
    const int verbose_;
    // Max number of vertices to be reconstructed
    const int maxVertices_;
    // Tracking cuts before sending tracks to vertex algo
    const float ptMin_;
    const float ptMax_;
    const bool method2;
    // Parameters for CLUEstering
    const float dc_{0.2f};    // Side length of box to calculate density
    const float rhoc_{10.f};  // Minimum energy density to NOT be an outlier
    const float dm_{0.75f};   // Side length of box to search for followers
    const int pPBin_{128};    // Average number of points found in a tile
    const bool wtAvg_{true};  // Decides how to copute error
                              // Input and output collections
    const edm::InputTag trackCollName;
    device::EDGetToken<TkSoADevice> token_Tracks;
    device::EDPutToken<ZVertexSoACollection> token_RecoVertex;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CLUEVertexProducer);
