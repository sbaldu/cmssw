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
          dc_(conf.getParameter<double>("dc")),
          rhoc_(conf.getParameter<double>("rhoc")),
          dm_(conf.getParameter<double>("dm")),
          seed_dc_(conf.getParameter<double>("seed_dc")),
          trackCollName(conf.getParameter<edm::InputTag>("TrackCollection")),
          token_Tracks(consumes(trackCollName)),
          //token_BeamSpot(consumes(conf.getParameter<edm::InputTag>("beamSpot"))),
          token_RecoVertex(produces()) {}

    ~CLUEVertexProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("Verbosity", 0);
      desc.add<int>("maxVertices", 1024);
      desc.add<double>("PtMin", 1.0);
      desc.add<double>("PtMax", 75.);
      desc.add<bool>("Method2", true);
      desc.add<double>("dc", 0.04);
      desc.add<double>("rhoc", 0.01);
      desc.add<double>("dm", 0.04);
      desc.add<double>("seed_dc", 0.04);
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
    }

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      auto const& tracks_d = event.get(token_Tracks);

      std::cout << "dc = " << dc_ << " rhoc = " << rhoc_ << " dm = " << dm_ << " seed_dc = " << seed_dc_ << std::endl;
      clueVertexFinder::Producer vertexProducer(dc_, rhoc_, dm_, seed_dc_, wtAvg_);
      event.emplace(
          token_RecoVertex,
          std::move(vertexProducer.makeAsync(event.queue(), tracks_d.view().tracks(), maxVertices_, ptMin_, ptMax_)));
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
    const float dc_;          // Side length of box to calculate density
    const float rhoc_;        // Minimum energy density to NOT be an outlier
    const float dm_;          // Side length of box to search for followers
    const float seed_dc_;     // Separation for seed promotion
    const bool wtAvg_{true};  // Decides how to copute error
                              // Input and output collections
    const edm::InputTag trackCollName;
    device::EDGetToken<TkSoADevice> token_Tracks;
    device::EDPutToken<reco::ZVertexSoACollection> token_RecoVertex;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CLUEVertexProducer);
