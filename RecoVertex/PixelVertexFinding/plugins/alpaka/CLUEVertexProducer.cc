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
          ptMin_(conf.getParameter<double>("PtMin")),
          method2(conf.getParameter<bool>("Method2")),
          trackCollName(conf.getParameter<edm::InputTag>("TrackCollection")),
          token_Tracks(consumes(trackCollName)),
          //token_BeamSpot(consumes(conf.getParameter<edm::InputTag>("beamSpot"))),
          token_RecoVertex(produces("CLUEVertex")) {
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
      std::cout << track_pt_min << " " << track_pt_max << " " << track_chi2_max << " " << track_prob_min << "\n";
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
      desc.add<double>("PtMin", 1.0);
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
      /* It's layout is:
             SOA_COLUMN(Quality, quality),
             SOA_COLUMN(float, chi2),
             SOA_COLUMN(int8_t, nLayers),
             SOA_COLUMN(float, eta),
             SOA_COLUMN(float, pt),
             // state at the beam spot: {phi, tip, 1/pt, cotan(theta), zip}
             SOA_EIGEN_COLUMN(Vector5f, state),
             SOA_EIGEN_COLUMN(Vector15f, covariance),
             SOA_SCALAR(int, nTracks),
             SOA_SCALAR(HitContainer, hitIndices),
             SOA_SCALAR(HitContainer, detIndices)
      */
      // const auto& bsHandle = event.get(token_BeamSpot);

      auto const& tracks_d_view = tracks_d.view();
      auto queue = event.queue();
      int maxVertices = 10;
      const auto maxTracks = tracks_d_view.metadata().size();
      std::cout << "maxTracks = " << maxTracks << std::endl;
      //const uint32_t nTracks; // SEG FAULT HERE; cannot access nTracks since
      // it's data allocated in device
      // will need to use this in the kernel
      /*TkSoAHost host_tracks; 
      alpaka::memcpy(queue, host_tracks, tracks_d);*/

      auto tracks_h = cms::alpakatools::CopyToHost<TkSoADevice>::copyAsync(queue, tracks_d);
      alpaka::memcpy(queue, tracks_h.buffer(), tracks_d.buffer());
      alpaka::wait(queue);
      const uint32_t nTracks = tracks_h.view().nTracks();

      std::cout << "nTracks = " << nTracks << std::endl;
      //const uint32_t nTracks = maxTracks;  // placeholder

      ZVertexSoACollection vertices({{maxVertices, maxTracks}}, queue);  // this object is in the device
      auto data = vertices.view();
      auto trkdata = vertices.view<::reco::ZVertexTracksSoA>();  // access the data in the ZVertexTracksSoA Layout
      auto vrtxdata = vertices.view<::reco::ZVertexSoA>();       // access the data in the ZVertexSoA Layout

      std::vector<float> coords;
      std::vector<float> pts;

      // TO DO: fill the coords vector appropriately
      for (auto idx = 0u; idx < nTracks; ++idx) {
        auto pt = (tracks_h.view()
                       .pt())[idx];  // instead of [idx] I was doing [idx + nTracks], but it never went into seg fault
        if (pt >= ptMin_) {
          coords.push_back(::reco::zip(tracks_h.view(), idx));
          pts.push_back(pt);
          /*coords.at(it) = reco::zip(tracks_h.view(), idx);
          coords.at(it + nTracks) = pt;*/
          if (idx < 10) {
            std::cout << "coords[" << idx << "] = " << coords[idx] << std::endl;
            std::cout << "pts[ " << idx << "] = " << pts[idx] << std::endl;
          }
        }
      }

      size_t trueTracks = coords.size();
      std::vector<int> results(2 * trueTracks);
      if (trueTracks != 0) {
        coords.insert(coords.end(), pts.begin(), pts.end());

        std::cout << "coords.size() and results.size() = " << coords.size() << " " << results.size() << std::endl;
        clueVertexFinder::Producer clusterer(m_dc, m_rhoc, m_dm, m_pPBin, m_wtAvg);
        // ************TRYING TO LOAD DATA INTO WORKSPACE************
		clusterer.makeClusters(queue, coords, results, trueTracks);
        std::cout << "Before calling makeAsync\n";
        // clusterer.makeAsync(queue, tracks_d_view, maxVertices, ptMin_);
        std::cout << "After calling makeAsync\n";
      }
	  event.put(std::move(vertices));
    }

  private:
    // ----------member data ---------------------------
    // Turn on debug printing if verbose_ > 0
    const int verbose_;
    // Tracking cuts before sending tracks to vertex algo
    const double ptMin_;
    const bool method2;
    const edm::InputTag trackCollName;
    device::EDGetToken<TkSoADevice> token_Tracks;
    // const device::EDGetToken<reco::BeamSpot> token_BeamSpot;
    device::EDPutToken<ZVertexSoACollection> token_RecoVertex;
    // Parameters for CLUEAlgoAlpaka
    float m_dc{0.2f};    // Side length of box to calculate density
    float m_rhoc{10.f};  // Minimum energy density to NOT be an outlier
    float m_dm{0.75f};   // Side length of box to search for followers
    int m_pPBin{128};    // Average number of points found in a tile
    bool m_wtAvg{true};  // Decides how to copute error
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CLUEVertexProducer);
