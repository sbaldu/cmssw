
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoVertex/PixelVertexFinding/interface/DivisiveVertexFinder.h"

#include <memory>
#include <string>
#include <cmath>

class CLUEVertexProducer : public edm::stream::EDProducer<> {
public:
  explicit CLUEVertexProducer(const edm::ParameterSet&);
  ~CLUEVertexProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  // Turn on debug printing if verbose_ > 0
  const int verbose_;
  // Tracking cuts before sending tracks to vertex algo
  const double ptMin_;
  const bool method2;
  const edm::InputTag trackCollName;
  const edm::EDGetTokenT<reco::TrackCollection> token_Tracks;
  const edm::EDGetTokenT<reco::BeamSpot> token_BeamSpot;

  DivisiveVertexFinder* dvf_;
};

void CLUEVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
  //descriptions.addWithDefaultLabel(desc);
  descriptions.add("clueVertices", desc);
}

CLUEVertexProducer::CLUEVertexProducer(const edm::ParameterSet& conf)
    // 0 silent, 1 chatty, 2 loud
    : verbose_(conf.getParameter<int>("Verbosity")),
      // 1.0 GeV
      ptMin_(conf.getParameter<double>("PtMin")),
      method2(conf.getParameter<bool>("Method2")),
      trackCollName(conf.getParameter<edm::InputTag>("TrackCollection")),
      token_Tracks(consumes<reco::TrackCollection>(trackCollName)),
      token_BeamSpot(consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpot"))) {
  // Register my product
  produces<reco::VertexCollection>("clueVertices");

  // Setup shop
  std::string finder = conf.getParameter<std::string>("Finder");  // DivisiveVertexFinder
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
    if (track_pt_min != ptMin_) {
      if (track_pt_min < ptMin_)
        edm::LogInfo("PixelVertexProducer")
            << "minimum track pT setting differs between PixelVertexProducer (" << ptMin_ << ") and PVcomparer ("
            << track_pt_min << ") [PVcomparer considers tracks w/ lower threshold than PixelVertexProducer does] !!!";
      else
        edm::LogInfo("PixelVertexProducer") << "minimum track pT setting differs between PixelVertexProducer ("
                                            << ptMin_ << ") and PVcomparer (" << track_pt_min << ") !!!";
    }
    track_pt_max = PVcomparerPSet.getParameter<double>("track_pt_max");
    track_chi2_max = PVcomparerPSet.getParameter<double>("track_chi2_max");
    track_prob_min = PVcomparerPSet.getParameter<double>("track_prob_min");
  }

  if (finder == "DivisiveVertexFinder") {
    if (verbose_ > 0)
      edm::LogInfo("PixelVertexProducer") << ": Using the DivisiveVertexFinder\n";
    dvf_ = new DivisiveVertexFinder(track_pt_min,
                                    track_pt_max,
                                    track_chi2_max,
                                    track_prob_min,
                                    zOffset,
                                    ntrkMin,
                                    useError,
                                    zSeparation,
                                    wtAverage,
                                    verbose_);
  } else {  // Finder not supported, or you made a mistake in your request
    // throw an exception once I figure out how CMSSW does this
  }
}

CLUEVertexProducer::~CLUEVertexProducer() { delete dvf_; }

void CLUEVertexProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  // First fish the pixel tracks out of the event
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByToken(token_Tracks, trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());
  if (verbose_ > 0)
    edm::LogInfo("PixelVertexProducer") << ": Found " << tracks.size() << " tracks in TrackCollection called "
                                        << trackCollName << "\n";

  // Second, make a collection of pointers to the tracks we want for the vertex finder
  reco::TrackRefVector trks;
  for (unsigned int i = 0; i < tracks.size(); i++) {
    if (tracks[i].pt() > ptMin_)
      trks.push_back(reco::TrackRef(trackCollection, i));
  }
  if (verbose_ > 0)
    edm::LogInfo("PixelVertexProducer") << ": Selected " << trks.size() << " of these tracks for vertexing\n";

  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByToken(token_BeamSpot, bsHandle);
  math::XYZPoint myPoint(0., 0., 0.);
  if (bsHandle.isValid())
    //FIXME: fix last coordinate with vertex.z() at same time
    myPoint = math::XYZPoint(bsHandle->x0(), bsHandle->y0(), 0.);

  // Third, ship these tracks off to be vertexed
  auto vertexes = std::make_unique<reco::VertexCollection>();
  bool ok;

  AlgebraicSymMatrix33 we;
  we(0, 0) = 10000;
  we(1, 1) = 10000;
  we(2, 2) = 10000;

  reco::Vertex v1(reco::Vertex::Point(1., 2., 3.), we);
  reco::Vertex v2(reco::Vertex::Point(4., 5., 6.), we);
  vertexes->push_back(v1);
  vertexes->push_back(v2);

  e.put(std::move(vertexes));
}

DEFINE_FWK_MODULE(CLUEVertexProducer);
