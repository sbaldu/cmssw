// system include files

#include "TTree.h"
#include "TFile.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// user include files
#include "FWCore/Framework/interface/ESHandle.h"

// math
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

// reco track
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// reco vertex

// simulated track
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

// pile-up
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

// vertexing
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"

// simulated vertex
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// reco track and vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

// associator
#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"

#include <numeric>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

class VertexValidator : public edm::one::EDAnalyzer<edm::one::SharedResources> {
private:
  using LorentzVector = math::XYZTLorentzVector;

  enum SignalVertexKind { HIGHEST_PT = 0, IS_ASSOC2FIRST_RECO = 1, IS_ASSOC2ANY_RECO = 2 };

  struct simPrimaryVertex {
    simPrimaryVertex(double x1, double y1, double z1)
        : x(x1),
          y(y1),
          z(z1),
          ptsq(0),
          closest_vertex_distance_z(-1.),
          nGenTrk(0),
          num_matched_reco_tracks(0),
          average_match_quality(0.0) {
      ptot.setPx(0);
      ptot.setPy(0);
      ptot.setPz(0);
      ptot.setE(0);
      p4 = LorentzVector(0, 0, 0, 0);
      r = sqrt(x * x + y * y);
    };
    double x, y, z, r;
    HepMC::FourVector ptot;
    LorentzVector p4;
    double ptsq;
    double closest_vertex_distance_z;
    int nGenTrk;
    int num_matched_reco_tracks;
    float average_match_quality;
    EncodedEventId eventId;
    TrackingVertexRef sim_vertex;
    std::vector<const reco::Vertex*> rec_vertices;
  };

  // auxiliary class holding reconstructed vertices
  struct recoPrimaryVertex {
    enum VertexProperties { NONE = 0, MATCHED = 1, DUPLICATE = 2, MERGED = 4 };
    recoPrimaryVertex(double x1, double y1, double z1)
        : x(x1),
          y(y1),
          z(z1),
          pt(0),
          ptsq(0),
          closest_vertex_distance_z(-1.),
          purity(-1.),
          nRecoTrk(0),
          num_matched_sim_tracks(0),
          kind_of_vertex(0),
          recVtx(nullptr) {
      r = sqrt(x * x + y * y);
    };
    double x, y, z, r;
    double pt;
    double ptsq;
    double closest_vertex_distance_z;
    double purity;  // calculated and assigned in calculatePurityAndFillHistograms
    int nRecoTrk;
    int num_matched_sim_tracks;
    int kind_of_vertex;
    std::vector<const TrackingVertex*> sim_vertices;
    std::vector<const simPrimaryVertex*> sim_vertices_internal;
    std::vector<float> sim_vertices_num_shared_tracks;
    const reco::Vertex* recVtx;
    reco::VertexBaseRef recVtxRef;
  };

public:
  explicit VertexValidator(const edm::ParameterSet&);
  ~VertexValidator() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::vector<VertexValidator::simPrimaryVertex> getSimPVs(const edm::Handle<TrackingVertexCollection>&);

  std::vector<VertexValidator::recoPrimaryVertex> getRecoPVs(const edm::Handle<edm::View<reco::Vertex>>&);

  void matchSim2RecoVertices(std::vector<simPrimaryVertex>& simpv, const reco::VertexSimToRecoCollection& vertex_s2r);
  void matchReco2SimVertices(std::vector<recoPrimaryVertex>&,
                             const reco::VertexRecoToSimCollection&,
                             const std::vector<simPrimaryVertex>&);

  void calculatePurityAndFillHistograms(std::vector<recoPrimaryVertex>& recopvs,
                                        int genpv_position_in_reco_collection,
                                        bool signal_is_highest_pt);
  bool matchRecoTrack2SimSignal(const reco::TrackBaseRef&);

  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // Counters
  int num_total_gen_vertices_assoc2reco_;
  int num_total_reco_vertices_assoc2gen_;
  int num_total_gen_vertices_multiassoc2reco_;
  int num_total_reco_vertices_multiassoc2gen_;
  int num_total_reco_vertices_duplicate_;
  int genpv_position_in_reco_collection_;

  TTree* output_tree_;

  bool use_only_charged_tracks_;
  const bool use_reconstructable_simvertices_;
  const bool do_generic_sim_plots_;

  float maxEta_;

  const reco::RecoToSimCollection* r2s_;
  const reco::SimToRecoCollection* s2r_;

  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> vecPileupSummaryInfoToken_;
  edm::EDGetTokenT<edm::View<reco::Vertex>> reco_vertex_collection_token_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexCollectionToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoAssociationToken_;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;
  edm::EDGetTokenT<reco::VertexToTrackingVertexAssociator> vertexAssociatorToken_;

  const int reco_tracks_for_reconstructable_simvertices_;
};

VertexValidator::VertexValidator(const edm::ParameterSet& iConfig)
    : use_only_charged_tracks_(iConfig.getParameter<bool>("use_only_charged_tracks")),
      use_reconstructable_simvertices_(iConfig.getParameter<bool>("use_reconstructable_simvertices")),
      do_generic_sim_plots_(iConfig.getParameter<bool>("do_generic_sim_plots")),
      vecPileupSummaryInfoToken_(consumes<std::vector<PileupSummaryInfo>>(edm::InputTag(std::string("addPileupInfo")))),
      reco_vertex_collection_token_(
          consumes<edm::View<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertexCollection"))),
      trackingParticleCollectionToken_(
          consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticleCollection"))),
      trackingVertexCollectionToken_(
          consumes<TrackingVertexCollection>(iConfig.getParameter<edm::InputTag>("trackingVertexCollection"))),
      simToRecoAssociationToken_(
          consumes<reco::SimToRecoCollection>(iConfig.getParameter<edm::InputTag>("trackAssociatorMap"))),
      recoToSimAssociationToken_(
          consumes<reco::RecoToSimCollection>(iConfig.getParameter<edm::InputTag>("trackAssociatorMap"))),
      vertexAssociatorToken_(
          consumes<reco::VertexToTrackingVertexAssociator>(iConfig.getParameter<edm::InputTag>("vertexAssociator"))),
      reco_tracks_for_reconstructable_simvertices_(
          iConfig.getParameter<int>("reco_tracks_for_reconstructable_simvertices")) {}

void VertexValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("do_generic_sim_plots", false);
  desc.add<int>("reco_tracks_for_reconstructable_simvertices", 2);
  desc.add<edm::InputTag>("trackAssociatorMap", edm::InputTag("tpToHLTpixelTrackAssociation"));
  desc.add<edm::InputTag>("trackingParticleCollection", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("trackingVertexCollection", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("vertexCollection", edm::InputTag("hltPhase2PixelVertices"));
  desc.add<bool>("use_only_charged_tracks", true);
  desc.add<bool>("use_reconstructable_simvertices", true);
  desc.add<edm::InputTag>("vertexAssociator", edm::InputTag("vertexAssociatorByPositionAndTracks4pixelTracks"));

  descriptions.addWithDefaultLabel(desc);
}

std::vector<VertexValidator::simPrimaryVertex> VertexValidator::getSimPVs(
    const edm::Handle<TrackingVertexCollection>& tVC) {
  std::vector<VertexValidator::simPrimaryVertex> simpv;
  int current_event = -1;

  for (TrackingVertexCollection::const_iterator v = tVC->begin(); v != tVC->end(); ++v) {
    // I'd rather change this and select only vertices that come from
    // BX=0.  We should keep only the first vertex from all the events
    // at BX=0.
    if (v->eventId().bunchCrossing() != 0)
      continue;
    if (v->eventId().event() != current_event) {
      current_event = v->eventId().event();
    } else {
      continue;
    }
    // TODO(rovere) is this really necessary?
    if (fabs(v->position().z()) > 1000)
      continue;  // skip funny junk vertices

    // could be a new vertex, check  all primaries found so far to avoid
    // multiple entries
    simPrimaryVertex sv(v->position().x(), v->position().y(), v->position().z());
    sv.eventId = v->eventId();
    sv.sim_vertex = TrackingVertexRef(tVC, std::distance(tVC->begin(), v));

    for (TrackingParticleRefVector::iterator iTrack = v->daughterTracks_begin(); iTrack != v->daughterTracks_end();
         ++iTrack) {
      // TODO(rovere) isn't it always the case? Is it really worth
      // checking this out?
      // sv.eventId = (**iTrack).eventId();
      assert((**iTrack).eventId().bunchCrossing() == 0);
    }
    // TODO(rovere) maybe get rid of this old logic completely ... ?
    simPrimaryVertex* vp = nullptr;  // will become non-NULL if a vertex
                                     // is found and then point to it
    for (std::vector<simPrimaryVertex>::iterator v0 = simpv.begin(); v0 != simpv.end(); v0++) {
      if ((sv.eventId == v0->eventId) && (fabs(sv.x - v0->x) < 1e-5) && (fabs(sv.y - v0->y) < 1e-5) &&
          (fabs(sv.z - v0->z) < 1e-5)) {
        vp = &(*v0);
        break;
      }
    }
    if (!vp) {
      // this is a new vertex, add it to the list of sim-vertices
      simpv.push_back(sv);
      vp = &simpv.back();
    }

    // Loop over daughter track(s) as Tracking Particles
    for (TrackingVertex::tp_iterator iTP = v->daughterTracks_begin(); iTP != v->daughterTracks_end(); ++iTP) {
      auto momentum = (*(*iTP)).momentum();
      const reco::Track* matched_best_reco_track = nullptr;
      double match_quality = -1;
      if (use_only_charged_tracks_ && (**iTP).charge() == 0)
        continue;
      if (s2r_->find(*iTP) != s2r_->end()) {
        matched_best_reco_track = (*s2r_)[*iTP][0].first.get();
        match_quality = (*s2r_)[*iTP][0].second;
      }
      vp->ptot.setPx(vp->ptot.x() + momentum.x());
      vp->ptot.setPy(vp->ptot.y() + momentum.y());
      vp->ptot.setPz(vp->ptot.z() + momentum.z());
      vp->ptot.setE(vp->ptot.e() + (**iTP).energy());
      vp->ptsq += ((**iTP).pt() * (**iTP).pt());
      if (matched_best_reco_track) {
        vp->num_matched_reco_tracks++;
        vp->average_match_quality += match_quality;
      }
      // TODO(rovere) get rid of cuts on sim-tracks
      // TODO(rovere) be consistent between simulated tracks and
      // reconstructed tracks selection
      // count relevant particles
      if (((**iTP).pt() > 0.2) && (fabs((**iTP).eta()) < maxEta_) && (**iTP).charge() != 0) {
        vp->nGenTrk++;
      }
    }  // End of for loop on daughters sim-particles

    // Remove the SimVertex if I cannot reconstruct it 'cause I miss at the very least reco_tracks_for_reconstructable_simvertices_ tracks
    if (use_reconstructable_simvertices_ &&
        vp->num_matched_reco_tracks <= reco_tracks_for_reconstructable_simvertices_) {
      simpv.pop_back();
      continue;
    }
    if (vp->num_matched_reco_tracks)
      vp->average_match_quality /= static_cast<float>(vp->num_matched_reco_tracks);
  }  // End of for loop on tracking vertices

  // In case of no simulated vertices, break here
  if (simpv.empty())
    return simpv;

  // Now compute the closest distance in z between all simulated vertex
  // first initialize
  auto prev_z = simpv.back().z;
  for (simPrimaryVertex& vsim : simpv) {
    vsim.closest_vertex_distance_z = std::abs(vsim.z - prev_z);
    prev_z = vsim.z;
  }
  // then calculate
  for (std::vector<simPrimaryVertex>::iterator vsim = simpv.begin(); vsim != simpv.end(); vsim++) {
    std::vector<simPrimaryVertex>::iterator vsim2 = vsim;
    vsim2++;
    for (; vsim2 != simpv.end(); vsim2++) {
      double distance = std::abs(vsim->z - vsim2->z);
      // need both to be complete
      vsim->closest_vertex_distance_z = std::min(vsim->closest_vertex_distance_z, distance);
      vsim2->closest_vertex_distance_z = std::min(vsim2->closest_vertex_distance_z, distance);
    }
  }
  return simpv;
}

/* Extract information form recoVertex and fill the helper class
 * recoPrimaryVertex with proper reco-level information */
std::vector<VertexValidator::recoPrimaryVertex> VertexValidator::getRecoPVs(
    const edm::Handle<edm::View<reco::Vertex>>& tVC) {
  std::vector<VertexValidator::recoPrimaryVertex> recopv;

  for (auto v = tVC->begin(); v != tVC->end(); ++v) {
    // Skip junk vertices
    if (fabs(v->z()) > 1000)
      continue;
    if (v->isFake() || !v->isValid())
      continue;

    recoPrimaryVertex sv(v->position().x(), v->position().y(), v->position().z());
    sv.recVtx = &(*v);
    sv.recVtxRef = reco::VertexBaseRef(tVC, std::distance(tVC->begin(), v));
    // this is a new vertex, add it to the list of reco-vertices
    recopv.push_back(sv);
    auto* vp = &recopv.back();

    // Loop over daughter track(s)
    for (auto iTrack = v->tracks_begin(); iTrack != v->tracks_end(); ++iTrack) {
      auto momentum = (*(*iTrack)).innerMomentum();
      // TODO(rovere) better handle the pixelVertices, whose tracks
      // do not have the innerMomentum defined. This is a temporary
      // hack to overcome this problem.
      if (momentum.mag2() == 0)
        momentum = (*(*iTrack)).momentum();
      vp->pt += std::sqrt(momentum.perp2());
      vp->ptsq += (momentum.perp2());
      vp->nRecoTrk++;

      auto matched = r2s_->find(*iTrack);
      if (matched != r2s_->end()) {
        vp->num_matched_sim_tracks++;
      }

    }  // End of for loop on daughters reconstructed tracks
  }  // End of for loop on tracking vertices

  // In case of no reco vertices, break here
  if (recopv.empty())
    return recopv;

  // Now compute the closest distance in z between all reconstructed vertex
  // first initialize
  auto prev_z = recopv.back().z;
  for (recoPrimaryVertex& vreco : recopv) {
    vreco.closest_vertex_distance_z = std::abs(vreco.z - prev_z);
    prev_z = vreco.z;
  }
  for (std::vector<recoPrimaryVertex>::iterator vreco = recopv.begin(); vreco != recopv.end(); vreco++) {
    std::vector<recoPrimaryVertex>::iterator vreco2 = vreco;
    vreco2++;
    for (; vreco2 != recopv.end(); vreco2++) {
      double distance = std::abs(vreco->z - vreco2->z);
      // need both to be complete
      vreco->closest_vertex_distance_z = std::min(vreco->closest_vertex_distance_z, distance);
      vreco2->closest_vertex_distance_z = std::min(vreco2->closest_vertex_distance_z, distance);
    }
  }
  return recopv;
}

void VertexValidator::matchSim2RecoVertices(std::vector<simPrimaryVertex>& simpv,
                                            const reco::VertexSimToRecoCollection& vertex_s2r) {
  for (std::vector<simPrimaryVertex>::iterator vsim = simpv.begin(); vsim != simpv.end(); vsim++) {
    auto matched = vertex_s2r.find(vsim->sim_vertex);
    if (matched != vertex_s2r.end()) {
      for (const auto& vertexRefQuality : matched->val) {
        vsim->rec_vertices.push_back(&(*(vertexRefQuality.first)));
      }
    }

  }  // end for loop on simulated vertices
}

void VertexValidator::matchReco2SimVertices(std::vector<recoPrimaryVertex>& recopv,
                                            const reco::VertexRecoToSimCollection& vertex_r2s,
                                            const std::vector<simPrimaryVertex>& simpv) {
  for (std::vector<recoPrimaryVertex>::iterator vrec = recopv.begin(); vrec != recopv.end(); vrec++) {
    auto matched = vertex_r2s.find(vrec->recVtxRef);
    if (matched != vertex_r2s.end()) {
      for (const auto& vertexRefQuality : matched->val) {
        const auto tvPtr = &(*(vertexRefQuality.first));
        for (const auto& vv : simpv) {
          if (&(*(vv.sim_vertex)) == tvPtr) {
            vrec->sim_vertices.push_back(tvPtr);
            vrec->sim_vertices_num_shared_tracks.push_back(vertexRefQuality.second);
            vrec->sim_vertices_internal.push_back(&vv);
            continue;
          }
        }
      }
    }
  }  // end for loop on reconstructed vertices
}

void VertexValidator::calculatePurityAndFillHistograms(std::vector<recoPrimaryVertex>& recopvs,
                                                       int genpv_position_in_reco_collection,
                                                       bool signal_is_highest_pt) {
  if (recopvs.empty())
    return;

  std::vector<double> vtx_sumpt_sigmatched;
  std::vector<double> vtx_sumpt2_sigmatched;

  vtx_sumpt_sigmatched.reserve(recopvs.size());
  vtx_sumpt2_sigmatched.reserve(recopvs.size());

  // Calculate purity
  for (auto& v : recopvs) {
    double sumpt_all = 0;
    double sumpt_sigmatched = 0;
    double sumpt2_sigmatched = 0;
    const reco::Vertex* vertex = v.recVtx;
    for (auto iTrack = vertex->tracks_begin(); iTrack != vertex->tracks_end(); ++iTrack) {
      double pt = (*iTrack)->pt();
      sumpt_all += pt;
      if (matchRecoTrack2SimSignal(*iTrack)) {
        sumpt_sigmatched += pt;
        sumpt2_sigmatched += pt * pt;
      }
    }
    v.purity = sumpt_sigmatched / sumpt_all;

    vtx_sumpt_sigmatched.push_back(sumpt_sigmatched);
    vtx_sumpt2_sigmatched.push_back(sumpt2_sigmatched);
  }

  // const double vtxAll_sumpt_sigmatched = std::accumulate(vtx_sumpt_sigmatched.begin(), vtx_sumpt_sigmatched.end(), 0.0);
  // const double vtxNot0_sumpt_sigmatched = vtxAll_sumpt_sigmatched - vtx_sumpt_sigmatched[0];

  // Fill purity
  std::string prefix = "RecoPVAssoc2GenPVNotMatched_";
  if (genpv_position_in_reco_collection == 0)
    prefix = "RecoPVAssoc2GenPVMatched_";
}

bool VertexValidator::matchRecoTrack2SimSignal(const reco::TrackBaseRef& recoTrack) {
  auto found = r2s_->find(recoTrack);

  // reco track not matched to any TP
  if (found == r2s_->end())
    return false;

  // reco track matched to some TP from signal vertex
  for (const auto& tp : found->val) {
    if (tp.first->eventId().bunchCrossing() == 0 && tp.first->eventId().event() == 0)
      return true;
  }

  // reco track not matched to any TP from signal vertex
  return false;
}

void VertexValidator::beginJob() {
  edm::Service<TFileService> fs;
  output_tree_ = fs->make<TTree>("output", "Simple Track Validation TTree");
  output_tree_->Branch("rt", &num_total_reco_vertices_assoc2gen_);
  output_tree_->Branch("at", &num_total_reco_vertices_multiassoc2gen_);
  output_tree_->Branch("st", &num_total_gen_vertices_assoc2reco_);
  output_tree_->Branch("dt", &num_total_reco_vertices_duplicate_);
  output_tree_->Branch("ast", &num_total_gen_vertices_multiassoc2reco_);
}

void VertexValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::cout << __LINE__ << std::endl;
  using edm::Handle;
  using edm::View;
  using std::cout;
  using std::endl;
  using std::vector;
  using namespace reco;

  std::vector<float> pileUpInfo_z;

  // get the pileup information
  edm::Handle<std::vector<PileupSummaryInfo>> puinfoH;
  if (iEvent.getByToken(vecPileupSummaryInfoToken_, puinfoH)) {
    for (auto const& pu_info : *puinfoH.product()) {
      if (pu_info.getBunchCrossing() == 0) {
        pileUpInfo_z = pu_info.getPU_zpositions();
        break;
      }
    }
  }

  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(trackingParticleCollectionToken_, TPCollectionH);
  if (!TPCollectionH.isValid())
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "TPCollectionH is not valid";

  edm::Handle<TrackingVertexCollection> TVCollectionH;
  iEvent.getByToken(trackingVertexCollectionToken_, TVCollectionH);
  if (!TVCollectionH.isValid())
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "TVCollectionH is not valid";

  edm::Handle<reco::SimToRecoCollection> simToRecoH;
  iEvent.getByToken(simToRecoAssociationToken_, simToRecoH);
  if (simToRecoH.isValid())
    s2r_ = simToRecoH.product();
  else
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "simToRecoH is not valid";

  edm::Handle<reco::RecoToSimCollection> recoToSimH;
  iEvent.getByToken(recoToSimAssociationToken_, recoToSimH);
  if (recoToSimH.isValid())
    r2s_ = recoToSimH.product();
  else
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "recoToSimH is not valid";

  // Vertex associator
  edm::Handle<reco::VertexToTrackingVertexAssociator> vertexAssociatorH;
  iEvent.getByToken(vertexAssociatorToken_, vertexAssociatorH);
  if (!vertexAssociatorH.isValid()) {
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "vertexAssociatorH is not valid";
    return;
  }
  const reco::VertexToTrackingVertexAssociator& vertexAssociator = *(vertexAssociatorH.product());

  std::vector<simPrimaryVertex> simpv;  // a list of simulated primary
                                        // MC vertices
  simpv = getSimPVs(TVCollectionH);
  int kind_of_signal_vertex = 0;
  // int num_pileup_vertices = simpv.size();
  bool signal_is_highest_pt =
      std::max_element(simpv.begin(), simpv.end(), [](const simPrimaryVertex& lhs, const simPrimaryVertex& rhs) {
        return lhs.ptsq < rhs.ptsq;
      }) == simpv.begin();
  kind_of_signal_vertex |= (signal_is_highest_pt << HIGHEST_PT);

  std::vector<recoPrimaryVertex> recopv;  // a list of reconstructed
                                          // primary MC vertices
  edm::Handle<edm::View<reco::Vertex>> recVtxs;
  iEvent.getByToken(reco_vertex_collection_token_, recVtxs);

  reco::VertexRecoToSimCollection vertex_r2s = vertexAssociator.associateRecoToSim(recVtxs, TVCollectionH);
  reco::VertexSimToRecoCollection vertex_s2r = vertexAssociator.associateSimToReco(recVtxs, TVCollectionH);

  for (auto& v : simpv) {
    v.rec_vertices.clear();
  }
  matchSim2RecoVertices(simpv, vertex_s2r);
  recopv = getRecoPVs(recVtxs);
  matchReco2SimVertices(recopv, vertex_r2s, simpv);

  int num_total_gen_vertices_assoc2reco = 0;
  int num_total_reco_vertices_assoc2gen = 0;
  int num_total_gen_vertices_multiassoc2reco = 0;
  int num_total_reco_vertices_multiassoc2gen = 0;
  int num_total_reco_vertices_duplicate = 0;
  int genpv_position_in_reco_collection = -1;
  for (auto const& v : simpv) {
    if (v.eventId.event() == 0) {
      if (!recVtxs->empty() &&
          std::find(v.rec_vertices.begin(), v.rec_vertices.end(), &((*recVtxs.product())[0])) != v.rec_vertices.end()) {
        // mistag = 0.;
        kind_of_signal_vertex |= (1 << IS_ASSOC2FIRST_RECO);
      } else {
        if (!v.rec_vertices.empty()) {
          kind_of_signal_vertex |= (1 << IS_ASSOC2ANY_RECO);
        }
      }
      // Now check at which location the Simulated PV has been
      // reconstructed in the primary vertex collection
      // at-hand. Mark it with fake index -1 if it was not
      // reconstructed at all.

      auto iv = (*recVtxs.product()).begin();
      for (int pv_position_in_reco_collection = 0; iv != (*recVtxs.product()).end();
           ++pv_position_in_reco_collection, ++iv) {
        if (std::find(v.rec_vertices.begin(), v.rec_vertices.end(), &(*iv)) != v.rec_vertices.end()) {
          const bool genPVMatchedToRecoPV = (pv_position_in_reco_collection == 0);

          if (genPVMatchedToRecoPV) {
            auto pv = recopv[0];
            assert(pv.recVtx == &(*iv));
          }
          genpv_position_in_reco_collection = pv_position_in_reco_collection;
          break;
        }
      }
    }

    if (!v.rec_vertices.empty())
      num_total_gen_vertices_assoc2reco++;
    if (v.rec_vertices.size() > 1)
      num_total_gen_vertices_multiassoc2reco++;
  }
  calculatePurityAndFillHistograms(recopv, genpv_position_in_reco_collection, signal_is_highest_pt);

  for (auto& v : recopv) {
    if (!v.sim_vertices.empty()) {
      num_total_reco_vertices_assoc2gen++;
      if (v.sim_vertices_internal[0]->rec_vertices.size() > 1) {
        num_total_reco_vertices_duplicate++;
      }
    }
    if (v.sim_vertices.size() > 1)
      num_total_reco_vertices_multiassoc2gen++;
  }

  num_total_gen_vertices_assoc2reco_ = num_total_gen_vertices_assoc2reco;
  num_total_reco_vertices_assoc2gen_ = num_total_reco_vertices_assoc2gen;
  num_total_gen_vertices_multiassoc2reco_ = num_total_gen_vertices_multiassoc2reco;
  num_total_reco_vertices_multiassoc2gen_ = num_total_reco_vertices_multiassoc2gen;
  num_total_reco_vertices_duplicate_ = num_total_reco_vertices_duplicate;
  genpv_position_in_reco_collection_ = genpv_position_in_reco_collection;
}

void VertexValidator::endJob() {
  output_tree_->Fill();
  std::cout << num_total_gen_vertices_multiassoc2reco_ << std::endl;
}

DEFINE_FWK_MODULE(VertexValidator);
