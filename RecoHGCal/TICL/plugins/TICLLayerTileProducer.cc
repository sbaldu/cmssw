// Author: Marco Rovere, marco.rovere@cern.ch
// Date: 05/2019
//
#include <memory>  // unique_ptr

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

class TICLLayerTileProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  explicit TICLLayerTileProducer(const edm::ParameterSet &ps);
  ~TICLLayerTileProducer() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::EDGetTokenT<reco::CaloClusterHostCollection> clusters_token_;
  edm::EDGetTokenT<reco::CaloClusterHostCollection> clusters_HFNose_token_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  hgcal::RecHitTools rhtools_;
  std::string detector_;
  bool doNose_;
  bool doBarrel_;
};

TICLLayerTileProducer::TICLLayerTileProducer(const edm::ParameterSet &ps)
    : detector_(ps.getParameter<std::string>("detector")) {
  geometry_token_ = esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>();

  doNose_ = (detector_ == "HFNose");
  doBarrel_ = (detector_ == "Barrel");

  if (doNose_) {
    clusters_HFNose_token_ =
        consumes<reco::CaloClusterHostCollection>(ps.getParameter<edm::InputTag>("layer_HFNose_clusters"));
    produces<TICLLayerTilesHFNose>();
  } else {
    if (doBarrel_) {
      produces<TICLLayerTilesBarrel>("ticlLayerTilesBarrel");
    }
    clusters_token_ = consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"));
    clusters_token_ = consumes<reco::CaloClusterHostCollection>(ps.getParameter<edm::InputTag>("layer_clusters"));
    produces<TICLLayerTiles>();
  }
}

void TICLLayerTileProducer::beginRun(edm::Run const &, edm::EventSetup const &es) {
  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);
}

void TICLLayerTileProducer::produce(edm::Event &evt, const edm::EventSetup &) {
  std::unique_ptr<TICLLayerTilesHFNose> resultHFNose;
  std::unique_ptr<TICLLayerTiles> result;
  std::unique_ptr<TICLLayerTilesBarrel> resultBarrel;
  if (doNose_) {
    resultHFNose = std::make_unique<TICLLayerTilesHFNose>();
  } else {
    if (doBarrel_)
      resultBarrel = std::make_unique<TICLLayerTilesBarrel>();
    result = std::make_unique<TICLLayerTiles>();
  }

  edm::Handle<reco::CaloClusterHostCollection> cluster_h;
  if (doNose_)
    evt.getByToken(clusters_HFNose_token_, cluster_h);
  else
    evt.getByToken(clusters_token_, cluster_h);

  const auto &layerClusters = *cluster_h;
  int lcId = 0;
  for (auto lc_idx = 0; lc_idx < layerClusters.view().position().metadata().size(); ++lc_idx) {
    auto layer = layerClusters.view().position()[lc_idx].layer();
    const auto x = layerClusters.view().position()[lc_idx].x();
    const auto y = layerClusters.view().position()[lc_idx].y();
    const auto z = layerClusters.view().position()[lc_idx].z();
    const auto cells = layerClusters.view().position()[lc_idx].cells();
    const auto energy = layerClusters.view().energy()[lc_idx].energy();
    assert(layer >= 0);

    const auto seed_detid = layerClusters.view().indexes()[lc_idx].seedID();

    const auto isBarrelLC = rhtools_.isBarrel(seed_detid);
    if (!isBarrelLC) {
      layer += rhtools_.lastLayer(doNose_) * ((rhtools_.zside(seed_detid) + 1) >> 1) - 1;
    }

    const auto eta = layerClusters.view().eta(lc_idx);
    const auto phi = layerClusters.view().phi(lc_idx);
    if (doNose_) {
      resultHFNose->fill(layer, eta, phi, lc_idx);
      LogDebug("TICLLayerTileProducer") << "Adding layerClusterId: " << lcId << " into bin [eta,phi]: [ "
                                        << (*resultHFNose)[layer].etaBin(eta) << ", "
                                        << (*resultHFNose)[layer].phiBin(phi) << "] for layer: " << layer;
    } else if (doBarrel_ && isBarrelLC) {
      resultBarrel->fill(layer, eta, phi, lc_idx);
      LogDebug("TICLLayerTileProducer") << "Adding layerClusterId: " << lcId << " into bin [eta,phi]: [ "
                                        << (*resultBarrel)[layer].etaBin(eta) << ", "
                                        << (*resultBarrel)[layer].phiBin(phi) << "] for layer: " << layer;
    } else {
      result->fill(layer, eta, phi, lc_idx);
      LogDebug("TICLLayerTileProducer") << "Adding layerClusterId: " << lcId << " into bin [eta,phi]: [ "
                                        << (*result)[layer].etaBin(eta) << ", "
                                        << (*result)[layer].phiBin(phi) << "] for layer: " << layer;
    }
  }

  if (doNose_)
    evt.put(std::move(resultHFNose));
  else {
    if (doBarrel_)
      evt.put(std::move(resultBarrel), "ticlLayerTilesBarrel");
    else
      evt.put(std::move(result));
  }
}

void TICLLayerTileProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layer_HFNose_clusters", edm::InputTag("hgcalLayerClustersHFNose"));
  descriptions.add("ticlLayerTileProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLLayerTileProducer);
