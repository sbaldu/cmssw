// Author: Marco Rovere, marco.rovere@cern.ch
// Date: 05/2019
//
#include <memory>  // unique_ptr

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TilesHost.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

class TICLLayerTileProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  explicit TICLLayerTileProducer(const edm::ParameterSet &ps);
  ~TICLLayerTileProducer() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_HFNose_token_;
  edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  ticlgeom::Tools rhtools_;
  std::string detector_;
  bool doNose_;
  bool doBarrel_;
};

TICLLayerTileProducer::TICLLayerTileProducer(const edm::ParameterSet &ps)
    : detector_(ps.getParameter<std::string>("detector")) {
  ticlGeomToken_ = esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", ""));
  ticlGeomLookupToken_ =
      esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", ""));
  ticlGeomLayersToken_ =
      esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", ""));

  doNose_ = (detector_ == "HFNose");
  doBarrel_ = (detector_ == "Barrel");

  if (doNose_) {
    clusters_HFNose_token_ =
        consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_HFNose_clusters"));
    produces<ticl::TICLLayerTilesHFNoseHost>();
  } else {
    if (doBarrel_) {
      produces<ticl::TICLLayerTilesBarrelHost>("ticlLayerTilesBarrel");
    }
    clusters_token_ = consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"));
    produces<ticl::TICLLayerTilesHost>();
  }
}

void TICLLayerTileProducer::beginRun(edm::Run const &, edm::EventSetup const &es) {
  rhtools_.setGeometry(es.getData(ticlGeomToken_), es.getData(ticlGeomLookupToken_), es.getData(ticlGeomLayersToken_));
}

void TICLLayerTileProducer::produce(edm::Event &evt, const edm::EventSetup &) {
  using Acc = alpaka_serial_sync::Acc1D;

  edm::Handle<std::vector<reco::CaloCluster>> cluster_h;
  if (doNose_)
    evt.getByToken(clusters_HFNose_token_, cluster_h);
  else
    evt.getByToken(clusters_token_, cluster_h);

  const auto &layerClusters = *cluster_h;

  std::array<std::vector<float>, ticl::TICLLayerTilesHost::TilesType::nLayers> etas;
  std::array<std::vector<float>, ticl::TICLLayerTilesHost::TilesType::nLayers> phis;
  std::array<std::vector<uint32_t>, ticl::TICLLayerTilesHost::TilesType::nLayers> lcIds;
  std::array<std::vector<float>, ticl::TICLLayerTilesBarrelHost::TilesType::nLayers> barrel_etas;
  std::array<std::vector<float>, ticl::TICLLayerTilesBarrelHost::TilesType::nLayers> barrel_phis;
  std::array<std::vector<uint32_t>, ticl::TICLLayerTilesBarrelHost::TilesType::nLayers> barrel_lcIds;
  std::array<std::vector<float>, ticl::TICLLayerTilesHFNoseHost::TilesType::nLayers> nose_etas;
  std::array<std::vector<float>, ticl::TICLLayerTilesHFNoseHost::TilesType::nLayers> nose_phis;
  std::array<std::vector<uint32_t>, ticl::TICLLayerTilesHFNoseHost::TilesType::nLayers> nose_lcIds;
  auto lcId = 0;
  for (auto const &lc : layerClusters) {
    const auto firstHitDetId = lc.hitsAndFractions()[0].first;
    auto layer = rhtools_.getLayerWithOffset(firstHitDetId);
    const auto isBarrelLC = rhtools_.isBarrel(firstHitDetId);
    if (!isBarrelLC) {
      layer += rhtools_.lastLayer(doNose_) * ((rhtools_.zside(firstHitDetId) + 1) >> 1) - 1;
    }

    if (doNose_) {
      nose_etas[layer].push_back(lc.eta());
      nose_phis[layer].push_back(lc.phi());
      nose_lcIds[layer].push_back(lcId);
    } else if (doBarrel_ && isBarrelLC) {
      barrel_etas[layer].push_back(lc.eta());
      barrel_phis[layer].push_back(lc.phi());
      barrel_lcIds[layer].push_back(lcId);
    } else if (!isBarrelLC) {
      etas[layer].push_back(lc.eta());
      phis[layer].push_back(lc.phi());
      lcIds[layer].push_back(lcId);
    }
    ++lcId;
  }

  alpaka_serial_sync::Queue queue(cms::alpakatools::host());
  if (doNose_) {
    std::array<int, ticl::TICLLayerTilesHFNoseHost::TilesType::nLayers> nose_sizes;
    std::transform(nose_etas.begin(),
                   nose_etas.begin() + ticl::TICLLayerTilesHFNoseHost::TilesType::nLayers,
                   nose_sizes.begin(),
                   [](const auto &layer) { return layer.size(); });

    auto resultHFNose = std::make_unique<ticl::TICLLayerTilesHFNoseHost>(nose_sizes);
    for (auto layer = 0; layer < ticl::TICLLayerTilesHFNoseHost::TilesType::nLayers; ++layer) {
      (*resultHFNose)[layer].template fill<Acc>(queue, nose_etas[layer], nose_phis[layer], nose_lcIds[layer]);
    }
    evt.put(std::move(resultHFNose));
  } else {
    if (doBarrel_) {
      std::array<int, ticl::TICLLayerTilesBarrelHost::TilesType::nLayers> barrel_sizes;
      std::ranges::transform(barrel_etas, barrel_sizes.begin(), [](const auto &layer) { return layer.size(); });
      auto resultBarrel = std::make_unique<ticl::TICLLayerTilesBarrelHost>(barrel_sizes);
      for (auto barrel_layer = 0; barrel_layer < ticl::TICLLayerTilesBarrelHost::TilesType::nLayers; ++barrel_layer) {
        (*resultBarrel)[barrel_layer].template fill<Acc>(
            queue, barrel_etas[barrel_layer], barrel_phis[barrel_layer], barrel_lcIds[barrel_layer]);
      }
      evt.put(std::move(resultBarrel), "ticlLayerTilesBarrel");
    } else {
      std::array<int, ticl::TICLLayerTilesHost::TilesType::nLayers> sizes;
      std::ranges::transform(etas, sizes.begin(), [](const auto &layer) { return layer.size(); });
      auto result = std::make_unique<ticl::TICLLayerTilesHost>(sizes);
      for (auto layer = 0; layer < ticl::TICLLayerTilesHost::TilesType::nLayers; ++layer) {
        (*result)[layer].template fill<Acc>(queue, etas[layer], phis[layer], lcIds[layer]);
      }
      evt.put(std::move(result));
    }
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
