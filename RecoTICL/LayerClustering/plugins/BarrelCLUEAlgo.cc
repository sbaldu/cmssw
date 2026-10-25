
#include "BarrelCLUEAlgo.h"

#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <cmath>
#include <limits>
#include <numeric>

#include <alpaka/alpaka.hpp>
#include <CLUEstering/CLUEstering.hpp>

using namespace hgcal_clustering;

template <typename T>
void BarrelCLUEAlgoT<T>::setThresholds(edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> EBThres,
                                       edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> HCALThres) {
  tok_ebThresholds_ = EBThres;
  tok_hcalThresholds_ = HCALThres;
}

template <typename T>
void BarrelCLUEAlgoT<T>::getEventSetupPerAlgorithm(const edm::EventSetup& es) {
  cells_.clear();
  numberOfClusters_ = 0;
  ebThresholds_ = &es.getData(tok_ebThresholds_);
  hcalThresholds_ = &es.getData(tok_hcalThresholds_);
  maxlayer_ = maxLayerIndex_;  //for testing purposes
  layerSizes_.assign(maxlayer_ + 1, 0u);
}

template <typename T>
unsigned int BarrelCLUEAlgoT<T>::layerIndex(DetId detid) {
  unsigned int layer = 0;
  if (detid.det() == DetId::Hcal) {
    HcalDetId hid(detid);
    layer = hid.depth();
    if (detid.subdetId() == HcalSubdetector::HcalOuter)
      layer += 1;
  }
  return layer;
}

template <typename T>
void BarrelCLUEAlgoT<T>::populate(const reco::PFRecHitCollection& hits) {
  // First count the rechits on each layer, then store each of them in the range of its layer
  constexpr auto kInvalidLayer = std::numeric_limits<unsigned int>::max();
  std::vector<unsigned int> layers(hits.size());
  for (unsigned int i = 0; i < hits.size(); ++i) {
    const auto layer = layerIndex(DetId(hits[i].detId()));
    // Rechits on a layer beyond maxLayerIndex are not clustered
    if (layer >= layerSizes_.size()) {
      layers[i] = kInvalidLayer;
      continue;
    }
    layers[i] = layer;
    ++layerSizes_[layer];
  }

  std::vector<uint32_t> cursors(layerSizes_.size());
  std::exclusive_scan(layerSizes_.begin(), layerSizes_.end(), cursors.begin(), 0u);
  cells_.resize(std::accumulate(layerSizes_.begin(), layerSizes_.end(), 0u));

  for (unsigned int i = 0; i < hits.size(); ++i) {
    if (layers[i] == kInvalidLayer)
      continue;
    const reco::PFRecHit& hit = hits[i];
    DetId detid(hit.detId());
    const GlobalPoint position = rhtools_.getPosition(detid);
    const auto j = cursors[layers[i]]++;

    cells_.detid[j] = detid.rawId();
    cells_.eta[j] = position.eta();
    cells_.phi[j] = position.phi();
    cells_.weight[j] = hit.energy();
    cells_.r[j] = position.mag();
    float sigmaNoise = 0.f;
    if (detid.det() == DetId::Ecal) {
      sigmaNoise = (*ebThresholds_)[detid];
    } else {
      const HcalPFCut* item = (*hcalThresholds_).getValues(detid);
      sigmaNoise = item->seedThreshold();
    }
    cells_.sigmaNoise[j] = sigmaNoise;
  }
}

template <typename T>
void BarrelCLUEAlgoT<T>::prepareDataStructures() {
  const auto cellsSize = cells_.detid.size();
  cells_.clusterIndex.assign(cellsSize, -1);
  cells_.isSeed.assign(cellsSize, false);
}

template <typename T>
void BarrelCLUEAlgoT<T>::makeClusters() {
  std::cout << "inside make clusters\n";
  prepareDataStructures();
  const auto nCells = static_cast<int32_t>(cells_.detid.size());
  if (nCells == 0)
    return;

  auto queue = clue::get_queue(0u);
  clue::PointsHost<2> h_points(
      queue, nCells, cells_.eta.data(), cells_.phi.data(), cells_.weight.data(), cells_.clusterIndex.data());
  h_points.set_density_uncertainty(cells_.sigmaNoise);
  h_points.set_tags(cells_.detid);
  clue::PointsDevice<2> d_points(queue, nCells);

  // As in the original implementation, rechits below the density threshold are only attached to a
  // higher-density rechit within deltac: the outlier distance is deltac as well.
  clue::Clusterer<2> clusterer(queue, deltac_, kappa_, deltac_);
  constexpr float kTwoPi = 2.f * static_cast<float>(M_PI);
  clusterer.setWrappedCoordinates(0, 1);
  clusterer.make_clusters(queue,
                          h_points,
                          d_points,
                          std::span<const uint32_t>(layerSizes_),
                          clue::PeriodicEuclideanMetric<2, float>{{0.f, kTwoPi}},
                          clue::FlatKernel<float>{0.5f});
  alpaka::wait(queue);

  const auto seeds = clusterer.getSeeds();
  for (const auto seed : seeds) {
    cells_.isSeed[seed] = true;
  }
  numberOfClusters_ = seeds.size();
}

template <typename T>
std::vector<reco::BasicCluster> BarrelCLUEAlgoT<T>::getClusters(bool) {
  std::vector<std::vector<int>> cellsInCluster(numberOfClusters_);
  for (unsigned int i = 0; i < cells_.detid.size(); ++i) {
    const auto clusterIndex = cells_.clusterIndex[i];
    if (clusterIndex >= 0)
      cellsInCluster[clusterIndex].push_back(i);
  }

  clusters_v_.clear();
  clusters_v_.reserve(numberOfClusters_);
  std::vector<std::pair<DetId, float>> thisCluster;
  for (const auto& cl : cellsInCluster) {
    auto position = calculatePosition(cl);
    float energy = 0.f;
    int seedDetId = -1;

    for (const auto cellIdx : cl) {
      energy += cells_.weight[cellIdx];
      thisCluster.emplace_back(DetId(cells_.detid[cellIdx]), 1.f);
      if (cells_.isSeed[cellIdx]) {
        seedDetId = cells_.detid[cellIdx];
      }
    }

    if constexpr (std::is_same_v<T, EBLayerTiles>) {
      clusters_v_.emplace_back(energy, position, reco::CaloID::DET_ECAL_BARREL, thisCluster, algoId_);
    } else if constexpr (std::is_same_v<T, HBLayerTiles>) {
      clusters_v_.emplace_back(energy, position, reco::CaloID::DET_HCAL_BARREL, thisCluster, algoId_);
    } else {
      clusters_v_.emplace_back(energy, position, reco::CaloID::DET_HO, thisCluster, algoId_);
    }
    clusters_v_.back().setSeed(seedDetId);
    thisCluster.clear();
  }
  return clusters_v_;
}

template <typename T>
math::XYZPoint BarrelCLUEAlgoT<T>::calculatePosition(const std::vector<int>& cellsInCluster) const {
  float total_weight = 0.f;
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;

  for (const auto i : cellsInCluster) {
    float rhEnergy = cells_.weight[i];
    total_weight += rhEnergy;
    float theta = 2 * std::atan(std::exp(-cells_.eta[i]));
    const GlobalPoint gp(GlobalPoint::Polar(theta, cells_.phi[i], cells_.r[i]));
    x += gp.x() * rhEnergy;
    y += gp.y() * rhEnergy;
    z += gp.z() * rhEnergy;
  }
  if (total_weight != 0.f) {
    float inv_total_weight = 1.f / total_weight;
    return math::XYZPoint(x * inv_total_weight, y * inv_total_weight, z * inv_total_weight);
  } else {
    return math::XYZPoint(0.f, 0.f, 0.f);
  }
}

template class BarrelCLUEAlgoT<EBLayerTiles>;
template class BarrelCLUEAlgoT<HBLayerTiles>;
