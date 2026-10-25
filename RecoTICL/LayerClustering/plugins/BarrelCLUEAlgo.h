#ifndef RecoParticleFlow_PFClusterProducer_BarrelCLUEAlgo_h
#define RecoParticleFlow_PFClusterProducer_BarrelCLUEAlgo_h

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "RecoTICL/LayerClustering/interface/HGCalLayerTiles.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalPFCuts.h"

// C/C++ headers
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using Density = hgcal_clustering::Density;

template <typename TILE>
class BarrelCLUEAlgoT : public HGCalClusteringAlgoBase {
public:
  BarrelCLUEAlgoT(const edm::ParameterSet& ps)
      : HGCalClusteringAlgoBase(
            (HGCalClusteringAlgoBase::VerbosityLevel)ps.getUntrackedParameter<unsigned int>("verbosity", 3),
            reco::CaloCluster::undefined),
        deltac_(ps.getParameter<double>("deltac")),
        kappa_(ps.getParameter<double>("kappa")),
        maxLayerIndex_(ps.getParameter<int>("maxLayerIndex")) {
    // CLUEstering does not implement the sharing of rechits between clusters
    if (ps.getParameter<bool>("doSharing")) {
      throw cms::Exception("Configuration") << "BarrelCLUEAlgo: doSharing is not supported";
    }
  }
  ~BarrelCLUEAlgoT() override {}

  void getEventSetupPerAlgorithm(const edm::EventSetup& es) override;
  void setThresholds(edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd>,
                     edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd>) override;

  void populate(const HGCRecHitCollection& hits) override {};
  void populate(const reco::PFRecHitCollection& hits) override;
  // this is the method that will start the clusterisation (it is possible to invoke this method
  // more than once - but make sure it is with different hit collections (or else use reset)

  void makeClusters() override;

  // this is the method to get the cluster collection out
  std::vector<reco::BasicCluster> getClusters(bool) override;

  void reset() override {
    clusters_v_.clear();
    cells_.clear();
    std::fill(layerSizes_.begin(), layerSizes_.end(), 0u);
    numberOfClusters_ = 0;
  }

  void computeThreshold();

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    // Unused: as in the original implementation, rechits below the density threshold are only
    // attached to a higher-density rechit within deltac
    iDesc.add<double>("outlierDeltaFactor", 2.);
    iDesc.add<double>("kappa", 1.34);
    iDesc.add<int>("maxLayerIndex", 0);
    iDesc.add<double>("deltac", 0.0175);
    // Unused, only relevant for the sharing
    iDesc.add<double>("fractionCutoff", 0.);
    iDesc.add<bool>("doSharing", false);
  }

  /// point in the space
  typedef math::XYZPoint Point;

private:
  // To get ECAL sigmaNoise
  edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> tok_ebThresholds_;
  const EcalPFRecHitThresholds* ebThresholds_;
  //To get HCAL sigmaNoise
  edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> tok_hcalThresholds_;
  const HcalPFCuts* hcalThresholds_;
  // The two parameters used to identify clusters
  double deltac_;
  double kappa_;
  int maxLayerIndex_;

  // The rechits of all the layers, stored contiguously layer after layer: CLUEstering
  // clusters each layer separately, given the number of rechits on each of them
  struct BarrelCells {
    std::vector<uint32_t> detid;
    std::vector<float> eta;
    std::vector<float> phi;
    std::vector<float> r;
    std::vector<float> weight;
    std::vector<float> sigmaNoise;
    std::vector<int> clusterIndex;
    std::vector<bool> isSeed;

    void resize(std::size_t size) {
      detid.resize(size);
      eta.resize(size);
      phi.resize(size);
      r.resize(size);
      weight.resize(size);
      sigmaNoise.resize(size);
      clusterIndex.resize(size);
      isSeed.resize(size);
    }

    void clear() {
      detid.clear();
      eta.clear();
      phi.clear();
      r.clear();
      weight.clear();
      sigmaNoise.clear();
      clusterIndex.clear();
      isSeed.clear();
    }
  };

  BarrelCells cells_;
  std::vector<uint32_t> layerSizes_;
  unsigned int numberOfClusters_ = 0;

  static unsigned int layerIndex(DetId detid);
  void prepareDataStructures();
  math::XYZPoint calculatePosition(const std::vector<int>& cellsInCluster) const;
};

// explicit template instantiation
extern template class BarrelCLUEAlgoT<EBLayerTiles>;
extern template class BarrelCLUEAlgoT<HBLayerTiles>;

using EBCLUEAlgo = BarrelCLUEAlgoT<EBLayerTiles>;
using HBCLUEAlgo = BarrelCLUEAlgoT<HBLayerTiles>;

#endif
