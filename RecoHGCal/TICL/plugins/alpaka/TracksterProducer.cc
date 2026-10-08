#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "RecoHGCal/TICL/plugins/alpaka/PatternRecognitionByCLUEstering.h"
#include "CLUEstering/CLUEstering.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TracksterProducer : public stream::EDProducer<> {
  public:
    TracksterProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          deviceTokenSoAClusters_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          legacyTrackstersToken_{produces()} {}
    ~TracksterProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalSoALayerClusters"));

      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      const auto& lc = iEvent.get(deviceTokenSoAClusters_);
      auto& queue = iEvent.queue();
      auto tracksters = algo_->makeTracksters(queue, lc);
      // std::cout << "n tracksters = " << tracksters.size() << std::endl;
      iEvent.emplace(legacyTrackstersToken_, std::move(tracksters));
    }

  private:
    device::EDGetToken<reco::CaloClusterDeviceCollection> const deviceTokenSoAClusters_;
    edm::EDPutTokenT<ticl::TracksterHost> const legacyTrackstersToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterProducer);
