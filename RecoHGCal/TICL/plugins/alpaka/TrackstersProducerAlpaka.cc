#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include <alpaka/alpaka.hpp>
#include "CLUEsteringAlgo.h"
#include "PointsAlpaka.h"
#include "Points.h"
#include "TilesAlpaka.h"
#include "ConvolutionalKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackstersProducerAlpaka : public global::EDProducer<> {
  public:
    TrackstersProducerAlpaka(edm::ParameterSet const& config)
        : clusters_token_(
              consumes<std::vector<reco::CaloCluster>>(config.getParameter<edm::InputTag>("layer_clusters"))) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
      // desc.add<int32_t>("size2");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      const auto& layerClusters = event.get(clusters_token_);
      const int Ndim = 3;

      // Host containers
      std::vector<VecArray<float, Ndim>> m_coords;
      std::vector<float> m_weight;

      // int n_points = layerClusters.size();

      for (auto const& lc : layerClusters) {
        VecArray<float, Ndim> lc_coords;
        float lc_x = lc.x();
        float lc_y = lc.y();
        float lc_z = lc.z();
        float lc_energy = lc.energy();

        lc_coords.push_back_unsafe(lc_x);
        lc_coords.push_back_unsafe(lc_y);
        lc_coords.push_back_unsafe(lc_z);

        m_coords.push_back(lc_coords);
        m_weight.push_back(lc_energy);
      }

      CLUEsteringAlgo<Acc1D, Ndim> algo_(0.5, 0.1, 1, 100, event.queue());

      // host and device points
      Points<Ndim> h_points(m_coords, m_weight);
      PointsAlpaka<Ndim> d_points(event.queue(), m_weight.size());
      FlatKernel kernel(0.5f);

      algo_.make_clusters(h_points, d_points, kernel, event.queue(), 256);
    }

  private:
    const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TrackstersProducerAlpaka);
