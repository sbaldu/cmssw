
#ifndef RecoHGCal_TICL_plugins_alpaka_PatternRecognitionByCLUEstering_h
#define RecoHGCal_TICL_plugins_alpaka_PatternRecognitionByCLUEstering_h

#include "CondCore/CondDB/interface/Exception.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterHost.h"
// #include "DataFormats/HGCalReco/interface/HGCalSoAClusters.h"
// #include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
// #include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
// #include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PatternRecognitionByCLUEstering {
  private:
    float m_rhoc;
    float m_dc;
    float m_dm;

  public:
    PatternRecognitionByCLUEstering(const edm::ParameterSet& config)
        : PatternRecognitionAlgoBase(config),
          m_rhoc(config.getParameter<float>("rho_c")),
          m_dc(config.getParameter<float>("dc")),
          m_dm(config.getParameter<float>("dm")) {}
    ~PatternRecognitionByCLUEstering() override = default;

    ticl::TracksterHost makeTracksters(Queue& queue, const reco::CaloClusterDeviceCollection& lc) override;

    static void fillPSetDescription(::edm::ParameterSetDescription& iDesc);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
