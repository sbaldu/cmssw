#ifndef RecoHGCal_TICL_PatternRecognitionPluginFactory_h
#define RecoHGCal_TICL_PatternRecognitionPluginFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"

typedef edmplugin::PluginFactory<ticl::PatternRecognitionAlgoBaseT<ticl::TICLLayerTilesHost>*(const edm::ParameterSet&,
                                                                                              edm::ConsumesCollector)>
    PatternRecognitionFactory;
typedef edmplugin::PluginFactory<ticl::PatternRecognitionAlgoBaseT<ticl::TICLLayerTilesHFNoseHost>*(
    const edm::ParameterSet&, edm::ConsumesCollector)>
    PatternRecognitionHFNoseFactory;
typedef edmplugin::PluginFactory<ticl::PatternRecognitionAlgoBaseT<ticl::TICLLayerTilesBarrelHost>*(
    const edm::ParameterSet&, edm::ConsumesCollector)>
    PatternRecognitionBarrelFactory;
#endif
