#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TICL/interface/CaloClusterHostCollection.h"
#include "DataFormats/TICL/interface/ClusterMaskHost.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::CaloClusterHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(ticl::ClusterMaskHost);
