
#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
//#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"
//#include "DataFormats/ClusterSoA/interface/CaloClusterSoA.h"
#include "DataFormats/ClusterSoA/interface/CaloClusterSoAHostCollection.h"

template <typename TDev>
SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(CaloClusterSoAHostCollection<TDev>);
