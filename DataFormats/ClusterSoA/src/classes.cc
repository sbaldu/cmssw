
#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
//#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"
//#include "DataFormats/ClusterSoA/interface/CaloClusterSoA.h"
#include "DataFormats/ClusterSoA/interface/CaloClusterSoAHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(CaloClusterSoAHostCollection<Device>);
}
