#ifndef DataFormats_ClusterSoA_CaloClusterSoAHostCollection_h
#define DataFormats_ClusterSoA_CaloClusterSoAHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/ClusterSoA/interface/CaloClusterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

using ALPAKA_ACCELERATOR_NAMESPACE::Device;

using CaloClusterSoAHostCollection =
    PortableHostCollection3<Position4D_Energy_SoA, Position4D_Energy_Errors_SoA, CaloClusterExtra_SoA<Device>>;

#endif
