#ifndef DataFormats_ClusterSoA_CaloClusterSoAHostCollection_h
#define DataFormats_ClusterSoA_CaloClusterSoAHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/ClusterSoA/interface/CaloClusterSoA.h"

template <typename TDev>
using CaloClusterSoAHostCollection = PortableHostCollection3<Position4D_Energy_SoA, Position4D_Energy_Errors_SoA, CaloClusterExtra_SoA<TDev>>;

#endif 
