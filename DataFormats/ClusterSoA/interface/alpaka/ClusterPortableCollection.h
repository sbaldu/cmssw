#ifndef DataFormats_Portable_interface_alpaka_ClusterPortableCollection_h
#define DataFormats_Portable_interface_alpaka_ClusterPortableCollection_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// Generic headers for portable collections
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

// The SoAs we want to combine
#include "DataFormats/ClusterSoA/interface/CaloClusterSoA.h"
#include "DataFormats/ClusterSoA/interface/CaloClusterSoAHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // However, if you have multiple SoAs to unify, use PortableMultiCollection:
  template <typename TDev>
  using CaloClusterSoACollection =
      PortableCollection3<Position4D_Energy_SoA, Position4D_Energy_Errors_SoA, CaloClusterExtra_SoA<TDev>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(CaloClusterSoACollection, ::CaloClusterSoAHostCollection);

#endif  // DataFormats_Portable_interface_alpaka_ClusterPortableCollection_h
