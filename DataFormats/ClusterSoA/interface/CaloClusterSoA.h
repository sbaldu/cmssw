#ifndef DataFormats_ClusterSoA_CaloClusterSoA_h
#define DataFormats_ClusterSoA_CaloClusterSoA_h

#include <array>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"
#include "SimDataFormats/Associations/interface/AssociationMap.h"

namespace {
    using TICL::FractionType;
}

enum AlgoId {
  island = 0,
  hybrid = 1,
  fixedMatrix = 2,
  dynamicHybrid = 3,
  multi5x5 = 4,
  particleFlow = 5,
  hgcal_em = 6,
  hgcal_had = 7,
  hgcal_scintillator = 8,
  hfnose = 9,
  undefined = 1000
};

GENERATE_SOA_LAYOUT(Position4D_Energy_SoALayout,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(float, raw_energy),
                    SOA_COLUMN(float, corrected_energy),
                    SOA_COLUMN(float, time))

using Position4D_Energy_SoA = Position4D_Energy_SoALayout<>;
using Position4D_Energy_SoAView = Position4D_Energy_SoA::View;
using Position4D_Energy_SoAConstView = Position4D_Energy_SoA::ConstView;

// This layout holds the minimal set of "common" fields
GENERATE_SOA_LAYOUT(Position4D_Energy_Errors_SoALayout,
                    SOA_COLUMN(float, xErr),  // error on x, if relevant
                    SOA_COLUMN(float, yErr),
                    SOA_COLUMN(float, zErr),
                    SOA_COLUMN(float, timeErr),
                    SOA_COLUMN(float, energyErr))

using Position4D_Energy_Errors_SoA = Position4D_Energy_Errors_SoALayout<>;
using Position4D_Energy_Errors_SoAView = Position4D_Energy_Errors_SoA::View;
using Position4D_Energy_Errors_SoAConstView = Position4D_Energy_Errors_SoA::ConstView;

template <typename TDev>
struct CaloClusterExtra_SoALayoutStruct {
  using AssociationMap = TICL::AssociationMap<TDev, FractionType>;

  GENERATE_SOA_LAYOUT(CaloClusterExtra_SoALayout,
                      SOA_COLUMN(AlgoId, algoId),
                      SOA_COLUMN(uint32_t, CaloID),
                      SOA_COLUMN(uint32_t, flags),
                      SOA_COLUMN(uint32_t, seedId),
                      SOA_SCALAR(AssociationMap, assocMap))
};

template <typename TDev>
using CaloClusterExtra_SoA = typename CaloClusterExtra_SoALayoutStruct<TDev>::template Layout<>;
template <typename TDev>
using CaloClusterExtra_SoAView = typename CaloClusterExtra_SoALayoutStruct<TDev>::template Layout<>::View;
template <typename TDev>
using CaloClusterExtra_SoAConstView = typename CaloClusterExtra_SoALayoutStruct<TDev>::template Layout<>::ConstView;

#endif  // DataFormats_ClusterSoA_CaloClusterSoA_h
