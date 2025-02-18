
#include "SimDataFormats/Associations/interface/AssociationMap.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class AlgoFillAssociator {
  public:
    void fill(AssociatorMap<FractionType, void, void, void>& map,
              const int* assocIds,
              int nbins,
              const int* indexes,
              const FractionType* values,
              size_t size,
              Queue queue);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
