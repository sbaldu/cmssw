
#include "AlgoFillAssociator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  AlgoFillAssociator::fill(AssociatorMap<FractionType, void, void, void>& map,
                           const int* assocIds,
                           int nbins,
                           const int* indexes,
                           const FractionType* values,
                           size_t size,
                           Queue queue) {
    map.fill(assocIds, nbins, indexes, values, size, queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
