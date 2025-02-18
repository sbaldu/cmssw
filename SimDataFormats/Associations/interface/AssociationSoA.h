
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace cms::alpakatools {

  template <typename ValueType, typename Score>
  struct AssociationElementsSoAStruct {
    GENERATE_SOA_LAYOUT(Layout, SOA_COLUMN(ValueType, values), SOA_COLUMN(Score, scores), SOA_COLUMN(int, indexes))
  };

  template <typename ValueType>
  struct AssociationElementsSoAStruct<ValueType, void> {
    GENERATE_SOA_LAYOUT(Layout, SOA_COLUMN(ValueType, values), SOA_COLUMN(int, indexes))
  };

  template <>
  struct AssociationElementsSoAStruct<void, void> {
    GENERATE_SOA_LAYOUT(Layout, SOA_COLUMN(int, indexes))
  };

  template <typename ValueType, typename Score>
  using AssociationElementsSoA = typename AssociationElementsSoAStruct<ValueType, Score>::template Layout<>;
  template <typename ValueType, typename Score>
  using AssociationElementsSoAView = typename AssociationElementsSoAStruct<ValueType, Score>::template Layout<>::View;

}  // namespace cms::alpakatools
