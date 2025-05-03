#include "./clueVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
 namespace clueVertexFinder {
      template <int dim, typename TrackerTraits>
      void Producer<dim, TrackerTraits>::makeClusters(clue::PointsHost<dim>& h_points, clue::PointsDevice<dim, Device>& d_points, Queue& queue) {
    
        clue::Clusterer<dim> algo(m_dc, m_rhoc, m_dm, m_pPBin, queue);
        const std::size_t block_size{256};
        algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
    
      }    
  } // namespace clueVertexFinder
} //namespace ALPAKA_ACCELERATOR_NAMESPACE
