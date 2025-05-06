#include "./clueVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
 namespace clueVertexFinder {
/*      void Producer::makeClusters(clue::PointsHost<1>& h_points, clue::PointsDevice<1, Device>& d_points, Queue& queue) {
    
        clue::Clusterer<1> algo(queue, m_dc, m_rhoc, m_dm);
        const std::size_t block_size{256};
        algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
    
      } 
*/
      void Producer::makeClusters(std::vector<float>& coords, std::vector<int>& results, Queue& queue) {
        
        int nTracks = results.size();      
        clue::PointsHost<1> h_points(queue, nTracks, coords, results);
        clue::PointsDevice<1, Device> d_points(queue, nTracks);
        
        clue::Clusterer<1> algo(queue, m_dc, m_rhoc, m_dm);
        const std::size_t block_size{256};
        algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);

     
      }    
  } // namespace clueVertexFinder
} //namespace ALPAKA_ACCELERATOR_NAMESPACE
