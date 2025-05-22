#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <vector>
#include <numeric>

#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/Error.h"

#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"

#include "./clueVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CLUEVertexProducer : public global::EDProducer<> {
     using TkSoADevice = TracksSoACollection<pixelTopology::Phase2>;
  public:
    CLUEVertexProducer(edm::ParameterSet const& conf)
        : EDProducer(conf),
          verbose_(conf.getParameter<int>("Verbosity")),
          // 1.0 GeV
          ptMin_(conf.getParameter<double>("PtMin")),
          method2(conf.getParameter<bool>("Method2")),
          trackCollName(conf.getParameter<edm::InputTag>("TrackCollection")),
          token_Tracks(consumes(trackCollName)),
          //token_BeamSpot(consumes(conf.getParameter<edm::InputTag>("beamSpot"))),
          token_RecoVertex(produces("CLUEVertex")) {
      // Register my product

      // Setup shop
      // std::string finder = conf.getParameter<std::string>("Finder");  // DivisiveVertexFinder
      bool useError = conf.getParameter<bool>("UseError");            // true
      bool wtAverage = conf.getParameter<bool>("WtAverage");          // true
      double zOffset = conf.getParameter<double>("ZOffset");          // 5.0 sigma
      double zSeparation = conf.getParameter<double>("ZSeparation");  // 0.05 cm
      int ntrkMin = conf.getParameter<int>("NTrkMin");                // 3
      // Tracking requirements before sending a track to be considered for vtx

      double track_pt_min = ptMin_;
      double track_pt_max = 10.;
      double track_chi2_max = 9999999.;
      double track_prob_min = -1.;
      std::cout << track_pt_min << " " << track_pt_max << " " << track_chi2_max << " " << track_prob_min << "\n";
      if (conf.exists("PVcomparer")) {
        edm::ParameterSet PVcomparerPSet = conf.getParameter<edm::ParameterSet>("PVcomparer");
        track_pt_min = PVcomparerPSet.getParameter<double>("track_pt_min");
        track_pt_max = PVcomparerPSet.getParameter<double>("track_pt_max");
        track_chi2_max = PVcomparerPSet.getParameter<double>("track_chi2_max");
        track_prob_min = PVcomparerPSet.getParameter<double>("track_prob_min");
      }
    }
    ~CLUEVertexProducer() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("Verbosity", 0);
      desc.add<double>("PtMin", 1.0);
      desc.add<bool>("Method2", true);
      desc.add<edm::InputTag>("TrackCollection", edm::InputTag("pixelTracks"));
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
      desc.add<std::string>("Finder", "DivisiveVertexFinder");
      desc.add<bool>("UseError", true);
      desc.add<bool>("WtAverage", true);
      desc.add<double>("ZOffset", 5.0);
      desc.add<double>("ZSeparation", 0.05);
      desc.add<int>("NTrkMin", 2);
      {
        edm::ParameterSetDescription PVComparerPSet;

        PVComparerPSet.add<double>("track_pt_min", 1.0);

        PVComparerPSet.add<double>("track_pt_max", 10.0);

        PVComparerPSet.add<double>("track_chi2_max", 999999.);

        PVComparerPSet.add<double>("track_prob_min", -1.);

        desc.addOptional<edm::ParameterSetDescription>("PVcomparer", PVComparerPSet);
      }

      // check label

      descriptions.addWithDefaultLabel(desc);

      //descriptions.add("CLUEVertex", desc);
    }
    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      auto const& tracks_d = event.get(token_Tracks);
      /* It's layout is:
             SOA_COLUMN(Quality, quality),
             SOA_COLUMN(float, chi2),
             SOA_COLUMN(int8_t, nLayers),
             SOA_COLUMN(float, eta),
             SOA_COLUMN(float, pt),
             // state at the beam spot: {phi, tip, 1/pt, cotan(theta), zip}
             SOA_EIGEN_COLUMN(Vector5f, state),
             SOA_EIGEN_COLUMN(Vector15f, covariance),
             SOA_SCALAR(int, nTracks),
             SOA_SCALAR(HitContainer, hitIndices),
             SOA_SCALAR(HitContainer, detIndices)
      */
      // const auto& bsHandle = event.get(token_BeamSpot);

      auto const& tracks_d_view = tracks_d.view();
      std::cout << __LINE__ << std::endl;
      Queue queue = event.queue();
      int maxVertices = 10;
      std::cout << __LINE__ << std::endl;
      const auto maxTracks = tracks_d_view.metadata().size();
      std::cout << "maxTracks = " << maxTracks << std::endl;
      std::cout << __LINE__ << std::endl;
      //const uint32_t nTracks; // SEG FAULT HERE; cannot access nTracks since 
                                                           // it's data allocated in device 
                                                           // will need to use this in the kernel
      /*TkSoAHost host_tracks; 
      alpaka::memcpy(queue, host_tracks, tracks_d);*/
      
      TracksHost<pixelTopology::Phase2> tracks_h(queue);  // in the host
      alpaka::memcpy(queue, tracks_h.buffer(), tracks_d.buffer());
      alpaka::wait(queue);
      std::cout << __LINE__ << std::endl;
      const uint32_t nTracks = tracks_h.view().nTracks();
      
      std::cout << "nTracks = " << nTracks << std::endl;
      //const uint32_t nTracks = maxTracks;  // placeholder
      std::cout << __LINE__ << std::endl;

      ZVertexSoACollection vertices({{maxVertices, maxTracks}}, queue); // this object is in the device
      std::cout << __LINE__ << std::endl;
      auto data = vertices.view();
      auto trkdata = vertices.view<reco::ZVertexTracksSoA>();  // access the data in the ZVertexTracksSoA Layout
      auto vrtxdata = vertices.view<reco::ZVertexSoA>();       // access the data in the ZVertexSoA Layout
      std::cout << __LINE__ << std::endl;

      std::vector<float> coords;
      std::vector<int> results(2*nTracks);
      
      for (auto idx = 0u; idx < nTracks; ++idx) {
        coords.push_back(reco::zip(tracks_h.view(), idx));
        if(idx < 10)
           std::cout << "coords[" << idx << "] = " << coords[idx] << std::endl;
      }
      for (auto idx = 0u; idx < nTracks; ++idx) {
        coords.push_back((tracks_h.view().pt())[idx + nTracks]); // also need to save the pt's
        if(idx < 10)   
           std::cout << "coords[nTracks + " << idx << "] = " << coords[idx + nTracks] << std::endl;
      }
      std::cout << "coords.size() and results.size() = " << coords.size() << " " << results.size() << std::endl;
      std::cout << __LINE__ << std::endl;
      clueVertexFinder::Producer clusterer(m_dc, m_rhoc, m_dm, m_pPBin, m_wtAvg);
      clusterer.makeClusters(coords, results, queue);

      std::cout << __LINE__ << std::endl;
      auto myClusters = std::span<const int>{results.data(), nTracks};
      auto isSeed = std::span<const int>(results.data() + nTracks, nTracks);
        
      std::cout << "myClusters.size() = " << myClusters.size() << " and isSeed.size() = " << isSeed.size() << std::endl;

      std::cout << __LINE__ << std::endl;
      int nClusters = *(std::max_element(myClusters.begin(), myClusters.end())) + 1;
      std::cout << "nClusters = " << nClusters << std::endl;
      std::vector<int> clusterCount(nClusters);  // need this to calculate averages later
      std::cout << __LINE__ << std::endl;
      /* ZVertexSoACollection is made of a ZVertexSoA and a ZVertexTracksSoA
      // ZvertexSoA is made of:
      //               SOA_COLUMN(float, zv),          // output z-posistion of found vertices
                       SOA_COLUMN(float, wv),          // output weight (1/error^2) on the above
                       SOA_COLUMN(float, chi2),        // vertices chi2
                       SOA_COLUMN(float, ptv2),        // vertices pt^2
                       SOA_COLUMN(uint16_t, sortInd),  // sorted index (by pt2)  ascending
                       SOA_SCALAR(uint32_t, nvFinal))  // the number of vertices
         and ZVertexTraksSoA is made of:
         SOA_COLUMN(int16_t, idv),   // vertex index for each associated (original) track
                                     // (-1 == not associate)
         SOA_COLUMN(int32_t, ndof))  // vertices number of dof
      */

      // Let's start filling out "vertices" !!
      // To fill out columns and scalars I have to use memcopys, cause I am on the host
      /*auto zv_hbuff = cms::alpakatools::make_host_buffer<float[]>(queue, nClusters);
      for(int i = 0; i < nClusters; ++i) {
        zv_hbuff[i] = coords[i];
      }*/

      // I will need this to compute chi2 of each vertex
      // std::for_each(myClusters.begin(), myClusters.end(), [&clusterCount](int idx) { clusterCount[idx]++; }); // SEG FAULT HERE
      
      std::cout << __LINE__ << std::endl;
      
      // vrtxdata.nvFinal() = nClusters; // SEG fault here, understandable: I'm trying to modify data in the device from the host

      std::cout << __LINE__ << std::endl;
      // Preparing to launch kernels

      /*
      for (auto i = 0u; i < nTracks; ++i) { 
        if (isSeed[i]) {
          vrtxdata[myClusters[i]].zv() = coords[i];
          trkdata[i].ndof() = clusterCount[myClusters[i]] - 1;
        }
        trkdata[i].idv() = myClusters[i];
        zAcc[myClusters[i]] += coords[i];
      }

      event.emplace(token_RecoVertex, std::move(vertices));
      */

      /* 

      auto vertexes = std::make_unique<reco::VertexCollection>();
      auto myClusters = algo.getClusters(h_points); // returns std::map<int, std::vector<int>> vertex to track ids map
      auto seeds = h_points.isSeed(); // array of indexes of the seeds, there are myClusters.size() seeds
	*/
      // Need to put the clusters into "vertexes", use the seed of each cluster as the point, and then put the points of the
      // cluster in the track of each Vector
      //
      /* if (bsHandle.isValid()) {
        
	const reco::BeamSpot& bs = *bsHandle;

        for(auto s : seeds) {
          double z = h_points.coords()[ s - 1 + 2 * n_points]; // coords() returns a single vector and it is ordered like
	 						       // x0, x1, x2, ... , y0, y1, y2, ... , z0, z1, z2, ...	 
      	  
	  // not sure about this one, do we actually have the BeamSpot object?	  
	  double x = bs.x0() + bs.dxdz() * (z - bs.z0());
	  double y = bs.y0() + bs.dydz() * (z - bs.zo());

	  int clusterIdx = h_points.clusterIndexes()[s];
	  int clusterSize = myClusters[clusterIdx].size();

	  // Hard coding error, chi2 and ndof
	
	  double avgPos = std::accumulate(myClusters[clusterIdx].brgin(), myClusters[clusterIdx].end(), 0.0,
			  [&h_points.coords()](double acc, size_t i) { return acc + h_points.coords()[i + n_points * 2]; } );
	  avgPos /= clusterSize;

	  // Computing error in 1D(z): Semidispersione massima / sqrt(num of points in cluster)
	  auto minmax = std::minmax_element(myClusters[clusterIdx].begin(), myClusters[clusterIdx].end(),
			  [&h_points.coords()](size_t i, size_t j) { 
			    return h_points.coords()[i + n_points * 2] < h_points.coords()[j + n_points * 2]; });
	  double min = h_points.coords()[*minmax.first];
	  double max = h_points.coords()[*minmax.second];

	  double error = (max - min) / std::sqrt(clusterSize);

 	  // Computing chi2 of z coordinate	
	  

	  double chi2 = std::accumulate(myClusters[clusterIdx].brgin(), myClusters[clusterIdx].end(), 0.0,
		  	  [&h_points.coords()](double acc, size_t i) { 
		    	    return acc + (h_points.coords()[i + n_points * 2] - avgPos)*(h_points.coords()[i + n_points * 2] - avgPos)/avgPos; } );
	  // ***

	  int ndof = clusterSize - 1;
      }*/

      // event.put(std::move(vertexes));
    }

  private:
    // ----------member data ---------------------------
    // Turn on debug printing if verbose_ > 0
    const int verbose_;
    // Tracking cuts before sending tracks to vertex algo
    const double ptMin_;
    const bool method2;
    const edm::InputTag trackCollName;
    device::EDGetToken<TkSoADevice> token_Tracks;
    // const device::EDGetToken<reco::BeamSpot> token_BeamSpot;
    device::EDPutToken<ZVertexSoACollection> token_RecoVertex;
    // Parameters for CLUEAlgoAlpaka
    float m_dc{0.2f};    // Side length of box to calculate density
    float m_rhoc{10.f};  // Minimum energy density to NOT be an outlier
    float m_dm{0.75f};    // Side length of box to search for followers
    int m_pPBin{128};    // Average number of points found in a tile
    bool m_wtAvg{true};  // Decides how to copute error
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

    DEFINE_FWK_ALPAKA_MODULE(CLUEVertexProducer);
