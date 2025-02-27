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

// #include "CLUEstering/CLUEstering.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CLUEVertexProducer : public global::EDProducer<> {
   using TkSoADevice = TracksSoACollection<pixelTopology::Phase1>;
   public: 
    CLUEVertexProducer(edm::ParameterSet const& conf)
	: verbose_(conf.getParameter<int>("Verbosity")),
      // 1.0 GeV
      ptMin_(conf.getParameter<double>("PtMin")),
      method2(conf.getParameter<bool>("Method2")),
      trackCollName(conf.getParameter<edm::InputTag>("TrackCollection")),
      token_Tracks(consumes(trackCollName)),
      //token_BeamSpot(consumes(conf.getParameter<edm::InputTag>("beamSpot"))),
      token_RecoVertex(produces()) {
  // Register my product

  // Setup shop
  std::string finder = conf.getParameter<std::string>("Finder");  // DivisiveVertexFinder
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


  //descriptions.add("clueVertices", desc);


}
    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
	
      // edm::Handle<reco::BeamSpot> bsHandle;
      // const auto& bsHandle = event.get(token_BeamSpot);
      std::cout << "Pippo \n";
      // Putting empty vertex into the event as a first step
      auto vertexes = std::make_unique<reco::VertexCollection>();
      
      AlgebraicSymMatrix33 we;
      we(0,0) = 10000;
      we(1,1) = 10000;
      we(2,2) = 10000;

      // auto vertices = std::make_unique<ZVertexSoACollection>({{10,10}}, event.queue());
      
      ZVertexSoACollection vertices({{10,10}}, event.queue());

      event.emplace(token_RecoVertex, {{10,10}}, event.queue());
      /*std::vector<int> results(2 * n_points);
       
      const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);

      PointsSoA<2> h_points(coords.data(), results.data(), PointInfo<2>{n_points});
      PointsAlpaka<2> d_points(queue_, n_points);

      CLUEAlgoAlpaka<2> algo(m_dc, m_rhoc, m_dm, m_pPBin, event.queue());

      algo.make_clusters(h_points, d_points, FlatKernel{.5f}, event.queue());

      // Now I need to figure out how to convert the clusters into vertexes and then put them in the event
      // Look at PixelVertexProducer.cc
      
      auto vertexes = std::make_unique<reco::VertexCollection>();
      auto my_clusters = algo.getClusters(h_points); // returns std::map<int, std::vector<int>> vertex to track ids map
      auto seeds = h_points.isSeed(); // array of indexes of the seeds, there are my_clusters.size() seeds
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
	  int clusterSize = my_clusters[clusterIdx].size();

	  // Hard coding error, chi2 and ndof
	
	  double avgPos = std::accumulate(my_clusters[clusterIdx].brgin(), my_clusters[clusterIdx].end(), 0.0,
			  [&h_points.coords()](double acc, size_t i) { return acc + h_points.coords()[i + n_points * 2]; } );
	  avgPos /= clusterSize;

	  // Computing error in 1D(z): Semidispersione massima / sqrt(num of points in cluster)
	  auto minmax = std::minmax_element(my_clusters[clusterIdx].begin(), my_clusters[clusterIdx].end(),
			  [&h_points.coords()](size_t i, size_t j) { 
			    return h_points.coords()[i + n_points * 2] < h_points.coords()[j + n_points * 2]; });
	  double min = h_points.coords()[*minmax.first];
	  double max = h_points.coords()[*minmax.second];

	  double error = (max - min) / std::sqrt(clusterSize);

 	  // Computing chi2 of z coordinate	
	  

	  double chi2 = std::accumulate(my_clusters[clusterIdx].brgin(), my_clusters[clusterIdx].end(), 0.0,
		  	  [&h_points.coords()](double acc, size_t i) { 
		    	    return acc + (h_points.coords()[i + n_points * 2] - avgPos)*(h_points.coords()[i + n_points * 2] - avgPos)/avgPos; } );
	  // ***

	  int ndof = clusterSize - 1;
	 
	
	  reco::Vertex v(reco::Vertex::Point(x, y, z), 
	  		  error,
			  chi2, 
			  ndof, 
			  clusterSize); 
	  // Completely uncertain about what to put here, I know I need to add the whole cluster
	  // but I don't know what kind of object v.add() wants
	  //
	  // it's performing an "emplace_back" on a std::vector<TrackBaseRef> object.
	  // What is "emplace_back"?
	  //
	  // THIS IS FOR SURE WRONG (BUT THE INTENTION IS THERE) GOTTA FIRST FIGURE OUT HOW TO 
	  // CONVERT PointSoA into Vertex
	  //v.add( need to add points of cluster with seed s );
        }
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
  const device::EDGetToken<TkSoADevice> token_Tracks;
  // const device::EDGetToken<reco::BeamSpot> token_BeamSpot;
  const device::EDPutToken<ZVertexSoACollection> token_RecoVertex;
    // Parameters for CLUEAlgoAlpaka
    float m_dc{1.5f}; // Side length of box to calculate density
    float m_rhoc{10.f}; // Minimum energy density to NOT be an outlier 
    float m_dm{1.5f}; // Side length of box to search for followers
    int m_pPBin{128}; // Average number of points found in a tile
    bool m_wtAvg{true}; // Decides how to copute error
  };
}

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
  DEFINE_FWK_ALPAKA_MODULE(CLUEVertexProducer);
