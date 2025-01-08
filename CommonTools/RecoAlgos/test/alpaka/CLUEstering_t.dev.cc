#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>

class TestCLUEstering : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestCLUEstering);
  CPPUNIT_TEST(testRun);
  CPPUNIT_TEST_SUITE_END();

public:
  void testRun();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestCLUEstering);

#include "../CLUEstering/CLUEstering/alpaka/CLUE/CLUEAlgoAlpaka.h"
#include "../CLUEstering/CLUEstering/alpaka/DataFormats/Points.h"
#include "../CLUEstering/CLUEstering/alpaka/DataFormats/alpaka/PointsAlpaka.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

void read(std::vector<std::vector<float>>& coords, const std::string& filename) {
  std::fstream file{"/afs/cern.ch/user/s/sbalducc/CMSSW_14_2_0_pre1/src/CommonTools/RecoAlgos/test/alpaka/sissa.csv"};
  assert(file.is_open());

  std::string buffer;
  getline(file, buffer);
  int pid = 0;
  while (getline(file, buffer)) {
    std::stringstream buffer_stream(buffer);
    std::string value;

    for (size_t i{}; i < 2; ++i) {
      getline(buffer_stream, value, ',');
      coords[pid][i] = std::stof(value);
    }
    getline(buffer_stream, value);
    ++pid;
  }

  file.close();
}

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace fs = std::filesystem;

void TestCLUEstering::testRun() {
  auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(device);
  const float dc = 20.f, rhoc = 10.f, dm = 20.f;
  const int ppbin = 128;
  CLUEAlgoAlpaka<2> algo(dc, rhoc, dm, ppbin, queue);

  const auto n = 999;
  std::vector<std::vector<float>> coords(n, std::vector<float>(2));
  read(coords, "./sissa.csv");
  std::vector<float> weights(n);
  std::fill(weights.begin(), weights.end(), 1.f);
  Points<2> h_points(coords, weights);
  PointsAlpaka<2> d_points(queue, n);

  auto result = algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, n);

  std::vector<int> truth;
  std::fstream resfile{"/afs/cern.ch/user/s/sbalducc/CMSSW_14_2_0_pre1/src/CommonTools/RecoAlgos/test/alpaka/sissa_truth.csv"};
  assert(resfile.is_open());
  std::string value;
  while (getline(resfile, value)) {
    truth.push_back(std::stoi(value));
  }
  resfile.close();

  CPPUNIT_ASSERT(std::equal(truth.begin(), truth.end(), result[0].begin()));
}
