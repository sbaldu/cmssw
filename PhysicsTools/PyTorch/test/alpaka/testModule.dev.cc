#include <alpaka/core/Common.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

//namespace ALPAKA_ACCELERATOR_NAMESPACE {
//
//template <typename TData> struct VecAdd {
//  template <typename TAcc>
//  void operator()(const TAcc &acc, const TData *a, const TData *b, TData *c,
//                  size_t n_points) const {
//    for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
//      c[i] = a[i] + b[i];
//    }
//  }
//};
//
//}; // namespace ALPAKA_ACCELERATOR_NAMESPACE

using namespace ALPAKA_ACCELERATOR_NAMESPACE;


void testModule() {
  const auto device = alpaka::getDevByIdx<Acc1D>(0u);
  Queue queue{device};


  const size_t n_points{1024};

  auto host_a = cms::alpakatool::make_host_buffer<float[]>(queue, n_points);
  auto host_b = cms::alpakatool::make_host_buffer<float[]>(queue, n_points);
  auto host_c = cms::alpakatool::make_host_buffer<float[]>(queue, n_points);

  for (size_t i{}; i < n_points; ++i) {
    host_a[i] = i;
    host_b[i] = i;
  }

  auto dev_a = cms::alpakatools::make_device_buffer<float[]>(queue, n_points);
  auto dev_b = cms::alpakatools::make_device_buffer<float[]>(queue, n_points);
  auto dev_c = cms::alpakatools::make_device_buffer<float[]>(queue, n_points);

  alpaka::memcpy(queue, dev_a, host_a);
  alpaka::memcpy(queue, dev_b, host_b);
  alpaka::wait(queue);

  std::string module_path = dataPath_ + "/vecAdd_model.pt";
  // initialize model
  torch::jit::script::Module model;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
# define TORCH_ARCH torch::kCUDA
#elif define ALPAKA_ACC_CPU_B_SEQ_T_ENABLED
# define TORCH_ARCH torch::kCPU
#else
# error "The backend is not supported"
#endif

  torch::Device device{TORCH_ARCH};
  model = torch::jit::load(model_path);
  model.to(device)

  try {
    // Convert pinned memory on GPU to Torch tensor on GPU
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    auto options = torch::TensorOptions().dtype(torch::kInt).device(TORCH_ARCH, 0).pinned_memory(true);
#elif ALPAKA_ACC_CPU_B_SEQ_T_ENABLED
    auto options = torch::TensorOptions().dtype(torch::kInt).device(TORCH_ARCH, 0);
#endif
    cout << "Converting vectors and result to Torch tensors on Device" << endl;
    torch::Tensor a_gpu_tensor = torch::from_blob(dev_a.data(), {N}, options);
    torch::Tensor b_gpu_tensor = torch::from_blob(dev_b.data(), {N}, options);
    torch::Tensor c_gpu_tensor = torch::from_blob(dev_c.data(), {N}, options);

    cout << "Verifying result using Torch tensors" << endl;
    std::vector<torch::jit::IValue> inputs{a_gpu_tensor, b_gpu_tensor};
    // Not fully understood but std::move() is needed
    // https://stackoverflow.com/questions/71790378/assign-memory-blob-to-py-torch-output-tensor-c-api 
    std::move(c_gpu_tensor) = model.forward(inputs).toTensor();

    //CPPUNIT_ASSERT(c_gpu_tensor.equal(output));
  } catch (exception& e) {
    cout << e.what() << endl;
    CPPUNIT_ASSERT(false);
  }

  // Copy memory to device and also synchronize (implicitly)
  cout << "Synchronizing CPU and GPU. Copying result from GPU to CPU" << endl;
  alpaka::memcpy(queue, host_c, dev_c);

  // Verify the result on the CPU
  for (size_t i{}; i < n_points; ++i) {
    REQUIRE(host_c[i] == host_a[i] + host_b[i]);
  }
}

TEST_CASE("Test alpaka module for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  testModel();
}
