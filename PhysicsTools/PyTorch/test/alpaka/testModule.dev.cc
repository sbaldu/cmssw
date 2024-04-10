#include <alpaka/core/Common.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include <cassert>
#include <exception>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>


#include "../testBase.h"

/*
namespace ALPAKA_ACCELERATOR_NAMESPACE {

template <typename TData> struct VecAdd {
  template <typename TAcc>
  void operator()(const TAcc &acc, const TData *a, const TData *b, TData *c,
                  size_t n_points) const {
    for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
      c[i] = a[i] + b[i];
    }
  }
};

}; // namespace ALPAKA_ACCELERATOR_NAMESPACE
*/

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED)  && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
/*
void testAlpakaModel::test() {
  CPPUNIT_ASSERT(false);
}
*/

int main() {
  std::cout << "Not available backend" << std::endl;
  return 1;
}
#else

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

//void testAlpakaModel::test() {
int main() {
  const auto alpaka_device = cms::alpakatools::devices<Platform>()[0];
  Queue queue{alpaka_device};

  std::string model_path = "./dnn_largeinput.pt";

  const size_t n_points{1024};

  std::vector<int> a(n_points), b(n_points), c(n_points);
  for (size_t i{}; i < n_points; ++i) {
    a[i] = i;
    b[i] = i;
  }

  auto cpu_options = torch::TensorOptions();
  torch::Tensor a_cpu_tensor = torch::from_blob(a.data(), {n_points}, cpu_options);
  torch::Tensor b_cpu_tensor = torch::from_blob(b.data(), {n_points}, cpu_options);
  torch::Tensor c_cpu_tensor = torch::from_blob(c.data(), {n_points}, cpu_options);

  torch::jit::script::Module cpu_model;

  torch::Device cpu_device{torch::kCPU};
  cpu_model = torch::jit::load(model_path);
  cpu_model.to(cpu_device);

  std::vector<torch::jit::IValue> cpu_inputs{a_cpu_tensor, b_cpu_tensor};
  // Not fully understood but std::move() is needed
  // https://stackoverflow.com/questions/71790378/assign-memory-blob-to-py-torch-output-tensor-c-api 
  std::move(c_cpu_tensor) = cpu_model.forward(cpu_inputs).toTensor();

  auto host_a = cms::alpakatools::make_host_buffer<int[]>(queue, n_points);
  auto host_b = cms::alpakatools::make_host_buffer<int[]>(queue, n_points);
  auto host_c = cms::alpakatools::make_host_buffer<int[]>(queue, n_points);

  for (size_t i{}; i < n_points; ++i) {
    host_a[i] = i;
    host_b[i] = i;
  }

  auto dev_a = cms::alpakatools::make_device_buffer<int[]>(queue, n_points);
  auto dev_b = cms::alpakatools::make_device_buffer<int[]>(queue, n_points);
  auto dev_c = cms::alpakatools::make_device_buffer<int[]>(queue, n_points);

  alpaka::memcpy(queue, dev_a, host_a);
  alpaka::memcpy(queue, dev_b, host_b);
  alpaka::wait(queue);

  std::string model_path = "./simple_dnn_largeinput.pt";
  // initialize model
  torch::jit::script::Module model;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
# define TORCH_ARCH torch::kCUDA
#elif defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
# define TORCH_ARCH torch::kCPU
#else
# warm "The backend is not supported"
#endif

  torch::Device device{TORCH_ARCH};
  model = torch::jit::load(model_path);
  model.to(device);

  try {
    // Convert pinned memory on GPU to Torch tensor on GPU
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    auto options = torch::TensorOptions().dtype(torch::kInt).device(TORCH_ARCH, 0).pinned_memory(true);
#elif defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    auto options = torch::TensorOptions();
#endif
    std::cout << "Converting vectors and result to Torch tensors on Device" << std::endl;
    torch::Tensor a_gpu_tensor = torch::from_blob(dev_a.data(), {n_points}, options);
    torch::Tensor b_gpu_tensor = torch::from_blob(dev_b.data(), {n_points}, options);
    torch::Tensor c_gpu_tensor = torch::from_blob(dev_c.data(), {n_points}, options);

    std::cout << "Verifying result using Torch tensors" << std::endl;
    std::vector<torch::jit::IValue> inputs{a_gpu_tensor, b_gpu_tensor};
    // Not fully understood but std::move() is needed
    // https://stackoverflow.com/questions/71790378/assign-memory-blob-to-py-torch-output-tensor-c-api 
    std::move(c_gpu_tensor) = model.forward(inputs).toTensor();

    //CPPUNIT_ASSERT(c_gpu_tensor.equal(output));
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
   // CPPUNIT_ASSERT(false);
  }

  // Copy memory to device and also synchronize (implicitly)
  std::cout << "Synchronizing CPU and GPU. Copying result from GPU to CPU" << std::endl;
  alpaka::memcpy(queue, host_c, dev_c);
  alpaka::wait(queue);

  // Verify the result on the CPU
  for (size_t i{}; i < n_points; ++i) {
    //CPPUNIT_ASSERT(host_c[i] == host_a[i] + host_b[i]);
    //assert(host_c[i] == host_a[i] + host_b[i]);
  }
  assert(memcmp(c.data(), host_c.data(), n_points) == 0);
  std::cout << "Correct!" << std::endl;
}
#endif
