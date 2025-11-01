#include <torch/torch.h>
#include <iostream>

template <typename T>
void pretty_print(const std::string& info, T&& data) {
  std::cout << info << std::endl;
  std::cout << data << std::endl << std::endl;
}

int main() {
  // Create an eye tensor
  int64_t size[] = {10, 10, 7};
  torch::Tensor tensor = torch::ones(size, at::device(at::kCPU));
  pretty_print("Ones tensor: ", tensor);
}