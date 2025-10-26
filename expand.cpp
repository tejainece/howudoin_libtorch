#include <torch/torch.h>
#include <iostream>

template <typename T>
void pretty_print(const std::string &info, T &&data)
{
    std::cout << info << std::endl;
    std::cout << data << std::endl
              << std::endl;
}

int main()
{
    {
        torch::Tensor tensorA = torch::eye(3, at::device(at::DeviceType::CPU).memory_format(at::MemoryFormat::Contiguous));
        torch::Tensor tensorB = torch::eye(3, at::device(at::DeviceType::CPU).memory_format(at::MemoryFormat::Contiguous));
        torch::Tensor res = tensorA.add_(tensorB, 5);
        pretty_print("TensorA: ", tensorA);
        pretty_print("TensorB: ", tensorB);
        pretty_print("Res: ", res);
    }
}