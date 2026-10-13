// CPU-only SM90 device linker for iaas-kernels' FP4 RDMA JIT.
#include <nvJitLink.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void print_logs(nvJitLinkHandle handle) {
  size_t size = 0;
  if (nvJitLinkGetInfoLogSize(handle, &size) == NVJITLINK_SUCCESS && size) {
    std::vector<char> log(size);
    if (nvJitLinkGetInfoLog(handle, log.data()) == NVJITLINK_SUCCESS)
      std::cerr << log.data() << '\n';
  }
  if (nvJitLinkGetErrorLogSize(handle, &size) == NVJITLINK_SUCCESS && size) {
    std::vector<char> log(size);
    if (nvJitLinkGetErrorLog(handle, log.data()) == NVJITLINK_SUCCESS)
      std::cerr << log.data() << '\n';
  }
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    unsigned int major = 0, minor = 0;
    if (nvJitLinkVersion(&major, &minor) != NVJITLINK_SUCCESS)
      return 1;
    std::cout << "nvJitLink " << major << '.' << minor << '\n';
    return major == 13 && minor == 0 ? 0 : 1;
  }
  if (argc != 5 || std::string(argv[3]) != "archive") {
    std::cerr << "usage: link_object kernel.o libnvshmem_device_lto.a archive "
                 "output.cubin\n";
    return 64;
  }
  // Match the toolchain used by the FP4 RS performance/correctness regressions.
  const char *options[] = {"-arch=sm_90a", "-lto",
                           "-O3",          "-lineinfo",
                           "-verbose",     "-Xptxas=--register-usage-level=10"};
  nvJitLinkHandle handle = nullptr;
  if (nvJitLinkCreate(&handle, sizeof(options) / sizeof(options[0]), options) !=
      NVJITLINK_SUCCESS)
    return 1;
  auto checked = [&](nvJitLinkResult result) {
    if (result == NVJITLINK_SUCCESS)
      return true;
    print_logs(handle);
    nvJitLinkDestroy(&handle);
    return false;
  };
  // The JIT passes an ELF object containing LTO IR, not a raw .ltoir file.
  if (!checked(nvJitLinkAddFile(handle, NVJITLINK_INPUT_OBJECT, argv[1])))
    return 2;
  if (!checked(nvJitLinkAddFile(handle, NVJITLINK_INPUT_LIBRARY, argv[2])))
    return 3;
  if (!checked(nvJitLinkComplete(handle)))
    return 4;
  print_logs(handle);
  size_t size = 0;
  if (!checked(nvJitLinkGetLinkedCubinSize(handle, &size)))
    return 5;
  std::vector<char> cubin(size);
  if (!checked(nvJitLinkGetLinkedCubin(handle, cubin.data())))
    return 6;
  std::ofstream output(argv[4], std::ios::binary);
  output.write(cubin.data(), cubin.size());
  output.close();
  if (!output) {
    nvJitLinkDestroy(&handle);
    return 7;
  }
  return nvJitLinkDestroy(&handle) == NVJITLINK_SUCCESS ? 0 : 8;
}
