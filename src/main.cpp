#include <iostream>
#include "config.h"

int main() {
    std::cout << "Lambda-CDM Universe Simulation" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // Feature availability
    std::cout << "Available features:" << std::endl;
    
#ifdef HAVE_CUDA
    std::cout << "  ✓ CUDA GPU acceleration" << std::endl;
#else
    std::cout << "  ✗ CUDA GPU acceleration (not available)" << std::endl;
#endif

#ifdef HAVE_MPI
    std::cout << "  ✓ MPI cluster support" << std::endl;
#else
    std::cout << "  ✗ MPI cluster support (not available)" << std::endl;
#endif

#ifdef HAVE_HDF5
    std::cout << "  ✓ HDF5 data I/O" << std::endl;
#else
    std::cout << "  ✗ HDF5 data I/O (not available)" << std::endl;
#endif

#ifdef HAVE_TENSORRT
    std::cout << "  ✓ TensorRT optimization" << std::endl;
#else
    std::cout << "  ✗ TensorRT optimization (not available)" << std::endl;
#endif

    std::cout << std::endl;
    std::cout << "Framework initialized successfully!" << std::endl;
    std::cout << "Use examples/ for complete simulation examples." << std::endl;
    
    return 0;
}