#!/bin/bash

# Development Environment Setup Script for Lambda-CDM Raytracing

set -e

echo "🚀 Setting up Lambda-CDM Raytracing development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            OS_FAMILY="debian"
            print_success "Detected Debian/Ubuntu Linux"
        elif command -v yum &> /dev/null; then
            OS_FAMILY="rhel"
            print_success "Detected RHEL/CentOS Linux"
        else
            print_error "Unsupported Linux distribution"
            exit 1
        fi
    else
        print_error "Unsupported operating system: $OSTYPE"
        print_error "This script supports Linux only"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if [[ "$OS_FAMILY" == "debian" ]]; then
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            ninja-build \
            git \
            wget \
            curl \
            pkg-config \
            clang-format \
            clang-tidy \
            valgrind \
            doxygen \
            graphviz
    elif [[ "$OS_FAMILY" == "rhel" ]]; then
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            cmake3 \
            ninja-build \
            git \
            wget \
            curl \
            pkgconfig \
            clang \
            valgrind \
            doxygen \
            graphviz
    fi
    
    print_success "System dependencies installed"
}

# Check and install CUDA
install_cuda() {
    print_status "Checking CUDA installation..."
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        print_success "CUDA $CUDA_VERSION is already installed"
        return
    fi
    
    print_warning "CUDA not found. Please install CUDA 11.0 or later manually:"
    echo "  - Download from: https://developer.nvidia.com/cuda-downloads"
    echo "  - Follow installation instructions for your OS"
    echo "  - Ensure nvcc is in your PATH"
    
    read -p "Continue without CUDA? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# Install MPI
install_mpi() {
    print_status "Installing MPI..."
    
    if [[ "$OS_FAMILY" == "debian" ]]; then
        sudo apt-get install -y libopenmpi-dev openmpi-bin
    elif [[ "$OS_FAMILY" == "rhel" ]]; then
        sudo yum install -y openmpi-devel
        echo 'export PATH=$PATH:/usr/lib64/openmpi/bin' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/openmpi/lib' >> ~/.bashrc
    fi
    
    print_success "MPI installed"
}

# Install HDF5
install_hdf5() {
    print_status "Installing HDF5..."
    
    if [[ "$OS_FAMILY" == "debian" ]]; then
        sudo apt-get install -y libhdf5-dev
    elif [[ "$OS_FAMILY" == "rhel" ]]; then
        sudo yum install -y hdf5-devel
    fi
    
    print_success "HDF5 installed"
}

# Install TensorRT (placeholder - requires manual download)
install_tensorrt() {
    print_status "Checking TensorRT installation..."
    
    if pkg-config --exists tensorrt; then
        print_success "TensorRT is already installed"
        return
    fi
    
    print_warning "TensorRT not found. Manual installation required:"
    echo "  1. Download TensorRT from NVIDIA Developer website"
    echo "  2. Extract and follow installation instructions"
    echo "  3. Ensure libraries are in your LD_LIBRARY_PATH"
    
    read -p "Continue without TensorRT? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# Install development tools
install_dev_tools() {
    print_status "Installing development tools..."
    
    # Install Google Test
    if [[ "$OS_FAMILY" == "debian" ]]; then
        sudo apt-get install -y libgtest-dev libgmock-dev
        
        # Build and install Google Test if not available as package
        if [[ ! -f /usr/lib/libgtest.a ]]; then
            cd /usr/src/gtest
            sudo cmake . -DBUILD_SHARED_LIBS=ON
            sudo make
            sudo cp lib/*.so /usr/lib/
            sudo cp -r ../googletest/googlemock/include/gmock /usr/include/
            cd -
        fi
    elif [[ "$OS_FAMILY" == "rhel" ]]; then
        # Install from source on RHEL
        git clone https://github.com/google/googletest.git /tmp/googletest
        cd /tmp/googletest
        mkdir build && cd build
        cmake .. -DBUILD_SHARED_LIBS=ON
        make -j$(nproc)
        sudo make install
        cd -
        rm -rf /tmp/googletest
    fi
    
    print_success "Development tools installed"
}

# Set up Git hooks
setup_git_hooks() {
    print_status "Setting up Git hooks..."
    
    # Pre-commit hook for formatting
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Auto-format code before commit
find include src examples -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" | \
    xargs clang-format -i

# Add formatted files back to staging
git add -u
EOF
    
    chmod +x .git/hooks/pre-commit
    
    print_success "Git hooks configured"
}

# Create development directories
setup_directories() {
    print_status "Creating development directories..."
    
    mkdir -p {build,build-debug,output,logs,models,data}
    
    print_success "Development directories created"
}

# Configure environment
setup_environment() {
    print_status "Setting up environment variables..."
    
    # Create environment setup script
    cat > setup_env.sh << 'EOF'
#!/bin/bash
# Lambda-CDM Raytracing Development Environment

# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# TensorRT (adjust path as needed)
export TENSORRT_ROOT=/opt/tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH

# Development settings
export CMAKE_BUILD_TYPE=Debug
export ENABLE_TESTING=ON
export ENABLE_BENCHMARKS=ON

echo "🚀 Lambda-CDM development environment loaded"
EOF
    
    chmod +x setup_env.sh
    
    print_success "Environment configuration created"
    print_status "Run 'source setup_env.sh' to load environment variables"
}

# Build project
build_project() {
    print_status "Building project..."
    
    # Debug build
    cmake -B build-debug \
        -DCMAKE_BUILD_TYPE=Debug \
        -DENABLE_TESTING=ON \
        -DENABLE_BENCHMARKS=ON \
        -G Ninja
    
    cmake --build build-debug --parallel $(nproc)
    
    # Release build
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_TESTING=ON \
        -DENABLE_BENCHMARKS=ON \
        -G Ninja
    
    cmake --build build --parallel $(nproc)
    
    print_success "Project built successfully"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    cd build-debug
    if ctest --output-on-failure; then
        print_success "All tests passed"
    else
        print_warning "Some tests failed - check output above"
    fi
    cd ..
}

# Main setup function
main() {
    echo "=========================================="
    echo "Lambda-CDM Raytracing Development Setup"
    echo "=========================================="
    
    check_os
    install_system_deps
    install_mpi
    install_hdf5
    install_dev_tools
    install_cuda
    install_tensorrt
    setup_directories
    setup_git_hooks
    setup_environment
    
    print_status "Setup complete! 🎉"
    echo ""
    echo "Next steps:"
    echo "  1. source setup_env.sh"
    echo "  2. cmake -B build -DCMAKE_BUILD_TYPE=Release"
    echo "  3. cmake --build build --parallel \$(nproc)"
    echo "  4. cd build && ctest"
    echo ""
    echo "For development:"
    echo "  - Use build-debug/ for debugging"
    echo "  - Use build/ for release builds"
    echo "  - Git hooks will auto-format code"
    echo "  - See CONTRIBUTING.md for guidelines"
}

# Run setup
main "$@"