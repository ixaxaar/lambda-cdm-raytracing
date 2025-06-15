# Contributing to Lambda-CDM Raytracing

We welcome contributions to the Lambda-CDM Universe Simulation project! This document provides guidelines for contributing.

## 🤝 How to Contribute

### Reporting Issues
1. Check existing issues first
2. Use the issue templates
3. Provide detailed reproduction steps
4. Include system information (GPU, CUDA version, etc.)

### Feature Requests
1. Describe the physics/computational need
2. Explain the expected behavior
3. Consider performance implications
4. Provide scientific references if applicable

### Code Contributions

#### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/lambda-cdm-raytracing.git
cd lambda-cdm-raytracing

# Create development branch
git checkout -b feature/your-feature-name

# Build in debug mode
mkdir build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON
make -j$(nproc)
```

#### Code Style
- **C++ Standard**: C++17
- **Style Guide**: Google C++ Style Guide
- **Formatting**: Use clang-format with provided `.clang-format`
- **Naming**: 
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_CASE`

#### Architecture Guidelines

##### Component Development
```cpp
// 1. Create interface-compliant component
class MyForceComputer : public IForceComputer {
private:
    std::string name_;
    // Component state
    
public:
    // Constructor with name
    explicit MyForceComputer(const std::string& name) : name_(name) {}
    
    // IComponent interface
    bool initialize(const SimulationContext& context) override;
    void finalize() override;
    std::string get_type() const override { return "MyForceComputer"; }
    std::string get_name() const override { return name_; }
    std::string get_version() const override { return "1.0.0"; }
    
    // IForceComputer interface
    void compute_forces(const float* positions, const float* masses,
                       float* forces, size_t num_particles,
                       const std::any& params = {}) override;
    
    bool supports_gpu() const override;
    bool supports_mpi() const override;
    size_t get_max_particles() const override;
};

// 2. Register in factory
ForceComputerFactory::register_force_computer<MyForceComputer>("MyForceComputer");
```

##### Configuration Schema
```cpp
// Add configuration validation
void validate_my_component_config(ConfigurationNode* node) {
    // Required parameters
    if (!node->has("required_param")) {
        throw std::runtime_error("MyComponent requires 'required_param'");
    }
    
    // Parameter validation
    auto value = node->get<double>("required_param");
    if (value <= 0.0) {
        throw std::runtime_error("required_param must be positive");
    }
}

// Register schema
config_manager.register_schema("MyComponent", validate_my_component_config);
```

#### Testing Requirements
- **Unit Tests**: Required for all new components
- **Integration Tests**: Required for physics algorithms
- **Performance Tests**: Required for compute-intensive features

```cpp
// Example unit test
TEST(MyForceComputerTest, InitializationTest) {
    auto computer = std::make_unique<MyForceComputer>("test");
    SimulationContext context;
    
    EXPECT_TRUE(computer->initialize(context));
    EXPECT_EQ(computer->get_name(), "test");
    EXPECT_EQ(computer->get_type(), "MyForceComputer");
}

// Example physics test
TEST(MyForceComputerTest, NewtonianGravityTest) {
    auto computer = std::make_unique<MyForceComputer>("test");
    
    // Test known configuration
    float positions[] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    float masses[] = {1.0f, 1.0f};
    float forces[6] = {0};
    
    computer->compute_forces(positions, masses, forces, 2);
    
    // Verify expected force magnitude
    float expected_force = 1.0f; // G*m1*m2/r^2
    EXPECT_NEAR(forces[0], expected_force, 1e-6);
}
```

#### Documentation
- **Header Comments**: Doxygen-style documentation
- **Algorithm Documentation**: Explain physics and computational approach
- **Performance Notes**: Document computational complexity

```cpp
/**
 * @brief Computes gravitational forces using the Barnes-Hut tree algorithm
 * 
 * This implementation uses an octree spatial decomposition to approximate
 * long-range forces, reducing computational complexity from O(N²) to O(N log N).
 * 
 * @param positions Array of particle positions [x,y,z,x,y,z,...]
 * @param masses Array of particle masses
 * @param forces Output array for computed forces [fx,fy,fz,...]
 * @param num_particles Number of particles
 * @param params Optional parameters (opening angle, etc.)
 * 
 * @complexity O(N log N) average case, O(N²) worst case
 * @memory O(N) for tree storage
 * 
 * @see Barnes & Hut (1986) "A hierarchical O(N log N) force-calculation algorithm"
 */
void TreeForceComputer::compute_forces(const float* positions, const float* masses,
                                      float* forces, size_t num_particles,
                                      const std::any& params) override;
```

#### Performance Considerations
- **GPU Memory**: Minimize transfers, use pinned memory
- **CUDA Kernels**: Optimize occupancy and memory coalescing
- **Threading**: Use thread-safe containers and synchronization
- **Memory Pools**: Prefer pool allocation over frequent malloc/free

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/descriptive-name
   ```

2. **Development Workflow**
   ```bash
   # Make changes
   git add .
   git commit -m "feat: Add Barnes-Hut tree force computer
   
   - Implement octree spatial decomposition
   - Add GPU acceleration with CUDA kernels
   - Include comprehensive unit tests
   - Document algorithm complexity and usage"
   ```

3. **Pre-submission Checklist**
   - [ ] Code follows style guidelines
   - [ ] All tests pass (`ctest`)
   - [ ] Documentation updated
   - [ ] Performance benchmarks (if applicable)
   - [ ] No memory leaks (`valgrind` or similar)
   - [ ] CUDA code tested on target GPUs

4. **Submit Pull Request**
   - Use descriptive title
   - Reference related issues
   - Include performance impact
   - Add reviewer suggestions

### Commit Message Format
```
type(scope): Brief description

Detailed explanation of changes, motivation, and impact.
Include references to issues, papers, or algorithms.

Fixes #123
Closes #456
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`

## 🔬 Physics Contributions

### Algorithm Implementation
- Provide scientific references
- Include accuracy validation tests
- Document parameter ranges and limitations
- Consider numerical stability

### Cosmology Models
- Follow standard cosmological parameter conventions
- Include unit tests against known analytical solutions
- Document assumptions and approximations

### Force Computation
- Benchmark against reference implementations
- Include convergence tests
- Document softening and resolution effects

## 🚀 Performance Optimization

### GPU Kernels
- Profile with `nvprof` or Nsight
- Optimize memory bandwidth utilization
- Consider shared memory usage patterns
- Document occupancy and performance metrics

### TensorRT Integration
- Provide ONNX model definitions
- Include accuracy validation against reference
- Document precision trade-offs (FP32 vs FP16)

## 🐛 Bug Reports

### Information to Include
- System configuration (OS, GPU, CUDA version)
- Compilation flags and CMake configuration
- Minimal reproducible example
- Expected vs actual behavior
- Error messages and stack traces

### Performance Issues
- Include profiling data
- Specify hardware configuration
- Provide timing comparisons
- Include memory usage statistics

## 📋 Code Review Process

### Review Criteria
- **Correctness**: Algorithm implementation
- **Performance**: Computational efficiency
- **Safety**: Memory management and error handling
- **Maintainability**: Code clarity and documentation
- **Testing**: Adequate test coverage

### Review Timeline
- Initial review within 48 hours
- Feedback incorporation and re-review
- Final approval from maintainers

## 🏷️ Release Process

### Version Numbering
- Semantic versioning (MAJOR.MINOR.PATCH)
- Major: Breaking API changes
- Minor: New features, backward compatible
- Patch: Bug fixes

### Release Checklist
- [ ] All tests pass on target platforms
- [ ] Performance benchmarks meet requirements
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version numbers incremented

## 📞 Getting Help

- **Discussions**: GitHub Discussions for questions
- **Issues**: Bug reports and feature requests
- **Wiki**: Detailed documentation and tutorials
- **Examples**: Reference implementations

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Academic publications (for significant contributions)

Thank you for contributing to advancing cosmological simulation capabilities!