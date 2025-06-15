#pragma once

#include <cmath>

// Check if we're compiling with CUDA
#ifdef __CUDACC__
    #include <cuda_runtime.h>
    // CUDA already defines float3, float4, double3, etc.
#else
    // Define our own types for non-CUDA compilation
    struct float3 {
        float x, y, z;
        
        float3() : x(0.0f), y(0.0f), z(0.0f) {}
        float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
        
        // Vector operations
        float3 operator+(const float3& other) const {
            return float3(x + other.x, y + other.y, z + other.z);
        }
        
        float3 operator-(const float3& other) const {
            return float3(x - other.x, y - other.y, z - other.z);
        }
        
        float3 operator*(float scalar) const {
            return float3(x * scalar, y * scalar, z * scalar);
        }
        
        float dot(const float3& other) const {
            return x * other.x + y * other.y + z * other.z;
        }
        
        float length() const {
            return std::sqrt(x * x + y * y + z * z);
        }
        
        float length_squared() const {
            return x * x + y * y + z * z;
        }
    };

    struct float4 {
        float x, y, z, w;
        
        float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
        float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    };

    struct double3 {
        double x, y, z;
        
        double3() : x(0.0), y(0.0), z(0.0) {}
        double3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
        
        // Vector operations
        double3 operator+(const double3& other) const {
            return double3(x + other.x, y + other.y, z + other.z);
        }
        
        double3 operator-(const double3& other) const {
            return double3(x - other.x, y - other.y, z - other.z);
        }
        
        double3 operator*(double scalar) const {
            return double3(x * scalar, y * scalar, z * scalar);
        }
        
        double dot(const double3& other) const {
            return x * other.x + y * other.y + z * other.z;
        }
        
        double length() const {
            return std::sqrt(x * x + y * y + z * z);
        }
        
        double length_squared() const {
            return x * x + y * y + z * z;
        }
    };

    // Helper functions for non-CUDA compilation
    inline float3 make_float3(float x, float y, float z) {
        return float3(x, y, z);
    }

    inline float4 make_float4(float x, float y, float z, float w) {
        return float4(x, y, z, w);
    }

    inline double3 make_double3(double x, double y, double z) {
        return double3(x, y, z);
    }
#endif

// Additional helper functions that work with both CUDA and non-CUDA
namespace core {
    // Length functions
    inline float length(const float3& v) {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    
    inline double length(const double3& v) {
        return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    
    // Dot product
    inline float dot(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    
    inline double dot(const double3& a, const double3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    
    // Cross product
    inline float3 cross(const float3& a, const float3& b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }
    
    inline double3 cross(const double3& a, const double3& b) {
        return make_double3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }
}