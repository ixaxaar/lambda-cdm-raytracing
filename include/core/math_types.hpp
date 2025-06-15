#pragma once

#include <cmath>

// Math types for CPU compatibility

struct float3 {
    float x, y, z;
    
    float3() : x(0.0f), y(0.0f), z(0.0f) {}
    float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    float3 operator+(const float3& other) const {
        return float3(x + other.x, y + other.y, z + other.z);
    }
    
    float3 operator-(const float3& other) const {
        return float3(x - other.x, y - other.y, z - other.z);
    }
    
    float3 operator*(float s) const {
        return float3(x * s, y * s, z * s);
    }
    
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    float length_squared() const {
        return x * x + y * y + z * z;
    }
};

struct double3 {
    double x, y, z;
    
    double3() : x(0.0), y(0.0), z(0.0) {}
    double3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    double3 operator+(const double3& other) const {
        return double3(x + other.x, y + other.y, z + other.z);
    }
    
    double3 operator-(const double3& other) const {
        return double3(x - other.x, y - other.y, z - other.z);
    }
    
    double3 operator*(double s) const {
        return double3(x * s, y * s, z * s);
    }
    
    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    double length_squared() const {
        return x * x + y * y + z * z;
    }
};

// Utility functions
inline float3 make_float3(float x, float y, float z) {
    return float3(x, y, z);
}

inline double3 make_double3(double x, double y, double z) {
    return double3(x, y, z);
}

inline float length(const float3& v) {
    return v.length();
}

inline double length(const double3& v) {
    return v.length();
}