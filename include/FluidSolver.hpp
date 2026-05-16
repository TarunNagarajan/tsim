#pragma once
#include <vector>
#include <algorithm>
#include <tracy/Tracy.hpp>

struct FluidGrid {
  int width, height;

  // previous densities, densities
  std::vector<float> density;
  std::vector<float> density_prev;

  // pressure, divergence
  std::vector<float> p;
  std::vector<float> div;

  // velocity, x direction
  std::vector<float> u;
  std::vector<float> u_prev;

  // velocity, y direction
  std::vector<float> v;
  std::vector<float> v_prev;

  FluidGrid(int w, int h) : width(w), height(h) {
    size_t size = (size_t)w * h;

    density.resize(size, 0.0f);
    density_prev.resize(size, 0.0f);
    
    p.resize(size, 0.0f);
    div.resize(size, 0.0f);

    u.resize(size, 0.0f);
    u_prev.resize(size, 0.0f);
    
    v.resize(size, 0.0f);
    v_prev.resize(size, 0.0f);
  }
};

void Advect(int width, int height, float dt,
            float* __restrict dest,
            const float* __restrict src, 
            const float* __restrict u, 
            const float* __restrict v);

void Project(int width, int height, 
    float* u, 
    float* v, 
    float* p, 
    float* div);
