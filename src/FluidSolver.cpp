#include "FluidSolver.hpp"
#include <tracy/Tracy.hpp>
#include <cmath>
#include <algorithm>

void Advect(int width, int height, float dt,
            float* __restrict dest,
            const float* __restrict src,
            const float* __restrict u,
            const float* __restrict v) {
  ZoneScoped;
  float dt0 = dt * std::max(height, width);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;

      // back-tracing: one timestep ago, which bit of fluid was headed towards me?
      // sooo we're trying to find the src pos.
      float px = x - dt0 * u[idx];
      float py = y - dt0 * v[idx];

      // boundary-clamping
      if (px < 0.5f) px = 0.5f;
      if (py < 0.5f) py = 0.5f;
      
      if (px > width - 1.5f) px = width - 1.5f;
      if (py > height - 1.5f) py = height - 1.5f;

      int i0 = (int)px;
      int i1 = i0 + 1;
      
      int j0 = (int)py;
      int j1 = j0 + 1;

      // weigted avg: src pos. might be closer to some neighbors than others
      float s1 = px - i0;
      float t1 = py - j0;

      float s0 = 1.0f - s1;
      float t0 = 1.0f - t1;

      // positioning: (right/left) averages, then combine and avg.
      dest[idx] = s0 * (t0 * src[j0 * width + i0] + t1 * src[j1 * width + i0]) + // left column
                  s1 * (t0 * src[j0 * width + i1] + t1 * src[j1 * width + i1]);  // right column
    }
  }
}

void Project(int width, int height, float* u, float* v, float* p, float* div) {
  ZoneScopedN("Project Iteration");
  float h = 1.0f / std::max(width, height);
  
  // div * h^2.
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      int i = y * width + x;
      div[i] = -0.5f * h * (u[i + 1] - u[i - 1] + v[i + width] - v[i - width]);
      p[i] = 0;
    }
  }

  // pressure, spread the error
  for (int iter = 0; iter < 20; iter++) {
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x += 8) {
        int i = y * width + x;

        __m256 v_div = _mm256_loadu_ps(&div[i]);
        __m256 v_pR = _mm256_loadu_ps(&p[i + 1]);
        __m256 v_pL = _mm256_loadu_ps(&p[i - 1]);
        __m256 v_pU = _mm256_loadu_ps(&p[i + width]);
        __m256 v_pD = _mm256_loadu_ps(&p[i - width]);

        __m256 v_sum = _mm256_add_ps(v_div, v_pR);
        v_sum = _mm256_add_ps(v_sum, v_pL);
        v_sum = _mm256_add_ps(v_sum, v_pU);
        v_sum = _mm256_add_ps(v_sum, v_pD);

        __m256 v_res = _mm256_mul_ps(v_sum, _mm256_set1_ps(0.25f));
        _mm256_storeu_ps(&p[i], v_res);

        // p[i] = (div[i] + p[i + 1] + p[i - 1] + p[i + width] + p[i - width]) / 4.0f;
      }
    }
  }

  // fix the pressure.
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      int i = y * width + x;
      u[i] -= 0.5f * (p[i + 1] - p[i - 1]) / h; 
      v[i] -= 0.5f * (p[i + width] - p[i - width]) / h;
    }
  }
}
