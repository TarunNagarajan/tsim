#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include "FluidSolver.hpp"

const int GRID_WIDTH = 256;
const int GRID_HEIGHT = 256;
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;

// global mouse state. 
double lastMouseX = 0, lastMouseY = 0;

int main() {
    // glfw non-init
    if (!glfwInit()) {
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "tsim", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // failed to init glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "failed to init glad\n";
        return -1;
    }

    FluidGrid grid(GRID_WIDTH, GRID_HEIGHT);

    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; i++) {
      grid.density[i] = 0.0f;
      grid.u[i] = 0.0f;
      grid.v[i] = 0.0f;
    }

    for (int y = 0; y < GRID_HEIGHT; y++) {
      for (int x = 0; x < GRID_WIDTH; x++) {
        int i = y * width + x;

        // first density blob that's moving left to right
        float dx1 = float(x) - GRID_WIDTH * 0.25f; // left wrt origin
        float dy1 = float(y) - GRID_HEIGHT * 0.5; // center wrt origin
        
        if (dx1 * dx1 + dy1 * dy1 < 25 * 25) {
          grid.density[i] = 1.0f;
          grid.u[i] = 0.6f; // positive right direction.
        }

        // second density blob that's moving right to left
        float dx2 = float(x) - GRID_WIDTH * 0.75f; // right wrt origin 
        float dy2 = float(y) - GRID_HEIGHT * 0.5; // center wrt origin 
        
        if (dx2 * dx2 + dy2 * dy2 < 25 * 25) {
          grid.density[i] = 1.0f;
          grid.u[i] = -0.6f; // negative left direction.
        }

      }
    }

    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glTextureStorage2D(texture, 1, GL_R32F, GRID_WIDTH, GRID_HEIGHT);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    float lastTime = (float)glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        float currentTime = (float)glfwGetTime();
        float dt = currentTime - lastTime;
        lastTime = currentTime;

        if (dt > 0.1f) dt = 0.1f;

        // --- SIMULATION STEP ---
        
        // 1. self-advection: velocity moves itself
        Advect(GRID_WIDTH, GRID_HEIGHT, dt, grid.u_prev.data(), grid.u.data(), grid.u.data(), grid.v.data());
        Advect(GRID_WIDTH, GRID_HEIGHT, dt, grid.v_prev.data(), grid.v.data(), grid.u.data(), grid.v.data());
        std::swap(grid.u, grid.u_prev);
        std::swap(grid.v, grid.v_prev);

        // 2. projection: make the fluid swirl and stay incompressible
        Project(GRID_WIDTH, GRID_HEIGHT, grid.u.data(), grid.v.data(), grid.p.data(), grid.div.data());

        // 3. dye-advection: transport the density using the new velocity
        Advect(GRID_WIDTH, GRID_HEIGHT, dt, grid.density_prev.data(), grid.density.data(), grid.u.data(), grid.v.data());
        std::swap(grid.density, grid.density_prev);

        glTextureSubImage2D(texture, 0, 0, 0, GRID_WIDTH, GRID_HEIGHT, GL_RED, GL_FLOAT, grid.density.data());

        // clear screen to dark grey.
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        GLuint fbo;
        glCreateFramebuffers(1, &fbo);
        glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, texture, 0);
        
        glBlitNamedFramebuffer(fbo, 0, 
            0, 0, GRID_WIDTH, GRID_HEIGHT, 
            0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 
            GL_COLOR_BUFFER_BIT, GL_LINEAR);
        
        glDeleteFramebuffers(1, &fbo);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
