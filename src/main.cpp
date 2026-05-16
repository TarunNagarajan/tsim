#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <tracy/Tracy.hpp>
#include "FluidSolver.hpp"

const int GRID_WIDTH = 256;
const int GRID_HEIGHT = 256;
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;

// simple shaders to handle the color
const char* vShaderSource = R"(
    #version 450 core
    void main() {
        // creates a full-screen quad using only 4 vertices
        vec2 vertices[4] = vec2[](vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));
        gl_Position = vec4(vertices[gl_VertexID], 0, 1);
    }
)";

const char* fShaderSource = R"(
    #version 450 core
    layout(binding = 0) uniform sampler2D tex;
    out vec4 color;
    void main() {
        // map pixels to 0.0-1.0 range and sample density
        float val = texture(tex, gl_FragCoord.xy / 800.0).r;
        // blue-ish water color (R=0, G=val*0.5, B=val)
        color = vec4(0.0, val * 0.5, val, 1.0);
    }
)";

int main() {
    // glfw non-init
    if (!glfwInit()) return -1;

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

    // compile shaders
    GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vShader, 1, &vShaderSource, nullptr);
    glCompileShader(vShader);

    GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fShader, 1, &fShaderSource, nullptr);
    glCompileShader(fShader);

    GLuint program = glCreateProgram();
    glAttachShader(program, vShader);
    glAttachShader(program, fShader);
    glLinkProgram(program);

    FluidGrid grid(GRID_WIDTH, GRID_HEIGHT);

    // experiment: colliding blobs
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; i++) {
        grid.density[i] = 0.0f;
        grid.u[i] = 0.0f;
        grid.v[i] = 0.0f;
    }

    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            int i = y * GRID_WIDTH + x;
            float dx1 = (float)x - GRID_WIDTH * 0.25f;
            float dy1 = (float)y - GRID_HEIGHT * 0.5f;
            if (dx1 * dx1 + dy1 * dy1 < 25 * 25) {
                grid.density[i] = 1.0f;
                grid.u[i] = 1.0f;
                grid.v[i] = -0.2f;
            }
            float dx2 = (float)x - GRID_WIDTH * 0.75f;
            float dy2 = (float)y - GRID_HEIGHT * 0.5f;
            if (dx2 * dx2 + dy2 * dy2 < 25 * 25) {
                grid.density[i] = 1.0f;
                grid.u[i] = -0.8f;
                grid.v[i] = 0.2f;
            }
        }
    }

    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glTextureStorage2D(texture, 1, GL_R32F, GRID_WIDTH, GRID_HEIGHT);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // need a dummy vao for core profile
    GLuint vao;
    glCreateVertexArrays(1, &vao);

    float lastTime = (float)glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        FrameMark;
        float currentTime = (float)glfwGetTime();
        float dt = currentTime - lastTime;
        lastTime = currentTime;
        if (dt > 0.1f) dt = 0.1f;

        // --- SIMULATION STEP ---
        Advect(GRID_WIDTH, GRID_HEIGHT, dt, grid.u_prev.data(), grid.u.data(), grid.u.data(), grid.v.data());
        Advect(GRID_WIDTH, GRID_HEIGHT, dt, grid.v_prev.data(), grid.v.data(), grid.u.data(), grid.v.data());
        std::swap(grid.u, grid.u_prev);
        std::swap(grid.v, grid.v_prev);

        Project(GRID_WIDTH, GRID_HEIGHT, grid.u.data(), grid.v.data(), grid.p.data(), grid.div.data());

        Advect(GRID_WIDTH, GRID_HEIGHT, dt, grid.density_prev.data(), grid.density.data(), grid.u.data(), grid.v.data());
        std::swap(grid.density, grid.density_prev);

        // --- RENDER STEP ---
        glTextureSubImage2D(texture, 0, 0, 0, GRID_WIDTH, GRID_HEIGHT, GL_RED, GL_FLOAT, grid.density.data());

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glBindTextureUnit(0, texture);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
