cmake_minimum_required(VERSION 3.20)

project(tsim LANGUAGES CXX C)

if (MSVC) 
  add_compile_options(/arch:AVX2)
else() 
  add_compile_options(-mavx2)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Fetch GLFW
include(FetchContent)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Fetch Tracy Profiler
FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.13.1
)
set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tracy)

# Find OpenGL
find_package(OpenGL REQUIRED)

add_executable(tsim src/main.cpp src/glad.c src/FluidSolver.cpp)

target_include_directories(tsim PRIVATE include)
target_link_libraries(tsim PRIVATE glfw OpenGL::GL TracyClient)

