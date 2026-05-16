# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-src")
  file(MAKE_DIRECTORY "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-build"
  "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-subbuild/tracy-populate-prefix"
  "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-subbuild/tracy-populate-prefix/tmp"
  "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-subbuild/tracy-populate-prefix/src/tracy-populate-stamp"
  "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-subbuild/tracy-populate-prefix/src"
  "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-subbuild/tracy-populate-prefix/src/tracy-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-subbuild/tracy-populate-prefix/src/tracy-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/ultim/tsim/build/tracy-csvexport/_deps/tracy-subbuild/tracy-populate-prefix/src/tracy-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
