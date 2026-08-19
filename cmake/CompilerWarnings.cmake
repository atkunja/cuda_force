# Warnings are applied through an interface target rather than globally so they
# reach first-party code without being forced onto vendored dependencies, where
# they would produce noise nobody can act on.

add_library(cudaforge_warnings INTERFACE)

if(MSVC)
  target_compile_options(cudaforge_warnings INTERFACE /W4 /permissive-)
else()
  target_compile_options(
    cudaforge_warnings
    INTERFACE -Wall
              -Wextra
              -Wpedantic
              -Wshadow
              -Wnon-virtual-dtor
              -Wcast-align
              -Wunused
              -Woverloaded-virtual
              -Wconversion
              -Wsign-conversion
              -Wdouble-promotion
              -Wformat=2)
endif()

option(CUDAFORGE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
if(CUDAFORGE_WARNINGS_AS_ERRORS)
  if(MSVC)
    target_compile_options(cudaforge_warnings INTERFACE /WX)
  else()
    target_compile_options(cudaforge_warnings INTERFACE -Werror)
  endif()
endif()
