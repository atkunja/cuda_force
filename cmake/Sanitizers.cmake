# Sanitizers are mutually exclusive at the ABI level: ThreadSanitizer and
# AddressSanitizer instrument the same allocations differently and cannot be
# linked together. This exposes one option rather than several booleans so the
# invalid combination is unrepresentable.

function(cudaforge_apply_sanitizer target)
  if(NOT CUDAFORGE_SANITIZER OR CUDAFORGE_SANITIZER STREQUAL "OFF")
    return()
  endif()

  if(MSVC)
    message(WARNING "CUDAFORGE_SANITIZER is not supported with MSVC; ignoring")
    return()
  endif()

  string(TOLOWER "${CUDAFORGE_SANITIZER}" sanitizer)
  if(sanitizer STREQUAL "address")
    set(flags -fsanitize=address -fno-omit-frame-pointer)
  elseif(sanitizer STREQUAL "thread")
    set(flags -fsanitize=thread -fno-omit-frame-pointer)
  elseif(sanitizer STREQUAL "undefined")
    set(flags -fsanitize=undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  else()
    message(FATAL_ERROR "Unknown CUDAFORGE_SANITIZER '${CUDAFORGE_SANITIZER}'")
  endif()

  target_compile_options(${target} PRIVATE ${flags} -g)
  target_link_options(${target} PRIVATE ${flags})
endfunction()
