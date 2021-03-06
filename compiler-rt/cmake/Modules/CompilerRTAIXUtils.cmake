include(CMakeParseArguments)
include(CompilerRTUtils)

function(get_aix_libatomic_default_link_flags link_flags export_list)
  set(${link_flags}
    "-Wl,-H512 -Wl,-D0 \
     -Wl,-T512 -Wl,-bhalt:4 -Wl,-bernotok \
     -Wl,-bnoentry -Wl,-bexport:${export_list} \
     -Wl,-bmodtype:SRE -Wl,-lc" PARENT_SCOPE)
endfunction()

macro(archive_aix_libatomic name)
  cmake_parse_arguments(LIB
    ""
    ""
    "ARCHS;PARENT_TARGET"
    ${ARGN})
  set(shared_libraries_to_archive "")
  foreach (arch ${LIB_ARCHS})
    if(CAN_TARGET_${arch})
      set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/libatomic-${arch}.dir")
      # FIXME: Target name should be kept consistent with definition
      # in AddCompilerRT.cmake added by
      # add_compiler_rt_runtime(<name> SHARED ...)
      set(target ${name}-dynamic-${arch})
      if(TARGET ${target})
        file(MAKE_DIRECTORY ${output_dir})
        add_custom_command(OUTPUT "${output_dir}/libatomic.so.1"
                           POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E
                           copy "$<TARGET_FILE:${target}>"
                                "${output_dir}/libatomic.so.1"
                           DEPENDS ${target})
        list(APPEND shared_libraries_to_archive "${output_dir}/libatomic.so.1")
      endif()
    endif()
  endforeach()
  if(shared_libraries_to_archive)
    set(output_dir "")
    set(install_dir "")
    get_compiler_rt_output_dir(${COMPILER_RT_DEFAULT_TARGET_ARCH} output_dir)
    get_compiler_rt_install_dir(${COMPILER_RT_DEFAULT_TARGET_ARCH} install_dir)
    add_custom_command(OUTPUT "${output_dir}/libatomic.a"
                       COMMAND ${CMAKE_AR} -X32_64 r "${output_dir}/libatomic.a"
                       ${shared_libraries_to_archive}
                       DEPENDS ${shared_libraries_to_archive})
    install(FILES "${output_dir}/libatomic.a"
            DESTINATION ${install_dir})
    add_custom_target(aix-libatomic
                      DEPENDS "${output_dir}/libatomic.a")
  endif()
  add_dependencies(${LIB_PARENT_TARGET} aix-libatomic)
endmacro()
