function(build_msvc build_type source_path)
    if(build_type STREQUAL "DEBUG")
        set(vcpkg_configuration "Debug")
        set(label "${TARGET_TRIPLET}-dbg")
        set(packages_dir "${CURRENT_PACKAGES_DIR}/debug")
    else()
        set(vcpkg_configuration "Release")
        set(label "${TARGET_TRIPLET}-rel")
        set(packages_dir "${CURRENT_PACKAGES_DIR}")
    endif()

    set(build_path "${CURRENT_BUILDTREES_DIR}/${label}")
    file(REMOVE_RECURSE "${build_path}")
    file(COPY "${source_path}/" DESTINATION "${build_path}")

    if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
        vcpkg_replace_string("${build_path}/src/include/port/win32.h" "__declspec (dllimport)" "")
    endif()
    vcpkg_replace_string("${build_path}/src/tools/msvc/MSBuildProject.pm" "perl " "\"${PERL}\" ")
    configure_file("${CURRENT_PORT_DIR}/libpq.props.in" "${build_path}/libpq.props" @ONLY)
    configure_file("${CURRENT_PORT_DIR}/vcpkg-libs.props.in" "${build_path}/vcpkg-libs.props" @ONLY)
    set(config "# Generated by ${CMAKE_CURRENT_LIST_FILE}\n\n")
    foreach(var IN ITEMS VCPKG_TARGET_ARCHITECTURE VCPKG_LIBRARY_LINKAGE VCPKG_CRT_LINKAGE)
        string(APPEND config "\$config->{${var}} = \"${${var}}\";\n")
    endforeach()
    foreach(option IN ITEMS icu lz4 nls openssl python tcl xml xslt zlib zstd)
        if(option IN_LIST FEATURES)
            string(APPEND config "\$config->{${option}} = \"${CURRENT_INSTALLED_DIR}\";\n")
        endif()
    endforeach()
    if("openssl" IN_LIST FEATURES)
        file(STRINGS "${CURRENT_INSTALLED_DIR}/lib/pkgconfig/openssl.pc" OPENSSL_VERSION REGEX "Version:")
        string(APPEND config "\$config->{openssl_version} = '${OPENSSL_VERSION}';\n")
    endif()
    string(APPEND config "\$config->{python_version} = '3.10';\n")
    string(APPEND config "\$config->{tcl_version} = '90';\n")
    file(WRITE "${build_path}/src/tools/msvc/config.pl" "${config}")

    set(build_in_parallel "-m")
    set(build_targets libpq libecpg_compat)
    set(install_target core)
    if(HAS_TOOLS AND NOT build_type STREQUAL "DEBUG")
        set(build_in_parallel "") # mitigate winflex races
        set(build_targets client)
        set(install_target client)
    endif()

    string(REPLACE "x86" "Win32" platform "${VCPKG_TARGET_ARCHITECTURE}")
    vcpkg_get_windows_sdk(VCPKG_TARGET_PLATFORM_VERSION)
    set(ENV{MSBFLAGS} "\
        /p:Platform=${platform} \
        /p:PlatformToolset=${VCPKG_PLATFORM_TOOLSET} \
        /p:VCPkgLocalAppDataDisabled=true \
        /p:UseIntelMKL=No \
        /p:WindowsTargetPlatformVersion=${VCPKG_TARGET_PLATFORM_VERSION} \
        /p:VcpkgConfiguration=${vcpkg_configuration} \
        ${build_in_parallel} \
        /p:ForceImportBeforeCppTargets=\"${SCRIPTS}/buildsystems/msbuild/vcpkg.targets;${build_path}/vcpkg-libs.props\" \
        /p:VcpkgTriplet=${TARGET_TRIPLET} \
        /p:VcpkgCurrentInstalledDir=\"${CURRENT_INSTALLED_DIR}\" \
        /p:ForceImportAfterCppTargets=\"${build_path}/libpq.props\" \
    ")

    message(STATUS "Building ${label}")
    foreach(target IN LISTS build_targets)
        string(REPLACE "client" "" target "${target}")
        vcpkg_execute_required_process(
            COMMAND "${PERL}" build.pl ${build_type} ${target}
            WORKING_DIRECTORY "${build_path}/src/tools/msvc"
            LOGNAME "build-${target}-${label}"
        )
    endforeach()

    message(STATUS "Installing ${label}")
    vcpkg_execute_required_process(
        COMMAND "${PERL}" install.pl "${packages_dir}" ${install_target}
        WORKING_DIRECTORY "${build_path}/src/tools/msvc"
        LOGNAME "install-${label}"
    )
endfunction()
