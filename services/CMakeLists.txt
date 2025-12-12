find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(GNUInstallDirs)

function(add_microservice service_name service_dir)
    set(venv_dir "${CMAKE_INSTALL_FULL_BINDIR}/.venv-${service_name}")
    set(requirements_file "${service_dir}/requirements.txt")

    install(CODE "
        if(NOT EXISTS ${venv_dir}/pyvenv.cfg)
            execute_process(
                COMMAND \"${Python3_EXECUTABLE}\" -m venv \"${venv_dir}\"
            )
            execute_process(
                COMMAND \"${venv_dir}/Scripts/pip\" install -r \"${requirements_file}\"
            )
        endif()
    ")
    install(
        DIRECTORY "${service_dir}"
        TYPE BIN
    )
endfunction()

add_microservice(spice "${CMAKE_CURRENT_SOURCE_DIR}/spice-micro-services")
add_microservice(vessel-detection "${CMAKE_CURRENT_SOURCE_DIR}/vessel-detection-microservice")
