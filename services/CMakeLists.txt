find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(add_microservice service_name service_dir)
    set(venv_dir "${CMAKE_CURRENT_BINARY_DIR}/.venv-${service_name}")
    set(requirements_file "${service_dir}/requirements.txt")

    add_custom_target("${service_name}-microservice"
        DEPENDS "${venv_dir}/pyvenv.cfg"
    )

    add_custom_command(
        OUTPUT "${venv_dir}/pyvenv.cfg"
        COMMAND "${Python3_EXECUTABLE}" -m venv "${venv_dir}"
        COMMAND "${venv_dir}/Scripts/pip" install -r "${requirements_file}"
    )

    install(
        DIRECTORY "${venv_dir}"
        TYPE BIN
    )
    install(
        DIRECTORY "${service_dir}"
        TYPE BIN
    )
endfunction()

add_microservice(spice "${CMAKE_CURRENT_SOURCE_DIR}/spice-micro-services")
add_microservice(vessel-detection "${CMAKE_CURRENT_SOURCE_DIR}/vessel-detection-microservice")
