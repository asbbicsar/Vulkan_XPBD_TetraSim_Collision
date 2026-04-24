cmake_minimum_required(VERSION 3.21)
project(Vulkan_XPBD_TetraSim_Interactive)

# Standard C++ version
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCES 
    src/Vulkan_XPBD_TetraSim_Interactive.cpp
)
include(FetchContent)

# -------------------------------------------------------------------
# 1. Vulkan SDK (need to be installed)
# -------------------------------------------------------------------
find_package(Vulkan REQUIRED)

# find glslc compiler (included in Vulkan SDK)
find_program(Vulkan_GLSL_COMPILER NAMES glslc HINTS "$ENV{VULKAN_SDK}/bin")
if(NOT Vulkan_GLSL_COMPILER)
    message(FATAL_ERROR "glslc (shader compiler) not found! Please install Vulkan SDK.")
endif()

# -------------------------------------------------------------------
# 2. GLFW (window generationa and input management)
# -------------------------------------------------------------------
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4	# recent version tag
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# -------------------------------------------------------------------
# 3. GLM (math lib)
# -------------------------------------------------------------------
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.3	# recent version tag
)
FetchContent_MakeAvailable(glm)

# -------------------------------------------------------------------
# 4. exe file generation and link
# -------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

# header and library link
target_link_libraries(${PROJECT_NAME} PRIVATE 
    Vulkan::Vulkan 
    glfw 
    glm::glm
)

# -------------------------------------------------------------------
# 5. copy shaders folder to exe file
# -------------------------------------------------------------------
# set shaders source folder and target folder
set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

# load shader files to compile
file(GLOB SHADER_FILES CONFIGURE_DEPENDS
    "${SHADER_SOURCE_DIR}/*.vert"
    "${SHADER_SOURCE_DIR}/*.frag"
    "${SHADER_SOURCE_DIR}/*.comp"
)

set(SPIRV_BINARY_FILES "")
foreach(SHADER_PATH ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER_PATH} NAME)
    set(SPIRV_PATH "${SHADER_BINARY_DIR}/${SHADER_NAME}.spv")

    # generate shaders output directory
    add_custom_command(
        OUTPUT "${SPIRV_PATH}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_BINARY_DIR}"
        COMMAND ${Vulkan_GLSL_COMPILER} "${SHADER_PATH}" -o "${SPIRV_PATH}"
        DEPENDS "${SHADER_PATH}"
        COMMENT "Compiling shader: ${SHADER_NAME} -> ${SHADER_NAME}.spv"
        VERBATIM
    )
    list(APPEND SPIRV_BINARY_FILES "${SPIRV_PATH}")
endforeach()

# add custom target to handle compiling shaders
add_custom_target(CompileShaders ALL DEPENDS ${SPIRV_BINARY_FILES})

# set compiling shaders go before building an execution file
add_dependencies(${PROJECT_NAME} CompileShaders)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADER_BINARY_DIR}"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMENT "Copying shader files to the execution folder..."
)

# -------------------------------------------------------------------
# 6. copy models folder to exe file
# -------------------------------------------------------------------
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/models"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/models"
    COMMENT "Copying model files to the execution folder..."
)