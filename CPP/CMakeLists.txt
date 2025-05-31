add_executable(cpp1
    ../../Logger.cc
)

find_package(TinyXML2 REQUIRED)

# Voeg de TinyXML2 bibliotheek expliciet toe aan de link-directives
target_link_libraries(cpp1 PRIVATE ${TinyXML2_LIBRARIES})
target_include_directories(cpp1 PRIVATE ${TinyXML2_INCLUDE_DIRS})
# Kopieer dracula.txt naar de build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dracula.txt ${CMAKE_CURRENT_BINARY_DIR}/dracula.txt COPYONLY)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(cpp1 PRIVATE ENABLE_LOGGING)
endif()

target_include_directories(cpp1 PRIVATE ../../)
