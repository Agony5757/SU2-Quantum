cmake_minimum_required(VERSION 3.1)
project(SU2)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)

set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")

set(LIB_LIST)
set(EXTRA_LDFLAGS)
set(LIB_SU2 Common)

include_directories(${PROJECT_SOURCE_DIR}/SU2_CFD/include)
include_directories(${PROJECT_SOURCE_DIR}/externals/cgns)

if(NOT CMAKE_BUILD_TYPE)
   message(STATUS ":  No build type selected, default is Release")
   set(CMAKE_BUILD_TYPE "Debug")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost)
if(NOT Boost_FOUND)
    message("Not found Boost")
elseif (Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
endif()


# SET(TECIO_ROOT "C:/Program Files/Tecplot/Tecplot 360 EX 2018 R1")

# include_directories(${TECIO_ROOT}/include)
# link_directories(${TECIO_ROOT}/lib)

if (MSVC)
    list(APPEND EXTRA_LDFLAGS
        -D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING
        -D_CRT_SECURE_NO_WARNINGS
        /bigobj)

    if (NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
        foreach (flag
            CMAKE_C_FLAGS
            CMAKE_C_FLAGS_DEBUG
            CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_MINSIZEREL
            CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS
            CMAKE_CXX_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL
            CMAKE_CXX_FLAGS_RELWITHDEBINFO)

            if (${flag} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
            endif()
            if (${flag} MATCHES "/MDd")
                string(REGEX REPLACE "/MDd" "/MTd" ${flag} "${${flag}}")
            endif()
            if (${flag} MATCHES "/W3")
                string(REGEX REPLACE "/W3" "/W0" ${flag} "${${flag}}")
            endif()
            
        endforeach()
    endif()
    
elseif (MINGW)
    list(APPEND EXTRA_LDFLAGS
        -w
        -DGTEST_USE_OWN_TR1_TUPLE=1
        -fpermissive)

    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb -Wa,-mbig-obj ")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG")
elseif (UNIX)
    list(APPEND EXTRA_LDFLAGS
        -w
        -DGTEST_USE_OWN_TR1_TUPLE=1
        -fPIC
        -fpermissive)
    list(APPEND LIB_LIST
        dl
        pthread)

    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG")
endif()
add_definitions(${EXTRA_LDFLAGS})

add_subdirectory(Common)
add_subdirectory(SU2_CFD)
add_subdirectory(externals)




