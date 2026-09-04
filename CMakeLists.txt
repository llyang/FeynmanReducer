cmake_minimum_required(VERSION 3.20)
project(FeynmanReducer CXX)

include(CheckIPOSupported)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
option(FR_ENABLE_NAUTY "Build the Traces/nauty symmetry backend" ON)
option(FR_ENABLE_BLISS "Build the bliss symmetry backend" ON)
option(FR_NATIVE_OPTIMIZATION
  "Optimize Release builds for the CPU used to compile FeynmanReducer" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

check_cxx_source_compiles(
  "
    #if !defined(__SIZEOF_INT128__)
    #error unsigned __int128 is unavailable
    #endif
    int main()
    {
      unsigned __int128 lhs = 3;
      unsigned __int128 rhs = 5;
      return lhs * rhs == 15 ? 0 : 1;
    }
  "
  FR_HAVE_UNSIGNED_INT128
)
if(NOT FR_HAVE_UNSIGNED_INT128)
  message(FATAL_ERROR
    "FeynmanReducer requires compiler and target support for unsigned __int128"
  )
endif()

find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLINT REQUIRED IMPORTED_TARGET flint)
pkg_check_modules(FIREFLY REQUIRED IMPORTED_TARGET firefly)
find_package(Threads REQUIRED)

if(NOT FIREFLY_VERSION VERSION_EQUAL "2.0.3")
  message(FATAL_ERROR
    "FeynmanReducer's reconstruction overlay requires FireFly 2.0.3; "
    "found ${FIREFLY_VERSION}"
  )
endif()

set(FR_HAVE_NAUTY FALSE)
if(FR_ENABLE_NAUTY)
  pkg_search_module(NAUTY QUIET IMPORTED_TARGET libnauty nauty)
  if(NAUTY_FOUND)
    find_path(
      FR_NAUTY_TRACES_INCLUDE_DIR
      traces.h
      HINTS ${NAUTY_INCLUDE_DIRS}
    )
    if(FR_NAUTY_TRACES_INCLUDE_DIR)
      set(FR_NAUTY_TARGET PkgConfig::NAUTY)
      set(FR_HAVE_NAUTY TRUE)
    endif()
  else()
    find_path(FR_NAUTY_INCLUDE_DIR nauty.h PATH_SUFFIXES nauty)
    find_path(FR_NAUTY_TRACES_INCLUDE_DIR traces.h PATH_SUFFIXES nauty)
    find_library(FR_NAUTY_LIBRARY NAMES nauty)
    if(FR_NAUTY_INCLUDE_DIR
        AND FR_NAUTY_TRACES_INCLUDE_DIR
        AND FR_NAUTY_LIBRARY)
      add_library(fr_nauty UNKNOWN IMPORTED)
      set_target_properties(fr_nauty PROPERTIES
        IMPORTED_LOCATION "${FR_NAUTY_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES
          "${FR_NAUTY_INCLUDE_DIR};${FR_NAUTY_TRACES_INCLUDE_DIR}"
      )
      set(FR_NAUTY_TARGET fr_nauty)
      set(FR_HAVE_NAUTY TRUE)
    endif()
  endif()
endif()

if(FR_HAVE_NAUTY)
  set(FR_SAVED_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
  set(FR_SAVED_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
  set(CMAKE_REQUIRED_LIBRARIES ${FR_NAUTY_TARGET})
  set(CMAKE_REQUIRED_INCLUDES ${FR_NAUTY_TRACES_INCLUDE_DIR})
  check_cxx_source_compiles(
    "
      #include <traces.h>
      int main()
      {
        auto* function = &Traces;
        return function == nullptr;
      }
    "
    FR_NAUTY_TRACES_LINKS
  )
  set(CMAKE_REQUIRED_LIBRARIES "${FR_SAVED_REQUIRED_LIBRARIES}")
  set(CMAKE_REQUIRED_INCLUDES "${FR_SAVED_REQUIRED_INCLUDES}")
  unset(FR_SAVED_REQUIRED_LIBRARIES)
  unset(FR_SAVED_REQUIRED_INCLUDES)
  if(NOT FR_NAUTY_TRACES_LINKS)
    set(FR_HAVE_NAUTY FALSE)
  endif()
endif()

set(FR_HAVE_BLISS FALSE)
if(FR_ENABLE_BLISS)
  find_package(Bliss QUIET CONFIG)
  if(TARGET Bliss::libbliss)
    set(FR_BLISS_TARGET Bliss::libbliss)
    set(FR_HAVE_BLISS TRUE)
  else()
    find_path(FR_BLISS_INCLUDE_DIR bliss/graph.hh)
    find_library(FR_BLISS_LIBRARY NAMES bliss)
    if(FR_BLISS_INCLUDE_DIR AND FR_BLISS_LIBRARY)
      add_library(fr_bliss UNKNOWN IMPORTED)
      set_target_properties(fr_bliss PROPERTIES
        IMPORTED_LOCATION "${FR_BLISS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FR_BLISS_INCLUDE_DIR}"
      )
      set(FR_BLISS_TARGET fr_bliss)
      set(FR_HAVE_BLISS TRUE)
    endif()
  endif()
endif()

if(NOT FR_HAVE_NAUTY AND NOT FR_HAVE_BLISS)
  message(FATAL_ERROR
    "FeynmanReducer requires at least one compiled symmetry backend. "
    "Neither Traces/nauty nor bliss was found and enabled. Install at least "
    "one backend and leave its FR_ENABLE_* option enabled. Runtime "
    "symmetry_backend: none remains available in supported builds."
  )
endif()

if(FR_HAVE_NAUTY)
  message(STATUS "FeynmanReducer symmetry backend: nauty (Traces)")
endif()
if(FR_HAVE_BLISS)
  message(STATUS "FeynmanReducer symmetry backend: bliss")
endif()

if(TARGET yaml-cpp::yaml-cpp)
  set(FR_YAMLCPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(FR_YAMLCPP_TARGET yaml-cpp)
endif()

add_library(feynman_reducer_warnings INTERFACE)
target_compile_options(feynman_reducer_warnings INTERFACE
  -Wall -Wextra -Wpedantic
  -Wconversion -Wsign-conversion
)

if(FR_NATIVE_OPTIMIZATION)
  check_cxx_compiler_flag("-march=native" FR_HAVE_MARCH_NATIVE)
  if(FR_HAVE_MARCH_NATIVE)
    set(FR_NATIVE_OPTIMIZATION_FLAG "-march=native")
  else()
    check_cxx_compiler_flag("-mcpu=native" FR_HAVE_MCPU_NATIVE)
    if(FR_HAVE_MCPU_NATIVE)
      set(FR_NATIVE_OPTIMIZATION_FLAG "-mcpu=native")
    else()
      message(FATAL_ERROR
        "FR_NATIVE_OPTIMIZATION is ON, but the compiler supports neither "
        "-march=native nor -mcpu=native. Configure with "
        "-DFR_NATIVE_OPTIMIZATION=OFF to build a portable binary."
      )
    endif()
  endif()
  target_compile_options(feynman_reducer_warnings INTERFACE
    $<$<CONFIG:Release>:${FR_NATIVE_OPTIMIZATION_FLAG}>
  )
  # Keep the target-CPU setting during an IPO/LTO link as well as compilation.
  target_link_options(feynman_reducer_warnings INTERFACE
    $<$<CONFIG:Release>:${FR_NATIVE_OPTIMIZATION_FLAG}>
  )
  message(STATUS
    "FeynmanReducer Release native optimization: ${FR_NATIVE_OPTIMIZATION_FLAG}"
  )
endif()

set(FR_SYMMETRY_SOURCES
  src/symmetry/BackendDispatch.cpp
  src/symmetry/GraphEncoding.cpp
  src/symmetry/PermutationValidation.cpp
  src/symmetry/SubsectorAnalysis.cpp
  src/symmetry/Symmetry.cpp
)
if(FR_HAVE_NAUTY)
  list(APPEND FR_SYMMETRY_SOURCES src/symmetry/NautyBackend.cpp)
endif()
if(FR_HAVE_BLISS)
  list(APPEND FR_SYMMETRY_SOURCES src/symmetry/BlissBackend.cpp)
endif()

add_library(feynman_reducer_model STATIC
  src/core/FlintRational.cpp
  src/core/IntegralFormatting.cpp
)
target_include_directories(feynman_reducer_model PUBLIC src)
target_link_libraries(feynman_reducer_model
  PUBLIC PkgConfig::FLINT
  PRIVATE feynman_reducer_warnings
)

add_library(feynman_reducer_topology STATIC
  src/topology/IntegralLayout.cpp
  src/topology/SectorUtils.cpp
)
target_include_directories(feynman_reducer_topology PUBLIC src)
target_link_libraries(feynman_reducer_topology
  PUBLIC feynman_reducer_model
  PRIVATE feynman_reducer_warnings
)

add_library(feynman_reducer_symmetry STATIC ${FR_SYMMETRY_SOURCES})
target_include_directories(feynman_reducer_symmetry PUBLIC src)
target_compile_definitions(feynman_reducer_symmetry PUBLIC
  FR_HAVE_NAUTY=$<BOOL:${FR_HAVE_NAUTY}>
  FR_HAVE_BLISS=$<BOOL:${FR_HAVE_BLISS}>
)
set_target_properties(feynman_reducer_symmetry PROPERTIES
  FR_HAVE_NAUTY "${FR_HAVE_NAUTY}"
  FR_HAVE_BLISS "${FR_HAVE_BLISS}"
)
target_link_libraries(feynman_reducer_symmetry
  PRIVATE
  feynman_reducer_topology
  feynman_reducer_warnings
)
if(FR_HAVE_NAUTY)
  target_link_libraries(feynman_reducer_symmetry PRIVATE ${FR_NAUTY_TARGET})
  target_include_directories(
    feynman_reducer_symmetry PRIVATE ${FR_NAUTY_TRACES_INCLUDE_DIR}
  )
endif()
if(FR_HAVE_BLISS)
  target_link_libraries(feynman_reducer_symmetry PRIVATE ${FR_BLISS_TARGET})
endif()

add_library(feynman_reducer_compiler STATIC
  src/compiler/YamlCompiler.cpp
  src/compiler/detail/ExactPolynomial.cpp
  src/compiler/detail/ExpressionParser.cpp
  src/compiler/detail/IntegralParser.cpp
  src/compiler/detail/SymanzikCompiler.cpp
)
target_include_directories(feynman_reducer_compiler PUBLIC src)
target_link_libraries(feynman_reducer_compiler
  PRIVATE
  ${FR_YAMLCPP_TARGET}
  PkgConfig::FLINT
  feynman_reducer_symmetry
  feynman_reducer_topology
  feynman_reducer_warnings
)

add_library(feynman_reducer_reduction STATIC
  src/basis/DSeparatingBasisSearch.cpp
  src/basis/DSeparatingReduction.cpp
  src/basis/FiniteField.cpp
  src/basis/LpNormalization.cpp
  src/basis/NativeFixedBasisOracle.cpp
  src/basis/UnivariateSeparation.cpp
  src/reduction/BlackBoxFeynman.cpp
  src/reduction/BlockTriangular.cpp
  src/reduction/BlockReplayBuilder.cpp
  src/reduction/CoefficientEvaluator.cpp
  src/reduction/CompiledReplay.cpp
  src/reduction/CompactPlanner.cpp
  src/reduction/ConstantRationalReconstruction.cpp
  src/reduction/DirectSeedExpansion.cpp
  src/reduction/ReferencePlanBuilder.cpp
  src/reduction/EquationGenerator.cpp
  src/reduction/FireflyResultImport.cpp
  src/reduction/FeynmanReducer.cpp
  src/masters/detail/GlobalBasisSelector.cpp
  src/reduction/JetEquationGenerator.cpp
  src/reduction/JetKernelPlanner.cpp
  src/reduction/JetSymmetryCanonicalizer.cpp
  src/reduction/KernelPlanner.cpp
  src/reduction/KernelPublication.cpp
  src/reduction/ProjectedSeedExpansion.cpp
  src/reduction/TopLpTargets.cpp
  src/reduction/ReplayExecutor.cpp
  src/reduction/ReplayFinalizer.cpp
  src/reduction/ReplaySelector.cpp
  src/reduction/SymmetryCanonicalizer.cpp
)
target_include_directories(feynman_reducer_reduction PUBLIC src)
target_include_directories(feynman_reducer_reduction SYSTEM BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/firefly-2.0.3-overlay
)
target_link_libraries(feynman_reducer_reduction
  PUBLIC
  feynman_reducer_model
  PkgConfig::FIREFLY
  PRIVATE
  Threads::Threads
  feynman_reducer_symmetry
  feynman_reducer_topology
  feynman_reducer_warnings
)

add_library(feynman_reducer_masters STATIC
  src/masters/MasterFinder.cpp
  src/masters/detail/SingularProcess.cpp
)
target_include_directories(feynman_reducer_masters PUBLIC src)
target_link_libraries(feynman_reducer_masters
  PUBLIC feynman_reducer_model
  PRIVATE
  Threads::Threads
  feynman_reducer_reduction
  feynman_reducer_symmetry
  feynman_reducer_topology
  feynman_reducer_warnings
)

add_library(feynman_reducer_output STATIC
  src/output/GenerateOutput.cpp
)
target_include_directories(feynman_reducer_output PUBLIC src)
target_link_libraries(feynman_reducer_output
  PUBLIC feynman_reducer_model
  PRIVATE feynman_reducer_warnings
)

add_library(feynman_reducer_app_support STATIC
  src/app/FireflyLogArchive.cpp
  src/app/RunLog.cpp
  src/app/RunTiming.cpp
)
target_include_directories(feynman_reducer_app_support PUBLIC src)
target_link_libraries(
  feynman_reducer_app_support PRIVATE feynman_reducer_warnings
)

add_executable(FeynmanReducer src/app/main.cpp)
target_link_libraries(FeynmanReducer PRIVATE
  feynman_reducer_compiler
  feynman_reducer_masters
  feynman_reducer_reduction
  feynman_reducer_output
  feynman_reducer_symmetry
  feynman_reducer_app_support
  feynman_reducer_warnings
)

add_executable(
  reduction_validate
  src/app/reduction_validate.cpp
  src/validation/ReductionValidator.cpp
)
target_link_libraries(reduction_validate PRIVATE
  feynman_reducer_compiler
  feynman_reducer_model
  feynman_reducer_warnings
)

check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
  set_target_properties(
    feynman_reducer_symmetry
    feynman_reducer_compiler
    feynman_reducer_model
    feynman_reducer_topology
    feynman_reducer_reduction
    feynman_reducer_masters
    feynman_reducer_output
    feynman_reducer_app_support
    FeynmanReducer
    reduction_validate
    PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
  )
endif()
