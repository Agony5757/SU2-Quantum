# SU2-Quantum

## Purpose
For code availability of ''Quantum Approach To Accelerate Finite Volume Method on Steady Computational Fluid Dynamics Problems''(https://link.springer.com/article/10.1007/s11128-022-03478-w).

## Directories

- SU2-Quantum
    
    We simplify one version of open source code SU2 (https://github.com/su2code/SU2) and append a quantum error simulator on it.

    The code supports running on both Linux and Windows.

- Examples

    Three examples presented in the article with mesh file and configuration file given.

## Build

### Tested Environments
- Windows: MSVC
- Linux: gcc

### Prerequisites
- C++ compiler: with C++14 support
- (optional) Boost
- (optional) Tecio (see TECIO support section)

### Build on Windows
- Open folder SU2-Quantum with Visual Studio to load CMakeLists.txt
- Start building 
    > SU2_Quantum/build/{platform}/bin/SU2_CFD.exe

### Build on Linux

- Work directory
    > SU2_Quantum/SU2_Quantum
- Execute the commands
```bash
mkdir build
cd build
cmake ..
make -j8
```

### TECIO support
To build with Tecio support, you have to first install the tecio library. Then check the following codes:

- SU2_Quantum/Common/basic_types/datatype_structure.hpp
    - In the last line, uncomment the #HAVE_TECIO

- SU2_Quantum/CMakeLists.txt
    - SET(TECIO_ROOT path_of_tecio)
    - then uncomment include_directories(...) and link_directories(...)

- SU2_Quantum/SU2_CFD/CMakeLists.txt
    - replace 'tecio.lib' with the tecio library name

Finally, it will be built with tecio support.

## Usage

1. Compile the SU2-Quantum with CMake and a C++ compiler (supporting at least C++14). The executable is compiled as build/release/bin/SU2_CFD (+.exe if on Windows)

2. Put the example config/mesh file and the executable into the same directory. Run with ''./SU2_CFD {ConfigName}''(SU2_CFD.exe if on Windows). Such as ''./SU2_CFD inv_ONERAM6.cfg''.

3. Set the config in the bottom region of the configuration file. Four configurations are supported which controls the quantum simulator.
    
    - USE_QUANTUM = [YES/NO] (if selected no, then the system remains to be the original SU2 CFD solver)
    
    - LINF_SAMPLING_CONSTANT= 32 (for the constant C presented in the $l_\infty$ tomography algorithm)

    - QUANTUM_ERROR_THRESHOLD = 1E-2 (for the $\epsilon$ presented in the $l_\infty$ tomography algorithm)

    - LINF_TOMOGRAPHY_VERSION = 3 (I wrote three versions of $l_\infty$ tomography, where the version 3 is shown in the paper and has the best performance.)

## Correspondance

If you have any question about this code, please send mail to chenzhaoyun@iai.ustc.edu.cn or wuyuchun@ustc.edu.cn