# SU2-Quantum

## Purpose
For code availability of ''Quantum Approach To Accelerate Finite Volume Method on Steady Computational Fluid Dynamics Problems''(https://arxiv.org/abs/2102.03557v1).

## Directories

- SU2-Quantum
    
    We simplify one version of open source code SU2 (https://github.com/su2code/SU2) and append a quantum error simulator on it.

    The code supports running on both Linux and Windows.

- Examples

    Three examples presented in the article with mesh file and configuration file given.

## Usage

1. Compile the SU2-Quantum with CMake and a C++ compiler (supporting at least C++14). The executable is compiled as build/release/bin/SU2_CFD (+.exe if on Windows)

2. Put the example config/mesh file and the executable into the same directory. Run with ''./SU2_CFD {ConfigName}''(SU2_CFD.exe if on Windows). Such as ''./SU2_CFD inv_ONERAM6.cfg''.

3. Set the config in the bottom region of the configuration file. Four configurations are supported which controls the quantum simulator.
    
    - USE_QUANTUM = [YES/NO] (if selected no, then the system remains to be the original SU2 CFD solver)
    
    - LINF_SAMPLING_CONSTANT= 32 (for the constant C presented in the $l_\infty$ tomography algorithm)

    - QUANTUM_ERROR_THRESHOLD = 1E-2 (for the $\epsilon$ presented in the $l_\infty$ tomography algorithm)

    - LINF_TOMOGRAPHY_VERSION = 3 (I wrote three versions of $l_\infty$ tomography, where the version 3 is shown in the paper and has the best performance.)

## Correspondance

If you have any question about this code, please send mail to wuyuchun@ustc.edu.cn