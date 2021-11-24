/*!
 * \file CFluidIteration.cpp
 * \brief Main subroutines used by SU2_CFD
 * \author F. Palacios, T. Economon
 * \version 7.0.6 "Blackbird"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../../include/iteration/CFluidIteration.hpp"
#include "../../include/output/COutput.hpp"

void CFluidIteration::Preprocess(COutput* output, CIntegration**** integration, CGeometry**** geometry,
                                 CSolver***** solver, CNumerics****** numerics, CConfig** config,
                                 CSurfaceMovement** surface_movement, CVolumetricMovement*** grid_movement,
                                 CFreeFormDefBox*** FFDBox, unsigned short val_iZone, unsigned short val_iInst) {
  unsigned long TimeIter = config[val_iZone]->GetTimeIter();

  bool fsi = config[val_iZone]->GetFSI_Simulation();
  unsigned long OuterIter = config[val_iZone]->GetOuterIter();

  /*--- Set the initial condition for FSI problems with subiterations ---*/
  /*--- This is done only in the first block subiteration.---*/
  /*--- From then on, the solver reuses the partially converged solution obtained in the previous subiteration ---*/
  if (fsi && (OuterIter == 0)) {
    solver[val_iZone][val_iInst][MESH_0][FLOW_SOL]->SetInitialCondition(
        geometry[val_iZone][val_iInst], solver[val_iZone][val_iInst], config[val_iZone], TimeIter);
  }

  /*--- Apply a Wind Gust ---*/

  if (config[val_iZone]->GetWind_Gust()) {
    SetWind_GustField(config[val_iZone], geometry[val_iZone][val_iInst], solver[val_iZone][val_iInst]);
  }
}

void CFluidIteration::Iterate(COutput* output, CIntegration**** integration, CGeometry**** geometry,
                              CSolver***** solver, CNumerics****** numerics, CConfig** config,
                              CSurfaceMovement** surface_movement, CVolumetricMovement*** grid_movement,
                              CFreeFormDefBox*** FFDBox, unsigned short val_iZone, unsigned short val_iInst) {
  unsigned long InnerIter, TimeIter;

  bool unsteady = (config[val_iZone]->GetTime_Marching() == DT_STEPPING_1ST) ||
                  (config[val_iZone]->GetTime_Marching() == DT_STEPPING_2ND);
  bool frozen_visc = (config[val_iZone]->GetContinuous_Adjoint() && config[val_iZone]->GetFrozen_Visc_Cont()) ||
                     (config[val_iZone]->GetDiscrete_Adjoint() && config[val_iZone]->GetFrozen_Visc_Disc());
  TimeIter = config[val_iZone]->GetTimeIter();

  /* --- Setting up iteration values depending on if this is a
   steady or an unsteady simulaiton */

  InnerIter = config[val_iZone]->GetInnerIter();

  /*--- Update global parameters ---*/

  switch (config[val_iZone]->GetKind_Solver()) {
    case EULER:
    case DISC_ADJ_EULER:
    case INC_EULER:
    case DISC_ADJ_INC_EULER:
      config[val_iZone]->SetGlobalParam(EULER, RUNTIME_FLOW_SYS);
      break;

    case NAVIER_STOKES:
    case DISC_ADJ_NAVIER_STOKES:
    case INC_NAVIER_STOKES:
    case DISC_ADJ_INC_NAVIER_STOKES:
      config[val_iZone]->SetGlobalParam(NAVIER_STOKES, RUNTIME_FLOW_SYS);
      break;

    case RANS:
    case DISC_ADJ_RANS:
    case INC_RANS:
    case DISC_ADJ_INC_RANS:
      config[val_iZone]->SetGlobalParam(RANS, RUNTIME_FLOW_SYS);
      break;
  }

  /*--- Solve the Euler, Navier-Stokes or Reynolds-averaged Navier-Stokes (RANS) equations (one iteration) ---*/

  integration[val_iZone][val_iInst][FLOW_SOL]->MultiGrid_Iteration(geometry, solver, numerics, config, RUNTIME_FLOW_SYS,
                                                                   val_iZone, val_iInst);

  if ((config[val_iZone]->GetKind_Solver() == RANS || config[val_iZone]->GetKind_Solver() == DISC_ADJ_RANS ||
       config[val_iZone]->GetKind_Solver() == INC_RANS || config[val_iZone]->GetKind_Solver() == DISC_ADJ_INC_RANS) &&
      !frozen_visc) {
    /*--- Solve the turbulence model ---*/

    config[val_iZone]->SetGlobalParam(RANS, RUNTIME_TURB_SYS);
    integration[val_iZone][val_iInst][TURB_SOL]->SingleGrid_Iteration(geometry, solver, numerics, config,
                                                                      RUNTIME_TURB_SYS, val_iZone, val_iInst);

    /*--- Solve transition model ---*/

    if (config[val_iZone]->GetKind_Trans_Model() == LM) {
      config[val_iZone]->SetGlobalParam(RANS, RUNTIME_TRANS_SYS);
      integration[val_iZone][val_iInst][TRANS_SOL]->SingleGrid_Iteration(geometry, solver, numerics, config,
                                                                         RUNTIME_TRANS_SYS, val_iZone, val_iInst);
    }
  }

  if (config[val_iZone]->GetWeakly_Coupled_Heat()) {
    config[val_iZone]->SetGlobalParam(RANS, RUNTIME_HEAT_SYS);
    integration[val_iZone][val_iInst][HEAT_SOL]->SingleGrid_Iteration(geometry, solver, numerics, config,
                                                                      RUNTIME_HEAT_SYS, val_iZone, val_iInst);
  }

  /*--- Incorporate a weakly-coupled radiation model to the analysis ---*/
  if (config[val_iZone]->AddRadiation()) {
    config[val_iZone]->SetGlobalParam(RANS, RUNTIME_RADIATION_SYS);
    integration[val_iZone][val_iInst][RAD_SOL]->SingleGrid_Iteration(geometry, solver, numerics, config,
                                                                     RUNTIME_RADIATION_SYS, val_iZone, val_iInst);
  }

  /*--- Adapt the CFL number using an exponential progression with under-relaxation approach. ---*/

  if (config[val_iZone]->GetCFL_Adapt() == YES) {
    SU2_OMP_PARALLEL
    solver[val_iZone][val_iInst][MESH_0][FLOW_SOL]->AdaptCFLNumber(geometry[val_iZone][val_iInst],
                                                                   solver[val_iZone][val_iInst], config[val_iZone]);
  }

  /*--- Call Dynamic mesh update if AEROELASTIC motion was specified ---*/

  if ((config[val_iZone]->GetGrid_Movement()) && (config[val_iZone]->GetAeroelastic_Simulation()) && unsteady) {
    SetGrid_Movement(geometry[val_iZone][val_iInst], surface_movement[val_iZone], grid_movement[val_iZone][val_iInst],
                     solver[val_iZone][val_iInst], config[val_iZone], InnerIter, TimeIter);

    /*--- Apply a Wind Gust ---*/

    if (config[val_iZone]->GetWind_Gust()) {
      if (InnerIter % config[val_iZone]->GetAeroelasticIter() == 0 && InnerIter != 0)
        SetWind_GustField(config[val_iZone], geometry[val_iZone][val_iInst], solver[val_iZone][val_iInst]);
    }
  }
}

void CFluidIteration::Update(COutput* output, CIntegration**** integration, CGeometry**** geometry, CSolver***** solver,
                             CNumerics****** numerics, CConfig** config, CSurfaceMovement** surface_movement,
                             CVolumetricMovement*** grid_movement, CFreeFormDefBox*** FFDBox, unsigned short val_iZone,
                             unsigned short val_iInst) {
  unsigned short iMesh;

  /*--- Dual time stepping strategy ---*/

  if ((config[val_iZone]->GetTime_Marching() == DT_STEPPING_1ST) ||
      (config[val_iZone]->GetTime_Marching() == DT_STEPPING_2ND)) {
    /*--- Update dual time solver on all mesh levels ---*/

    for (iMesh = 0; iMesh <= config[val_iZone]->GetnMGLevels(); iMesh++) {
      integration[val_iZone][val_iInst][FLOW_SOL]->SetDualTime_Solver(geometry[val_iZone][val_iInst][iMesh],
                                                                      solver[val_iZone][val_iInst][iMesh][FLOW_SOL],
                                                                      config[val_iZone], iMesh);
      integration[val_iZone][val_iInst][FLOW_SOL]->SetConvergence(false);
    }

    /*--- Update dual time solver for the dynamic mesh solver ---*/
    if (config[val_iZone]->GetDeform_Mesh()) {
      solver[val_iZone][val_iInst][MESH_0][MESH_SOL]->SetDualTime_Mesh();
    }

    /*--- Update dual time solver for the turbulence model ---*/

    if ((config[val_iZone]->GetKind_Solver() == RANS) || (config[val_iZone]->GetKind_Solver() == DISC_ADJ_RANS) ||
        (config[val_iZone]->GetKind_Solver() == INC_RANS) ||
        (config[val_iZone]->GetKind_Solver() == DISC_ADJ_INC_RANS)) {
      integration[val_iZone][val_iInst][TURB_SOL]->SetDualTime_Solver(geometry[val_iZone][val_iInst][MESH_0],
                                                                      solver[val_iZone][val_iInst][MESH_0][TURB_SOL],
                                                                      config[val_iZone], MESH_0);
      integration[val_iZone][val_iInst][TURB_SOL]->SetConvergence(false);
    }

    /*--- Update dual time solver for the transition model ---*/

    if (config[val_iZone]->GetKind_Trans_Model() == LM) {
      integration[val_iZone][val_iInst][TRANS_SOL]->SetDualTime_Solver(geometry[val_iZone][val_iInst][MESH_0],
                                                                       solver[val_iZone][val_iInst][MESH_0][TRANS_SOL],
                                                                       config[val_iZone], MESH_0);
      integration[val_iZone][val_iInst][TRANS_SOL]->SetConvergence(false);
    }
  }
}

bool CFluidIteration::Monitor(COutput* output, CIntegration**** integration, CGeometry**** geometry,
                              CSolver***** solver, CNumerics****** numerics, CConfig** config,
                              CSurfaceMovement** surface_movement, CVolumetricMovement*** grid_movement,
                              CFreeFormDefBox*** FFDBox, unsigned short val_iZone, unsigned short val_iInst) {
  bool StopCalc = false;

  StopTime = SU2_MPI::Wtime();

  UsedTime = StopTime - StartTime;

  if (config[val_iZone]->GetMultizone_Problem() || config[val_iZone]->GetSinglezone_Driver()) {
    output->SetHistory_Output(geometry[val_iZone][INST_0][MESH_0], solver[val_iZone][INST_0][MESH_0], config[val_iZone],
                              config[val_iZone]->GetTimeIter(), config[val_iZone]->GetOuterIter(),
                              config[val_iZone]->GetInnerIter());
  }

  /*--- If convergence was reached --*/
  StopCalc = output->GetConvergence();

  /* --- Checking convergence of Fixed CL mode to target CL, and perform finite differencing if needed  --*/

  if (config[val_iZone]->GetFixed_CL_Mode()) {
    StopCalc = MonitorFixed_CL(output, geometry[val_iZone][INST_0][MESH_0], solver[val_iZone][INST_0][MESH_0],
                               config[val_iZone]);
  }

  return StopCalc;
}

void CFluidIteration::Postprocess(COutput* output, CIntegration**** integration, CGeometry**** geometry,
                                  CSolver***** solver, CNumerics****** numerics, CConfig** config,
                                  CSurfaceMovement** surface_movement, CVolumetricMovement*** grid_movement,
                                  CFreeFormDefBox*** FFDBox, unsigned short val_iZone, unsigned short val_iInst) {
  /*--- Temporary: enable only for single-zone driver. This should be removed eventually when generalized. ---*/

  if (config[val_iZone]->GetSinglezone_Driver()) {
    /*--- Compute the tractions at the vertices ---*/
    solver[val_iZone][val_iInst][MESH_0][FLOW_SOL]->ComputeVertexTractions(geometry[val_iZone][val_iInst][MESH_0],
                                                                           config[val_iZone]);

    if (config[val_iZone]->GetKind_Solver() == DISC_ADJ_EULER ||
        config[val_iZone]->GetKind_Solver() == DISC_ADJ_NAVIER_STOKES ||
        config[val_iZone]->GetKind_Solver() == DISC_ADJ_RANS) {
      /*--- Read the target pressure ---*/

      //      if (config[val_iZone]->GetInvDesign_Cp() == YES)
      //        output->SetCp_InverseDesign(solver[val_iZone][val_iInst][MESH_0][FLOW_SOL],geometry[val_iZone][val_iInst][MESH_0],
      //        config[val_iZone], config[val_iZone]->GetExtIter());

      //      /*--- Read the target heat flux ---*/

      //      if (config[val_iZone]->GetInvDesign_HeatFlux() == YES)
      //        output->SetHeatFlux_InverseDesign(solver[val_iZone][val_iInst][MESH_0][FLOW_SOL],geometry[val_iZone][val_iInst][MESH_0],
      //        config[val_iZone], config[val_iZone]->GetExtIter());
    }
  }
}

void CFluidIteration::Solve(COutput* output, CIntegration**** integration, CGeometry**** geometry, CSolver***** solver,
                            CNumerics****** numerics, CConfig** config, CSurfaceMovement** surface_movement,
                            CVolumetricMovement*** grid_movement, CFreeFormDefBox*** FFDBox, unsigned short val_iZone,
                            unsigned short val_iInst) {
  /*--- Boolean to determine if we are running a static or dynamic case ---*/
  bool steady = !config[val_iZone]->GetTime_Domain();

  unsigned long Inner_Iter, nInner_Iter = config[val_iZone]->GetnInner_Iter();
  bool StopCalc = false;

  /*--- Synchronization point before a single solver iteration.
        Compute the wall clock time required. ---*/

  StartTime = SU2_MPI::Wtime();

  /*--- Preprocess the solver ---*/
  Preprocess(output, integration, geometry, solver, numerics, config, surface_movement, grid_movement, FFDBox,
             val_iZone, INST_0);

  /*--- For steady-state flow simulations, we need to loop over ExtIter for the number of time steps ---*/
  /*--- However, ExtIter is the number of FSI iterations, so nIntIter is used in this case ---*/

  for (Inner_Iter = 0; Inner_Iter < nInner_Iter; Inner_Iter++) {
    config[val_iZone]->SetInnerIter(Inner_Iter);

    /*--- Run a single iteration of the solver ---*/
    Iterate(output, integration, geometry, solver, numerics, config, surface_movement, grid_movement, FFDBox, val_iZone,
            INST_0);

    /*--- Monitor the pseudo-time ---*/
    StopCalc = Monitor(output, integration, geometry, solver, numerics, config, surface_movement, grid_movement, FFDBox,
                       val_iZone, INST_0);

    /*--- Output files at intermediate iterations if the problem is single zone ---*/

    if (singlezone && steady) {
      Output(output, geometry, solver, config, Inner_Iter, StopCalc, val_iZone, val_iInst);
    }

    /*--- If the iteration has converged, break the loop ---*/
    if (StopCalc) break;
  }

  if (multizone && steady) {
    Output(output, geometry, solver, config, config[val_iZone]->GetOuterIter(), StopCalc, val_iZone, val_iInst);

    /*--- Set the fluid convergence to false (to make sure outer subiterations converge) ---*/

    integration[val_iZone][INST_0][FLOW_SOL]->SetConvergence(false);
  }
}

void CFluidIteration::SetWind_GustField(CConfig* config, CGeometry** geometry, CSolver*** solver) {
  // The gust is imposed on the flow field via the grid velocities. This method called the Field Velocity Method is
  // described in the NASA TM–2012-217771 - Development, Verification and Use of Gust Modeling in the NASA Computational
  // Fluid Dynamics Code FUN3D the desired gust is prescribed as the negative of the grid velocity.

  // If a source term is included to account for the gust field, the method is described by Jones et al. as the Split
  // Velocity Method in Simulation of Airfoil Gust Responses Using Prescribed Velocities. In this routine the gust
  // derivatives needed for the source term are calculated when applicable. If the gust derivatives are zero the source
  // term is also zero. The source term itself is implemented in the class CSourceWindGust

  if (rank == MASTER_NODE) cout << endl << "Running simulation with a Wind Gust." << endl;
  unsigned short iDim, nDim = geometry[MESH_0]->GetnDim();  // We assume nDim = 2
  if (nDim != 2) {
    if (rank == MASTER_NODE) {
      cout << endl << "WARNING - Wind Gust capability is only verified for 2 dimensional simulations." << endl;
    }
  }

  /*--- Gust Parameters from config ---*/
  unsigned short Gust_Type = config->GetGust_Type();
  su2double xbegin = config->GetGust_Begin_Loc();   // Location at which the gust begins.
  su2double L = config->GetGust_WaveLength();       // Gust size
  su2double tbegin = config->GetGust_Begin_Time();  // Physical time at which the gust begins.
  su2double gust_amp = config->GetGust_Ampl();      // Gust amplitude
  su2double n = config->GetGust_Periods();          // Number of gust periods
  unsigned short GustDir = config->GetGust_Dir();   // Gust direction

  /*--- Variables needed to compute the gust ---*/
  unsigned short Kind_Grid_Movement = config->GetKind_GridMovement();
  unsigned long iPoint;
  unsigned short iMGlevel, nMGlevel = config->GetnMGLevels();

  su2double x, y, x_gust, dgust_dx, dgust_dy, dgust_dt;
  su2double *Gust, *GridVel, *NewGridVel, *GustDer;

  su2double Physical_dt = config->GetDelta_UnstTime();
  unsigned long TimeIter = config->GetTimeIter();
  su2double Physical_t = TimeIter * Physical_dt;

  su2double Uinf = solver[MESH_0][FLOW_SOL]->GetVelocity_Inf(0);  // Assumption gust moves at infinity velocity

  Gust = new su2double[nDim];
  NewGridVel = new su2double[nDim];
  for (iDim = 0; iDim < nDim; iDim++) {
    Gust[iDim] = 0.0;
    NewGridVel[iDim] = 0.0;
  }

  GustDer = new su2double[3];
  for (unsigned short i = 0; i < 3; i++) {
    GustDer[i] = 0.0;
  }

  // Vortex variables
  unsigned long nVortex = 0;
  vector<su2double> x0, y0, vort_strenth, r_core;  // vortex is positive in clockwise direction.
  if (Gust_Type == VORTEX) {
    InitializeVortexDistribution(nVortex, x0, y0, vort_strenth, r_core);
  }

  /*--- Check to make sure gust lenght is not zero or negative (vortex gust doesn't use this). ---*/
  if (L <= 0.0 && Gust_Type != VORTEX) {
    SU2_MPI::Error("The gust length needs to be positive", CURRENT_FUNCTION);
  }

  /*--- Loop over all multigrid levels ---*/

  for (iMGlevel = 0; iMGlevel <= nMGlevel; iMGlevel++) {
    /*--- Loop over each node in the volume mesh ---*/

    for (iPoint = 0; iPoint < geometry[iMGlevel]->GetnPoint(); iPoint++) {
      /*--- Reset the Grid Velocity to zero if there is no grid movement ---*/
      if (Kind_Grid_Movement == GUST) {
        for (iDim = 0; iDim < nDim; iDim++) geometry[iMGlevel]->nodes->SetGridVel(iPoint, iDim, 0.0);
      }

      /*--- initialize the gust and derivatives to zero everywhere ---*/

      for (iDim = 0; iDim < nDim; iDim++) {
        Gust[iDim] = 0.0;
      }
      dgust_dx = 0.0;
      dgust_dy = 0.0;
      dgust_dt = 0.0;

      /*--- Begin applying the gust ---*/

      if (Physical_t >= tbegin) {
        x = geometry[iMGlevel]->nodes->GetCoord(iPoint)[0];  // x-location of the node.
        y = geometry[iMGlevel]->nodes->GetCoord(iPoint)[1];  // y-location of the node.

        // Gust coordinate
        x_gust = (x - xbegin - Uinf * (Physical_t - tbegin)) / L;

        /*--- Calculate the specified gust ---*/
        switch (Gust_Type) {
          case TOP_HAT:
            // Check if we are in the region where the gust is active
            if (x_gust > 0 && x_gust < n) {
              Gust[GustDir] = gust_amp;
              // Still need to put the gust derivatives. Think about this.
            }
            break;

          case SINE:
            // Check if we are in the region where the gust is active
            if (x_gust > 0 && x_gust < n) {
              Gust[GustDir] = gust_amp * (sin(2 * PI_NUMBER * x_gust));

              // Gust derivatives
              // dgust_dx = gust_amp*2*PI_NUMBER*(cos(2*PI_NUMBER*x_gust))/L;
              // dgust_dy = 0;
              // dgust_dt = gust_amp*2*PI_NUMBER*(cos(2*PI_NUMBER*x_gust))*(-Uinf)/L;
            }
            break;

          case ONE_M_COSINE:
            // Check if we are in the region where the gust is active
            if (x_gust > 0 && x_gust < n) {
              Gust[GustDir] = gust_amp * (1 - cos(2 * PI_NUMBER * x_gust));

              // Gust derivatives
              // dgust_dx = gust_amp*2*PI_NUMBER*(sin(2*PI_NUMBER*x_gust))/L;
              // dgust_dy = 0;
              // dgust_dt = gust_amp*2*PI_NUMBER*(sin(2*PI_NUMBER*x_gust))*(-Uinf)/L;
            }
            break;

          case EOG:
            // Check if we are in the region where the gust is active
            if (x_gust > 0 && x_gust < n) {
              Gust[GustDir] = -0.37 * gust_amp * sin(3 * PI_NUMBER * x_gust) * (1 - cos(2 * PI_NUMBER * x_gust));
            }
            break;

          case VORTEX:

            /*--- Use vortex distribution ---*/
            // Algebraic vortex equation.
            for (unsigned long i = 0; i < nVortex; i++) {
              su2double r2 = pow(x - (x0[i] + Uinf * (Physical_t - tbegin)), 2) + pow(y - y0[i], 2);
              su2double r = sqrt(r2);
              su2double v_theta = vort_strenth[i] / (2 * PI_NUMBER) * r / (r2 + pow(r_core[i], 2));
              Gust[0] = Gust[0] + v_theta * (y - y0[i]) / r;
              Gust[1] = Gust[1] - v_theta * (x - (x0[i] + Uinf * (Physical_t - tbegin))) / r;
            }
            break;

          case NONE:
          default:

            /*--- There is no wind gust specified. ---*/
            if (rank == MASTER_NODE) {
              cout << "No wind gust specified." << endl;
            }
            break;
        }
      }

      /*--- Set the Wind Gust, Wind Gust Derivatives and the Grid Velocities ---*/

      GustDer[0] = dgust_dx;
      GustDer[1] = dgust_dy;
      GustDer[2] = dgust_dt;

      solver[iMGlevel][FLOW_SOL]->GetNodes()->SetWindGust(iPoint, Gust);
      solver[iMGlevel][FLOW_SOL]->GetNodes()->SetWindGustDer(iPoint, GustDer);

      GridVel = geometry[iMGlevel]->nodes->GetGridVel(iPoint);

      /*--- Store new grid velocity ---*/

      for (iDim = 0; iDim < nDim; iDim++) {
        NewGridVel[iDim] = GridVel[iDim] - Gust[iDim];
        geometry[iMGlevel]->nodes->SetGridVel(iPoint, iDim, NewGridVel[iDim]);
      }
    }
  }

  delete[] Gust;
  delete[] GustDer;
  delete[] NewGridVel;
}

void CFluidIteration::InitializeVortexDistribution(unsigned long& nVortex, vector<su2double>& x0, vector<su2double>& y0,
                                                   vector<su2double>& vort_strength, vector<su2double>& r_core) {
  /*--- Read in Vortex Distribution ---*/
  std::string line;
  std::ifstream file;
  su2double x_temp, y_temp, vort_strength_temp, r_core_temp;
  file.open("vortex_distribution.txt");
  /*--- In case there is no vortex file ---*/
  if (file.fail()) {
    SU2_MPI::Error("There is no vortex data file!!", CURRENT_FUNCTION);
  }

  // Ignore line containing the header
  getline(file, line);
  // Read in the information of the vortices (xloc, yloc, lambda(strength), eta(size, gradient))
  while (file.good()) {
    getline(file, line);
    std::stringstream ss(line);
    if (line.size() != 0) {  // ignore blank lines if they exist.
      ss >> x_temp;
      ss >> y_temp;
      ss >> vort_strength_temp;
      ss >> r_core_temp;
      x0.push_back(x_temp);
      y0.push_back(y_temp);
      vort_strength.push_back(vort_strength_temp);
      r_core.push_back(r_core_temp);
    }
  }
  file.close();
  // number of vortices
  nVortex = x0.size();
}

bool CFluidIteration::MonitorFixed_CL(COutput *output, CGeometry *geometry, CSolver **solver, CConfig *config) {

  CSolver* flow_solver= solver[FLOW_SOL];

  bool fixed_cl_convergence = flow_solver->FixedCL_Convergence(config, output->GetConvergence());

  /* --- If Fixed CL mode has ended and Finite Differencing has started: --- */

  if (flow_solver->GetStart_AoA_FD() && flow_solver->GetIter_Update_AoA() == config->GetInnerIter()){

    /* --- Print convergence history and volume files since fixed CL mode has converged--- */
    if (rank == MASTER_NODE) output->PrintConvergenceSummary();

    output->SetResult_Files(geometry, config, solver,
                            config->GetInnerIter(), true);

    /* --- Set finite difference mode in config (disables output) --- */
    config->SetFinite_Difference_Mode(true);
  }

  /* --- Set convergence based on fixed CL convergence  --- */
  return fixed_cl_convergence;
}
