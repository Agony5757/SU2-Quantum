/*!
 * \file CDiscAdjMultizoneDriver.cpp
 * \brief The main subroutines for driving adjoint multi-zone problems
 * \author O. Burghardt, T. Albring, R. Sanchez
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

#include "../../include/drivers/CDiscAdjMultizoneDriver.hpp"
#include "../../include/solvers/CFEASolver.hpp"
#include "../../include/output/COutputFactory.hpp"
#include "../../include/output/COutputLegacy.hpp"
#include "../../include/output/COutput.hpp"
#include "../../include/iteration/CIterationFactory.hpp"
#include "../../../Common/include/toolboxes/CQuasiNewtonInvLeastSquares.hpp"

CDiscAdjMultizoneDriver::CDiscAdjMultizoneDriver(char* confFile,
                                                 unsigned short val_nZone,
                                                 SU2_Comm MPICommunicator)

                        : CMultizoneDriver(confFile, val_nZone, MPICommunicator) {

  retape = !config_container[ZONE_0]->GetFull_Tape();

  RecordingState = NONE;

  direct_nInst.resize(nZone,1);
  nInnerIter.resize(nZone);

  for (iZone = 0; iZone < nZone; iZone++)
    nInnerIter[iZone] = config_container[iZone]->GetnInner_Iter();

  Has_Deformation.resize(nZone) = false;

  direct_iteration = new CIteration**[nZone];
  direct_output = new COutput*[nZone];

  for (iZone = 0; iZone < nZone; iZone++) {

    /*--- Instantiate a direct iteration for each zone. ---*/

    direct_iteration[iZone] = new CIteration*[direct_nInst[iZone]];

    for(iInst = 0; iInst < direct_nInst[iZone]; iInst++) {

      switch (config_container[iZone]->GetKind_Solver()) {

        case DISC_ADJ_EULER: case DISC_ADJ_NAVIER_STOKES: case DISC_ADJ_RANS:
          direct_iteration[iZone][iInst] = CIterationFactory::CreateIteration(EULER, config_container[iZone]);
          break;
        case DISC_ADJ_INC_EULER: case DISC_ADJ_INC_NAVIER_STOKES: case DISC_ADJ_INC_RANS:
          direct_iteration[iZone][iInst] = CIterationFactory::CreateIteration(INC_EULER, config_container[iZone]);
          break;
        case DISC_ADJ_HEAT:
          direct_iteration[iZone][iInst] = CIterationFactory::CreateIteration(HEAT_EQUATION, config_container[iZone]);
          break;
        case DISC_ADJ_FEM:
          direct_iteration[iZone][iInst] = CIterationFactory::CreateIteration(FEM_ELASTICITY, config_container[iZone]);
          break;
        default:
          SU2_MPI::Error("There is no discrete adjoint functionality for one of the specified solvers yet.",
                         CURRENT_FUNCTION);
      }
    }

    /*--- Instantiate a direct output to get the results of each direct zone. ---*/

    switch (config_container[iZone]->GetKind_Solver()) {

      case DISC_ADJ_EULER: case DISC_ADJ_NAVIER_STOKES: case DISC_ADJ_RANS:
        direct_output[iZone] = COutputFactory::CreateOutput(EULER, config_container[iZone], nDim);
        break;
      case DISC_ADJ_INC_EULER: case DISC_ADJ_INC_NAVIER_STOKES: case DISC_ADJ_INC_RANS:
        direct_output[iZone] = COutputFactory::CreateOutput(INC_EULER, config_container[iZone], nDim);
        break;
      case DISC_ADJ_HEAT:
        direct_output[iZone] = COutputFactory::CreateOutput(HEAT_EQUATION, config_container[iZone], nDim);
        break;
      case DISC_ADJ_FEM:
        direct_output[iZone] = COutputFactory::CreateOutput(FEM_ELASTICITY, config_container[iZone], nDim);
        break;
      default:
        direct_output[iZone] = nullptr;
        break;
    }

    direct_output[iZone]->PreprocessHistoryOutput(config_container[iZone], false);

  }

}

CDiscAdjMultizoneDriver::~CDiscAdjMultizoneDriver(){

  for (iZone = 0; iZone < nZone; iZone++){
    for (iInst = 0; iInst < direct_nInst[iZone]; iInst++){
      delete direct_iteration[iZone][iInst];
    }
    delete [] direct_iteration[iZone];
    delete direct_output[iZone];
  }

  delete[] direct_iteration;
  delete[] direct_output;

}

void CDiscAdjMultizoneDriver::StartSolver() {

  /*--- Main external loop of the solver. Runs for the number of time steps required. ---*/

  if (rank == MASTER_NODE) {
    cout <<"\n------------------------------ Begin Solver -----------------------------" << endl;

    cout << "\nSimulation Run using the Discrete Adjoint Multizone Driver" << endl;

    if (driver_config->GetTime_Domain())
      SU2_MPI::Error("The discrete adjoint multizone driver is not ready for unsteady computations yet.",
                     CURRENT_FUNCTION);
  }

  for (iZone = 0; iZone < nZone; iZone++){

    /*--- Set the value of the external iteration to TimeIter. -------------------------------------*/
    /*--- TODO: This should be generalised for an homogeneous criteria throughout the code. --------*/
    config_container[iZone]->SetTimeIter(0);

  }

  /*--- Size and initialize the matrix of cross-terms. ---*/

  InitializeCrossTerms();

  /*--- We directly start the (steady-state) discrete adjoint computation. ---*/

  Run();

  /*--- Output the solution in files. ---*/

  Output(TimeIter);

}

void CDiscAdjMultizoneDriver::Run() {

  unsigned long wrt_sol_freq = 9999;
  unsigned long nOuterIter = driver_config->GetnOuter_Iter();
  vector<CQuasiNewtonInvLeastSquares<passivedouble> > fixPtCorrector(nZone);

  for (iZone = 0; iZone < nZone; iZone++) {

    wrt_sol_freq = min(wrt_sol_freq, config_container[iZone]->GetVolume_Wrt_Freq());

    iteration_container[iZone][INST_0]->Preprocess(output_container[iZone], integration_container, geometry_container,
                                                   solver_container, numerics_container, config_container, surface_movement,
                                                   grid_movement, FFDBox, iZone, INST_0);

    /*--- Set BGS_Solution_k to Solution, this is needed to restart
     *    correctly as the OF gradient will overwrite the solution. ---*/

    Set_BGSSolution_k_To_Solution(iZone);

    /*--- Prepare quasi-Newton drivers. ---*/

    if (config_container[iZone]->GetnQuasiNewtonSamples() > 1) {
      fixPtCorrector[iZone].resize(config_container[iZone]->GetnQuasiNewtonSamples(),
                                   geometry_container[iZone][INST_0][MESH_0]->GetnPoint(),
                                   GetTotalNumberOfVariables(iZone, true),
                                   geometry_container[iZone][INST_0][MESH_0]->GetnPointDomain());
    }
  }

  /*--- Evaluate the objective function gradient w.r.t. the solutions of all zones. ---*/

  SetRecording(NONE, Kind_Tape::OBJECTIVE_FUNCTION_TAPE, ZONE_0);
  SetRecording(SOLUTION_VARIABLES, Kind_Tape::OBJECTIVE_FUNCTION_TAPE, ZONE_0);
  RecordingState = NONE;

  AD::ClearAdjoints();
  SetAdj_ObjFunction();
  AD::ComputeAdjoint(OBJECTIVE_FUNCTION, START);

  /*--- Initialize External with the objective function gradient. ---*/

  for (iZone = 0; iZone < nZone; iZone++) {

    iteration_container[iZone][INST_0]->Iterate(output_container[iZone], integration_container, geometry_container,
                                                solver_container, numerics_container, config_container,
                                                surface_movement, grid_movement, FFDBox, iZone, INST_0);
    Add_Solution_To_External(iZone);
  }

  /*--- Loop over the number of outer iterations. ---*/

  for (unsigned long iOuterIter = 0, StopCalc = false; !StopCalc; iOuterIter++) {

    driver_config->SetOuterIter(iOuterIter);

    for (iZone = 0; iZone < nZone; iZone++)
      config_container[iZone]->SetOuterIter(iOuterIter);

    /*--- For the adjoint iteration we need the derivatives of the iteration function with
     *    respect to the state (and possibly the mesh coordinate) variables.
     *    Since these derivatives do not change in the steady state case we only have to record
     *    if the current recording is different from them.
     *
     *    To set the tape appropriately, the following recording methods are provided:
     *    (1) NONE: All information from a previous recording is removed.
     *    (2) SOLUTION_VARIABLES: State variables of all solvers in a zone as input.
     *    (3) MESH_COORDS / MESH_DEFORM: Mesh coordinates as input.
     *    (4) SOLUTION_AND_MESH: Mesh coordinates and state variables as input.
     *
     *    By default, all (state and mesh coordinate variables) will be declared as output,
     *    since it does not change the computational effort. ---*/


    /*--- If we want to set up zone-specific tapes (retape), we do not need to record
     *    here. Otherwise, the whole tape of a coupled run will be created. ---*/

    if (!retape && (RecordingState != SOLUTION_VARIABLES)) {
      SetRecording(NONE, Kind_Tape::FULL_TAPE, ZONE_0);
      SetRecording(SOLUTION_VARIABLES, Kind_Tape::FULL_TAPE, ZONE_0);
    }

    /*-- Start loop over zones. ---*/

    for (iZone = 0; iZone < nZone; iZone++) {

      config_container[iZone]->Set_StartTime(SU2_MPI::Wtime());

      if (retape) {
        SetRecording(NONE, Kind_Tape::FULL_TAPE, ZONE_0);
        SetRecording(SOLUTION_VARIABLES, Kind_Tape::ZONE_SPECIFIC_TAPE, iZone);
      }

      /*--- Start inner iterations from where we stopped in previous outer iteration. ---*/

      Set_Solution_To_BGSSolution_k(iZone);

      /*--- Inner loop to allow for multiple adjoint updates with respect to solvers in iZone. ---*/

      bool eval_transfer = false;
      const bool restart = config_container[iZone]->GetRestart();
      const bool no_restart = (iOuterIter > 0) || !restart;

      /*--- Reset QN driver for new inner iterations. ---*/

      if (fixPtCorrector[iZone].size()) {
        fixPtCorrector[iZone].reset();
        if(restart && (iOuterIter==1)) GetAllSolutions(iZone, true, fixPtCorrector[iZone]);
      }

      for (unsigned long iInnerIter = 0; iInnerIter < nInnerIter[iZone]; iInnerIter++) {

        config_container[iZone]->SetInnerIter(iInnerIter);

        /*--- Add off-diagonal contribution (including the OF gradient) to Solution. ---*/

        if (no_restart || (iInnerIter > 0)) {
          Add_External_To_Solution(iZone);
        }
        else {
          /*--- If we restarted, Solution already has all contributions,
           *    we run only one inner iter to compute the cross terms. ---*/
          eval_transfer = true;
        }

        /*--- Evaluate the tape section belonging to solvers in iZone.
         *    Only evaluate TRANSFER terms on the last iteration or after convergence. ---*/

        eval_transfer = eval_transfer || (iInnerIter == nInnerIter[iZone]-1);

        ComputeAdjoints(iZone, eval_transfer);

        /*--- Extracting adjoints for solvers in iZone w.r.t. to outputs in iZone (diagonal part). ---*/

        iteration_container[iZone][INST_0]->Iterate(output_container[iZone], integration_container, geometry_container,
                                                    solver_container, numerics_container, config_container,
                                                    surface_movement, grid_movement, FFDBox, iZone, INST_0);

        /*--- Use QN driver to improve the solution. ---*/

        if (fixPtCorrector[iZone].size()) {
          GetAllSolutions(iZone, true, fixPtCorrector[iZone].FPresult());
          fixPtCorrector[iZone].compute();
          if(iInnerIter) SetAllSolutions(iZone, true, fixPtCorrector[iZone]);
        }

        /*--- This is done explicitly here for multizone cases, only in inner iterations and not when
         *    extracting cross terms so that the adjoint residuals in each zone still make sense. ---*/

        Set_SolutionOld_To_Solution(iZone);

        /*--- Print out the convergence data to screen and history file. ---*/

        bool converged = iteration_container[iZone][INST_0]->Monitor(output_container[iZone], integration_container,
                                                    geometry_container, solver_container, numerics_container,
                                                    config_container, surface_movement, grid_movement, FFDBox, iZone, INST_0);
        if (eval_transfer) break;
        eval_transfer = converged;

      }

      /*--- Off-diagonal (coupling term) BGS update. ---*/

      for (unsigned short jZone = 0; jZone < nZone; jZone++) {

        if (jZone != iZone && interface_container[jZone][iZone] != nullptr) {

          /*--- Extracting adjoints for solvers in jZone w.r.t. to the output of all solvers in iZone,
           *    that is, for the cases iZone != jZone we are evaluating cross derivatives between zones. ---*/

          config_container[jZone]->SetInnerIter(0);
          iteration_container[jZone][INST_0]->Iterate(output_container[jZone], integration_container, geometry_container,
                                                      solver_container, numerics_container, config_container,
                                                      surface_movement, grid_movement, FFDBox, jZone, INST_0);

          /*--- Extract the cross-term performing a relaxed update of it and of the sum (External) for jZone. ---*/

          Update_Cross_Term(iZone, jZone);
        }
      }

      /*--- Compute residual from Solution and Solution_BGS_k and update the latter. ---*/

      SetResidual_BGS(iZone);

    }

    /*--- Set the multizone output. ---*/

    driver_output->SetMultizoneHistory_Output(output_container, config_container, driver_config, 0, iOuterIter);

    /*--- Check for convergence. ---*/

    StopCalc = driver_output->GetConvergence() || (iOuterIter == nOuterIter-1);

    /*--- Clear the stored adjoint information to be ready for a new evaluation. ---*/

    AD::ClearAdjoints();

    /*--- Compute the geometrical sensitivities and write them to file. ---*/

    bool checkSensitivity = StopCalc || ((iOuterIter % wrt_sol_freq == 0) && (iOuterIter != 0));

    if (checkSensitivity)
      EvaluateSensitivities(iOuterIter, StopCalc);
  }
}

void CDiscAdjMultizoneDriver::EvaluateSensitivities(unsigned long iOuterIter, bool StopCalc) {

  /*--- SetRecording stores the computational graph on one iteration of the direct problem. Calling it with NONE
   *    as argument ensures that all information from a previous recording is removed. ---*/

  SetRecording(NONE, Kind_Tape::FULL_TAPE, ZONE_0);

  /*--- Store the computational graph of one direct iteration with the mesh coordinates as input. ---*/

  SetRecording(MESH_COORDS, Kind_Tape::FULL_TAPE, ZONE_0);

  /*--- Initialize the adjoint of the output variables of the iteration with the adjoint solution
   *    of the current iteration. The values are passed to the AD tool. ---*/

  for (iZone = 0; iZone < nZone; iZone++) {

    Set_Solution_To_BGSSolution_k(iZone);

    Add_External_To_Solution(iZone);

    iteration_container[iZone][INST_0]->InitializeAdjoint(solver_container, geometry_container,
                                                          config_container, iZone, INST_0);
  }

  /*--- Initialize the adjoint of the objective function with 1.0. ---*/

  SetAdj_ObjFunction();

  /*--- Interpret the stored information by calling the corresponding routine of the AD tool. ---*/

  AD::ComputeAdjoint();

  /*--- Extract the computed sensitivity values. ---*/

  for (iZone = 0; iZone < nZone; iZone++) {

    auto config = config_container[iZone];
    auto solvers = solver_container[iZone][INST_0][MESH_0];
    auto geometry = geometry_container[iZone][INST_0][MESH_0];

    switch (config_container[iZone]->GetKind_Solver()) {

      case DISC_ADJ_EULER:     case DISC_ADJ_NAVIER_STOKES:     case DISC_ADJ_RANS:
      case DISC_ADJ_INC_EULER: case DISC_ADJ_INC_NAVIER_STOKES: case DISC_ADJ_INC_RANS:

        if(Has_Deformation(iZone)) {
          solvers[ADJMESH_SOL]->SetSensitivity(geometry, solvers, config);
        } else {
          solvers[ADJFLOW_SOL]->SetSensitivity(geometry, solvers, config);
        }
        break;

      case DISC_ADJ_HEAT:

        solvers[ADJHEAT_SOL]->SetSensitivity(geometry, solvers, config);
        break;

      case DISC_ADJ_FEM:

        solvers[ADJFEA_SOL]->SetSensitivity(geometry, solvers, config);
        break;

      default:
        if (rank == MASTER_NODE)
          cout << "WARNING: Sensitivities not set for one of the specified discrete adjoint solvers!" << endl;
        break;
    }
  }

  /*--- Clear the stored adjoint information to be ready for a new evaluation. ---*/

  AD::ClearAdjoints();

  for (iZone = 0; iZone < nZone; iZone++) {

    output_container[iZone]->SetResult_Files(geometry_container[iZone][INST_0][MESH_0],
                                             config_container[iZone],
                                             solver_container[iZone][INST_0][MESH_0], iOuterIter, StopCalc);
  }
}

void CDiscAdjMultizoneDriver::SetRecording(unsigned short kind_recording, Kind_Tape tape_type, unsigned short record_zone) {

  AD::Reset();

  /*--- Prepare for recording by resetting the flow solution to the initial converged solution---*/

  for(iZone = 0; iZone < nZone; iZone++) {
    for (unsigned short iSol=0; iSol < MAX_SOLS; iSol++) {
      auto solver = solver_container[iZone][INST_0][MESH_0][iSol];
      if (solver && solver->GetAdjoint()) {
        for (unsigned short iMesh = 0; iMesh <= config_container[iZone]->GetnMGLevels(); iMesh++) {
          solver->SetRecording(geometry_container[iZone][INST_0][iMesh], config_container[iZone]);
        }
      }
    }
  }

  if (rank == MASTER_NODE) {
    cout << "\n-------------------------------------------------------------------------\n";
    switch(kind_recording) {
    case NONE:        cout << "Clearing the computational graph." << endl; break;
    case MESH_COORDS: cout << "Storing computational graph wrt MESH COORDINATES." << endl; break;
    case SOLUTION_VARIABLES:   cout << "Storing computational graph wrt CONSERVATIVE VARIABLES." << endl; break;
    }
  }

  /*--- Enable recording and register input of the flow iteration (conservative variables or node coordinates) --- */

  if(kind_recording != NONE) {

    AD::StartRecording();

    AD::Push_TapePosition(); /// START

    for (iZone = 0; iZone < nZone; iZone++) {

      /*--- In multi-physics, MESH_COORDS is an umbrella term for "geometric sensitivities",
       *    if a zone has mesh deformation its recording type needs to change to MESH_DEFORM
       *    as those sensitivities are managed by the adjoint mesh solver instead. ---*/

      unsigned short type_recording = kind_recording;

      if (Has_Deformation(iZone) && (kind_recording == MESH_COORDS)) {
        type_recording = MESH_DEFORM;
      }

      iteration_container[iZone][INST_0]->RegisterInput(solver_container, geometry_container,
                                                        config_container, iZone, INST_0, type_recording);
    }
  }

  AD::Push_TapePosition(); /// REGISTERED

  for (iZone = 0; iZone < nZone; iZone++) {

    iteration_container[iZone][INST_0]->SetDependencies(solver_container, geometry_container, numerics_container,
                                                        config_container, iZone, INST_0, kind_recording);
  }

  AD::Push_TapePosition(); /// DEPENDENCIES

  /*--- Extract the objective function and store it.
   *    It is necessary to include data transfer and mesh updates in this section as some functions
   *    computed in one zone depend explicitly on the variables of others through that path. --- */

  if ((tape_type == Kind_Tape::OBJECTIVE_FUNCTION_TAPE) || (kind_recording == MESH_COORDS)) {
    HandleDataTransfer();
  }

  SetObjFunction(kind_recording);

  AD::Push_TapePosition(); /// OBJECTIVE_FUNCTION

  if (tape_type != Kind_Tape::OBJECTIVE_FUNCTION_TAPE) {

    /*--- We do the communication here to not differentiate wrt updated boundary data.
     *    For recording w.r.t. mesh coordinates the transfer was included before the
     *    objective function, so we do not repeat it here. ---*/

    if (kind_recording != MESH_COORDS) {
      HandleDataTransfer();
    }

    AD::Push_TapePosition(); /// TRANSFER

    for(iZone = 0; iZone < nZone; iZone++) {

      AD::Push_TapePosition(); /// enter_zone

      if (tape_type == Kind_Tape::ZONE_SPECIFIC_TAPE) {
        if (iZone == record_zone) {
          DirectIteration(iZone, kind_recording);
        }
      }
      else {
        DirectIteration(iZone, kind_recording);
      }

      iteration_container[iZone][INST_0]->RegisterOutput(solver_container, geometry_container,
                                                         config_container, output_container[iZone], iZone, INST_0);

      AD::Push_TapePosition(); /// leave_zone
    }
  }

  if (rank == MASTER_NODE) {
    if(kind_recording != NONE && config_container[record_zone]->GetWrt_AD_Statistics()) {
      AD::PrintStatistics();
    }
    cout << "-------------------------------------------------------------------------\n" << endl;
  }

  AD::StopRecording();

  RecordingState = kind_recording;
}

void CDiscAdjMultizoneDriver::DirectIteration(unsigned short iZone, unsigned short kind_recording) {

  /*--- Do one iteration of the direct solver ---*/
  direct_iteration[iZone][INST_0]->Preprocess(output_container[iZone], integration_container, geometry_container,
                                              solver_container, numerics_container, config_container,
                                              surface_movement, grid_movement, FFDBox, iZone, INST_0);

  /*--- Iterate the zone as a block a single time ---*/
  direct_iteration[iZone][INST_0]->Iterate(output_container[iZone], integration_container, geometry_container,
                                           solver_container, numerics_container, config_container,
                                           surface_movement, grid_movement, FFDBox, iZone, INST_0);

  /*--- Print residuals in the first iteration ---*/

  if (rank == MASTER_NODE && kind_recording == SOLUTION_VARIABLES) {

    auto solvers = solver_container[iZone][INST_0][MESH_0];

    switch (config_container[iZone]->GetKind_Solver()) {

      case DISC_ADJ_EULER:     case DISC_ADJ_NAVIER_STOKES:
      case DISC_ADJ_INC_EULER: case DISC_ADJ_INC_NAVIER_STOKES:
        cout << " Zone " << iZone << " (flow)       - log10[U(0)]    : "
             << log10(solvers[FLOW_SOL]->GetRes_RMS(0)) << endl;
        if (config_container[iZone]->AddRadiation()) {

          cout << " Zone " << iZone << " (radiation)  - log10[Rad(0)]  : "
               << log10(solvers[RAD_SOL]->GetRes_RMS(0)) << endl;
        }
        break;

      case DISC_ADJ_RANS: case DISC_ADJ_INC_RANS:
        cout << " Zone " << iZone << " (flow)       - log10[U(0)]    : "
             << log10(solvers[FLOW_SOL]->GetRes_RMS(0)) << endl;

        if (!config_container[iZone]->GetFrozen_Visc_Disc()) {

          cout << " Zone " << iZone << " (turbulence) - log10[Turb(0)] : "
               << log10(solvers[TURB_SOL]->GetRes_RMS(0)) << endl;
        }
        if (config_container[iZone]->AddRadiation()) {

          cout << " Zone " << iZone << " (radiation)  - log10[Rad(0)]  : "
               << log10(solvers[RAD_SOL]->GetRes_RMS(0)) << endl;
        }
        break;

      case DISC_ADJ_HEAT:
        cout << " Zone " << iZone << " (heat)       - log10[Heat(0)] : "
             << log10(solvers[HEAT_SOL]->GetRes_RMS(0)) << endl;
        break;

      case DISC_ADJ_FEM:
        cout << " Zone " << iZone << " (structure)  - ";
        if(config_container[iZone]->GetGeometricConditions() == LARGE_DEFORMATIONS)
          cout << "log10[RTOL-A]  : " << log10(solvers[FEA_SOL]->GetRes_FEM(1)) << endl;
        else
          cout << "log10[RMS Ux]  : " << log10(solvers[FEA_SOL]->GetRes_RMS(0)) << endl;
        break;

      default:
        break;
    }
  }
}

void CDiscAdjMultizoneDriver::SetObjFunction(unsigned short kind_recording) {

  ObjFunc = 0.0;
  su2double Weight_ObjFunc;

  unsigned short iMarker_Analyze, nMarker_Analyze;

  /*--- Call objective function calculations. ---*/

  for (iZone = 0; iZone < nZone; iZone++) {

    auto config = config_container[iZone];
    auto solvers = solver_container[iZone][INST_0][MESH_0];
    auto geometry = geometry_container[iZone][INST_0][MESH_0];

    switch (config->GetKind_Solver()) {

      case DISC_ADJ_EULER:     case DISC_ADJ_NAVIER_STOKES:     case DISC_ADJ_RANS:
      case DISC_ADJ_INC_EULER: case DISC_ADJ_INC_NAVIER_STOKES: case DISC_ADJ_INC_RANS:

        solvers[FLOW_SOL]->Pressure_Forces(geometry, config);
        solvers[FLOW_SOL]->Momentum_Forces(geometry, config);
        solvers[FLOW_SOL]->Friction_Forces(geometry, config);

        if(config->GetWeakly_Coupled_Heat()) {
          solvers[HEAT_SOL]->Heat_Fluxes(geometry, solvers, config);
        }
        solvers[FLOW_SOL]->Evaluate_ObjFunc(config);
        break;
      case DISC_ADJ_HEAT:
        solvers[HEAT_SOL]->Heat_Fluxes(geometry, solvers, config);
        break;
    }

    direct_output[iZone]->SetHistory_Output(geometry, solvers, config);
  }

  /*--- Extract objective function values. ---*/

  for (iZone = 0; iZone < nZone; iZone++) {

    auto config = config_container[iZone];
    auto solvers = solver_container[iZone][INST_0][MESH_0];
    auto geometry = geometry_container[iZone][INST_0][MESH_0];

    nMarker_Analyze = config->GetnMarker_Analyze();

    for (iMarker_Analyze = 0; iMarker_Analyze < nMarker_Analyze; iMarker_Analyze++) {

      Weight_ObjFunc = config->GetWeight_ObjFunc(iMarker_Analyze);

      switch (config->GetKind_Solver()) {

        case DISC_ADJ_EULER: case DISC_ADJ_NAVIER_STOKES: case DISC_ADJ_RANS:
          // per-surface output to be added soon
          break;
        case HEAT_EQUATION: case DISC_ADJ_HEAT:
          // per-surface output to be added soon
          break;
        default:
          break;
      }
    }

    /*--- Not-per-surface objective functions (shall not be included above) ---*/

    Weight_ObjFunc = config->GetWeight_ObjFunc(0);

    bool ObjectiveNotCovered = false;

    switch (config->GetKind_Solver()) {

      case DISC_ADJ_EULER:     case DISC_ADJ_NAVIER_STOKES:     case DISC_ADJ_RANS:
      case DISC_ADJ_INC_EULER: case DISC_ADJ_INC_NAVIER_STOKES: case DISC_ADJ_INC_RANS:
      {
        string FieldName;

        switch (config->GetKind_ObjFunc()) {

          // Aerodynamic coefficients

          case DRAG_COEFFICIENT:      FieldName = "DRAG";       break;
          case LIFT_COEFFICIENT:      FieldName = "LIFT";       break;
          case SIDEFORCE_COEFFICIENT: FieldName = "SIDEFORCE";  break;
          case EFFICIENCY:            FieldName = "EFFICIENCY"; break;
          case MOMENT_X_COEFFICIENT:  FieldName = "MOMENT-X";   break;
          case MOMENT_Y_COEFFICIENT:  FieldName = "MOMENT-Y";   break;
          case MOMENT_Z_COEFFICIENT:  FieldName = "MOMENT-Z";   break;
          case FORCE_X_COEFFICIENT:   FieldName = "FORCE-X";    break;
          case FORCE_Y_COEFFICIENT:   FieldName = "FORCE-Y";    break;
          case FORCE_Z_COEFFICIENT:   FieldName = "FORCE-Z";    break;

          // Other surface-related output values

          case SURFACE_MASSFLOW:            FieldName = "AVG_MASSFLOW";              break;
          case SURFACE_MACH:                FieldName = "AVG_MACH";                  break;
          case SURFACE_UNIFORMITY:          FieldName = "UNIFORMITY";                break;
          case SURFACE_SECONDARY:           FieldName = "SECONDARY_STRENGTH";        break;
          case SURFACE_MOM_DISTORTION:      FieldName = "MOMENTUM_DISTORTION";       break;
          case SURFACE_SECOND_OVER_UNIFORM: FieldName = "SECONDARY_OVER_UNIFORMITY"; break;
          case TOTAL_AVG_TEMPERATURE:       FieldName = "AVG_TOTALTEMP";             break;
          case SURFACE_TOTAL_PRESSURE:      FieldName = "AVG_TOTALPRESS";            break;

          // Not yet covered by new output structure. Be careful these use MARKER_MONITORING.

          case SURFACE_PRESSURE_DROP:
            ObjFunc += config->GetSurface_PressureDrop(0)*Weight_ObjFunc;
            break;
          case SURFACE_STATIC_PRESSURE:
            ObjFunc += config->GetSurface_Pressure(0)*Weight_ObjFunc;
            break;
          case TOTAL_HEATFLUX:
            ObjFunc += solvers[FLOW_SOL]->GetTotal_HeatFlux()*Weight_ObjFunc;
            break;

          default:
            ObjectiveNotCovered = true;
            break;
        }

        if(!FieldName.empty())
          ObjFunc += direct_output[iZone]->GetHistoryFieldValue(FieldName)*Weight_ObjFunc;

        break;
      }
      case DISC_ADJ_HEAT:
      {
        switch(config->GetKind_ObjFunc()) {

          // Not yet covered by new output structure. Be careful these use MARKER_MONITORING.

          case TOTAL_HEATFLUX:
            ObjFunc += solvers[HEAT_SOL]->GetTotal_HeatFlux()*Weight_ObjFunc;
            break;
          case TOTAL_AVG_TEMPERATURE:
            ObjFunc += solvers[HEAT_SOL]->GetTotal_AvgTemperature()*Weight_ObjFunc;
            break;

          default:
            ObjectiveNotCovered = true;
            break;
        }
        break;
      }
      case DISC_ADJ_FEM:
      {
        switch(config->GetKind_ObjFunc()) {

          case REFERENCE_NODE:
            solvers[FEA_SOL]->Compute_OFRefNode(geometry, config);
            ObjFunc += solvers[FEA_SOL]->GetTotal_OFRefNode()*Weight_ObjFunc;
            break;
          case REFERENCE_GEOMETRY:
            solvers[FEA_SOL]->Compute_OFRefGeom(geometry, config);
            ObjFunc += solvers[FEA_SOL]->GetTotal_OFRefGeom()*Weight_ObjFunc;
            break;
          case TOPOL_COMPLIANCE:
            static_cast<CFEASolver*>(solvers[FEA_SOL])->Integrate_FSI_Loads(geometry, config);
            solvers[FEA_SOL]->Compute_OFCompliance(geometry, config);
            ObjFunc += solvers[FEA_SOL]->GetTotal_OFCompliance()*Weight_ObjFunc;
            break;
          case VOLUME_FRACTION:
          case TOPOL_DISCRETENESS:
            solvers[FEA_SOL]->Compute_OFVolFrac(geometry, config);
            ObjFunc += solvers[FEA_SOL]->GetTotal_OFVolFrac()*Weight_ObjFunc;
            break;

          default:
            ObjectiveNotCovered = true;
            break;
        }
        break;
      }
      default:
        break;
    }

    if (ObjectiveNotCovered && (rank == MASTER_NODE) && (kind_recording == SOLUTION_VARIABLES))
      cout << " Objective function not covered in Zone " << iZone << endl;
  }

  if (rank == MASTER_NODE) {
    AD::RegisterOutput(ObjFunc);
    AD::SetIndex(ObjFunc_Index, ObjFunc);
    if (kind_recording == SOLUTION_VARIABLES) {
      cout << " Objective function                   : " << ObjFunc;
      if (driver_config->GetWrt_AD_Statistics()){
        cout << " (" << ObjFunc_Index << ")\n";
      }
      cout << endl;
    }
  }
}

void CDiscAdjMultizoneDriver::SetAdj_ObjFunction() {

  bool time_stepping = config_container[ZONE_0]->GetTime_Marching() != STEADY;
  unsigned long IterAvg_Obj = config_container[ZONE_0]->GetIter_Avg_Objective();
  su2double seeding = 1.0;

  if (time_stepping){
    if (TimeIter < IterAvg_Obj){
      // Default behavior (in case no specific window is chosen) is to use Square-Windowing, i.e. the numerator equals 1.0
      auto windowEvaluator = CWindowingTools();
      su2double weight = windowEvaluator.GetWndWeight(config_container[ZONE_0]->GetKindWindow(), TimeIter, IterAvg_Obj-1);
      seeding = weight / IterAvg_Obj;
    }
    else{
      seeding = 0.0;
    }
  }
  if (rank == MASTER_NODE) {
    AD::SetDerivative(ObjFunc_Index, SU2_TYPE::GetValue(seeding));
  }
}

void CDiscAdjMultizoneDriver::ComputeAdjoints(unsigned short iZone, bool eval_transfer) {

  unsigned short enter_izone = iZone*2+1 + ITERATION_READY;
  unsigned short leave_izone = iZone*2 + ITERATION_READY;

  AD::ClearAdjoints();

  /*--- Initialize the adjoints in iZone ---*/

  iteration_container[iZone][INST_0]->InitializeAdjoint(solver_container, geometry_container,
                                                        config_container, iZone, INST_0);

  /*--- Interpret the stored information by calling the corresponding routine of the AD tool. ---*/

  AD::ComputeAdjoint(enter_izone, leave_izone);

  /*--- Compute adjoints of transfer and mesh deformation routines, only stricktly needed
   *    on the last inner iteration, so if for this zone this section is expensive
   *    (due to mesh deformation) we delay its evaluation. ---*/

  if (eval_transfer || !Has_Deformation(iZone))
    AD::ComputeAdjoint(TRANSFER, OBJECTIVE_FUNCTION);

  /*--- Adjoints of dependencies, needed if derivatives of variables
   *    are extracted (e.g. AoA, Mach, etc.) ---*/

  AD::ComputeAdjoint(DEPENDENCIES, START);

}

void CDiscAdjMultizoneDriver::InitializeCrossTerms() {

  Cross_Terms.resize(nZone, vector<vector<su2passivematrix> >(nZone));

  for(unsigned short iZone = 0; iZone < nZone; iZone++) {
    for (unsigned short jZone = 0; jZone < nZone; jZone++) {
      if (iZone != jZone || interface_container[jZone][iZone] != nullptr) {

        /*--- If jZone contributes to iZone in the primal problem, then
         *    iZone contributes to jZone in the adjoint problem. ---*/

        Cross_Terms[iZone][jZone].resize(MAX_SOLS);

        for (unsigned short iSol=0; iSol < MAX_SOLS; iSol++) {
          CSolver* solver = solver_container[jZone][INST_0][MESH_0][iSol];
          if (solver && solver->GetAdjoint()) {
            unsigned long nPoint = geometry_container[jZone][INST_0][MESH_0]->GetnPoint();
            unsigned short nVar = solver->GetnVar();
            Cross_Terms[iZone][jZone][iSol].resize(nPoint,nVar) = 0.0;
          }
        }
      }
    }
  }
}

void CDiscAdjMultizoneDriver::HandleDataTransfer() {

  unsigned long ExtIter = 0;

  for(iZone = 0; iZone < nZone; iZone++) {

    /*--- In principle, the mesh does not need to be updated ---*/
    bool DeformMesh = false;

    /*--- Transfer from all the remaining zones ---*/
    for (unsigned short jZone = 0; jZone < nZone; jZone++){
      /*--- The target zone is iZone ---*/
      if (jZone != iZone && interface_container[iZone][jZone] != nullptr) {
        DeformMesh = DeformMesh || Transfer_Data(jZone, iZone);
      }
    }
    /*--- If a mesh update is required due to the transfer of data ---*/
    if (DeformMesh) DynamicMeshUpdate(iZone, ExtIter);

    Has_Deformation(iZone) = DeformMesh;
  }
}

void CDiscAdjMultizoneDriver::Add_Solution_To_External(unsigned short iZone) {

  for (unsigned short iSol=0; iSol < MAX_SOLS; iSol++) {
    auto solver = solver_container[iZone][INST_0][MESH_0][iSol];
    if (solver && solver->GetAdjoint())
      solver->Add_Solution_To_External();
  }
}

void CDiscAdjMultizoneDriver::Add_External_To_Solution(unsigned short iZone) {

  for (unsigned short iSol=0; iSol < MAX_SOLS; iSol++) {
    auto solver = solver_container[iZone][INST_0][MESH_0][iSol];
    if (solver && solver->GetAdjoint())
      solver->Add_External_To_Solution();
  }
}

void CDiscAdjMultizoneDriver::Set_SolutionOld_To_Solution(unsigned short iZone) {

  for (unsigned short iSol=0; iSol < MAX_SOLS; iSol++) {
    auto solver = solver_container[iZone][INST_0][MESH_0][iSol];
    if (solver && solver->GetAdjoint())
      solver->Set_OldSolution();
  }
}

void CDiscAdjMultizoneDriver::Update_Cross_Term(unsigned short iZone, unsigned short jZone) {

  for (unsigned short iSol=0; iSol < MAX_SOLS; iSol++) {
    auto solver = solver_container[jZone][INST_0][MESH_0][iSol];
    if (solver && solver->GetAdjoint())
      solver->Update_Cross_Term(config_container[jZone], Cross_Terms[iZone][jZone][iSol]);
  }
}

void CDiscAdjMultizoneDriver::Set_Solution_To_BGSSolution_k(unsigned short iZone) {

  for (unsigned short iSol=0; iSol < MAX_SOLS; iSol++) {
    auto solver = solver_container[iZone][INST_0][MESH_0][iSol];
    if (solver && solver->GetAdjoint())
      solver->GetNodes()->Restore_BGSSolution_k();
  }
}

void CDiscAdjMultizoneDriver::Set_BGSSolution_k_To_Solution(unsigned short iZone) {

  for (unsigned short iSol=0; iSol < MAX_SOLS; iSol++) {
    auto solver = solver_container[iZone][INST_0][MESH_0][iSol];
    if (solver && solver->GetAdjoint())
      solver->GetNodes()->Set_BGSSolution_k();
  }
}

void CDiscAdjMultizoneDriver::SetResidual_BGS(unsigned short iZone) {

  for (unsigned short iSol=0; iSol < MAX_SOLS; iSol++) {
    auto solver = solver_container[iZone][INST_0][MESH_0][iSol];
    if (solver && solver->GetAdjoint())
      solver->ComputeResidual_Multizone(geometry_container[iZone][INST_0][MESH_0], config_container[iZone]);
  }
}
