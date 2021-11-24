/*!
 * \file driver_adjoint_singlezone.cpp
 * \brief The main subroutines for driving adjoint single-zone problems.
 * \author R. Sanchez
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

#include "../../include/drivers/CDiscAdjSinglezoneDriver.hpp"
#include "../../include/output/tools/CWindowingTools.hpp"
#include "../../include/output/COutputFactory.hpp"
#include "../../include/output/COutputLegacy.hpp"
#include "../../include/output/COutput.hpp"
#include "../../include/iteration/CIterationFactory.hpp"
#include "../../include/iteration/CTurboIteration.hpp"
#include "../../../Common/include/toolboxes/CQuasiNewtonInvLeastSquares.hpp"

CDiscAdjSinglezoneDriver::CDiscAdjSinglezoneDriver(char* confFile,
                                                   unsigned short val_nZone,
                                                   SU2_Comm MPICommunicator) : CSinglezoneDriver(confFile,
                                                                                                 val_nZone,
                                                                                                 MPICommunicator) {


  /*--- Store the number of internal iterations that will be run by the adjoint solver ---*/
  nAdjoint_Iter = config_container[ZONE_0]->GetnInner_Iter();


  /*--- Store the pointers ---*/
  config      = config_container[ZONE_0];
  iteration   = iteration_container[ZONE_0][INST_0];
  solver      = solver_container[ZONE_0][INST_0][MESH_0];
  numerics    = numerics_container[ZONE_0][INST_0][MESH_0];
  geometry    = geometry_container[ZONE_0][INST_0][MESH_0];
  integration = integration_container[ZONE_0][INST_0];

  /*--- Store the recording state ---*/
  RecordingState = NONE;

  /*--- Determine if the problem is a turbomachinery problem ---*/
  bool turbo = config->GetBoolTurbomachinery();

  bool compressible = config->GetKind_Regime() == COMPRESSIBLE;

  /*--- Determine if the problem has a mesh deformation solver ---*/
  bool mesh_def = config->GetDeform_Mesh();

  /*--- Initialize the direct iteration ---*/

  switch (config->GetKind_Solver()) {

  case DISC_ADJ_EULER: case DISC_ADJ_NAVIER_STOKES: case DISC_ADJ_RANS:
    case DISC_ADJ_INC_EULER: case DISC_ADJ_INC_NAVIER_STOKES: case DISC_ADJ_INC_RANS:
    if (rank == MASTER_NODE)
      cout << "Direct iteration: Euler/Navier-Stokes/RANS equation." << endl;
    if (turbo) {
      direct_iteration = new CTurboIteration(config);
      output_legacy = COutputFactory::CreateLegacyOutput(config_container[ZONE_0]);
    }
    else       direct_iteration = CIterationFactory::CreateIteration(EULER, config);
    if (compressible) direct_output = COutputFactory::CreateOutput(EULER, config, nDim);
    else direct_output =  COutputFactory::CreateOutput(INC_EULER, config, nDim);
    MainVariables = SOLUTION_VARIABLES;
    if (mesh_def) SecondaryVariables = MESH_DEFORM;
    else          SecondaryVariables = MESH_COORDS;
    break;

  case DISC_ADJ_FEM_EULER : case DISC_ADJ_FEM_NS : case DISC_ADJ_FEM_RANS :
    if (rank == MASTER_NODE)
      cout << "Direct iteration: Euler/Navier-Stokes/RANS equation." << endl;
    direct_iteration = CIterationFactory::CreateIteration(FEM_EULER, config);
    direct_output = COutputFactory::CreateOutput(FEM_EULER, config, nDim);
    MainVariables = SOLUTION_VARIABLES;
    SecondaryVariables = MESH_COORDS;
    break;

  case DISC_ADJ_FEM:
    if (rank == MASTER_NODE)
      cout << "Direct iteration: elasticity equation." << endl;
    direct_iteration =  CIterationFactory::CreateIteration(FEM_ELASTICITY, config);
    direct_output = COutputFactory::CreateOutput(FEM_ELASTICITY, config, nDim);
    MainVariables = SOLUTION_VARIABLES;
    SecondaryVariables = MESH_COORDS;
    break;

  case DISC_ADJ_HEAT:
    if (rank == MASTER_NODE)
      cout << "Direct iteration: heat equation." << endl;
    direct_iteration = CIterationFactory::CreateIteration(HEAT_EQUATION, config);
    direct_output = COutputFactory::CreateOutput(HEAT_EQUATION, config, nDim);
    MainVariables = SOLUTION_VARIABLES;
    SecondaryVariables = MESH_COORDS;
    break;

  }

 direct_output->PreprocessHistoryOutput(config, false);

}

CDiscAdjSinglezoneDriver::~CDiscAdjSinglezoneDriver(void) {

  delete direct_iteration;
  delete direct_output;

}

void CDiscAdjSinglezoneDriver::Preprocess(unsigned long TimeIter) {

  config_container[ZONE_0]->SetTimeIter(TimeIter);

  /*--- NOTE: Inv Design Routines moved to CDiscAdjFluidIteration::Preprocess ---*/

  /*--- Preprocess the adjoint iteration ---*/

  iteration->Preprocess(output_container[ZONE_0], integration_container, geometry_container,
                        solver_container, numerics_container, config_container,
                        surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

  /*--- For the adjoint iteration we need the derivatives of the iteration function with
   *--- respect to the conservative variables. Since these derivatives do not change in the steady state case
   *--- we only have to record if the current recording is different from the main variables. ---*/

  if (RecordingState != MainVariables){

    MainRecording();

  }

}

void CDiscAdjSinglezoneDriver::Run() {

  const bool steady = !config->GetTime_Domain();

  CQuasiNewtonInvLeastSquares<passivedouble> fixPtCorrector;
  if (config->GetnQuasiNewtonSamples() > 1) {
    fixPtCorrector.resize(config->GetnQuasiNewtonSamples(),
                          geometry_container[ZONE_0][INST_0][MESH_0]->GetnPoint(),
                          GetTotalNumberOfVariables(ZONE_0,true),
                          geometry_container[ZONE_0][INST_0][MESH_0]->GetnPointDomain());

    if (TimeIter != 0) GetAllSolutions(ZONE_0, true, fixPtCorrector);
  }

  for (auto Adjoint_Iter = 0ul; Adjoint_Iter < nAdjoint_Iter; Adjoint_Iter++) {

    /*--- Initialize the adjoint of the output variables of the iteration with the adjoint solution
     *--- of the previous iteration. The values are passed to the AD tool.
     *--- Issues with iteration number should be dealt with once the output structure is in place. ---*/

    config->SetInnerIter(Adjoint_Iter);

    iteration->InitializeAdjoint(solver_container, geometry_container, config_container, ZONE_0, INST_0);

    /*--- Initialize the adjoint of the objective function with 1.0. ---*/

    SetAdj_ObjFunction();

    /*--- Interpret the stored information by calling the corresponding routine of the AD tool. ---*/

    AD::ComputeAdjoint();

    /*--- Extract the computed adjoint values of the input variables and store them for the next iteration. ---*/

    iteration->Iterate(output_container[ZONE_0], integration_container, geometry_container,
                         solver_container, numerics_container, config_container,
                         surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

    /*--- Monitor the pseudo-time ---*/

    StopCalc = iteration->Monitor(output_container[ZONE_0], integration_container, geometry_container,
                                  solver_container, numerics_container, config_container,
                                  surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

    /*--- Clear the stored adjoint information to be ready for a new evaluation. ---*/

    AD::ClearAdjoints();

    /*--- Output files for steady state simulations. ---*/

    if (steady) {
      iteration->Output(output_container[ZONE_0], geometry_container, solver_container,
                        config_container, Adjoint_Iter, false, ZONE_0, INST_0);
    }

    if (StopCalc) break;

    /*--- Correct the solution with the quasi-Newton approach. ---*/

    if (fixPtCorrector.size()) {
      GetAllSolutions(ZONE_0, true, fixPtCorrector.FPresult());
      SetAllSolutions(ZONE_0, true, fixPtCorrector.compute());
    }

  }

}

void CDiscAdjSinglezoneDriver::Postprocess() {

  switch(config->GetKind_Solver())
  {
    case DISC_ADJ_EULER :     case DISC_ADJ_NAVIER_STOKES :     case DISC_ADJ_RANS :
    case DISC_ADJ_INC_EULER : case DISC_ADJ_INC_NAVIER_STOKES : case DISC_ADJ_INC_RANS :

      /*--- Compute the geometrical sensitivities ---*/
      SecondaryRecording();
      break;

    case DISC_ADJ_FEM :

      /*--- Compute the geometrical sensitivities ---*/
      SecondaryRecording();

      iteration->Postprocess(output_container[ZONE_0], integration_container, geometry_container,
                             solver_container, numerics_container, config_container,
                             surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);
      break;
  }//switch

}

void CDiscAdjSinglezoneDriver::SetRecording(unsigned short kind_recording){

  AD::Reset();

  /*--- Prepare for recording by resetting the solution to the initial converged solution---*/

  iteration->SetRecording(solver_container, geometry_container, config_container, ZONE_0, INST_0, kind_recording);

  /*---Enable recording and register input of the iteration --- */

  if (kind_recording != NONE){

    AD::StartRecording();

    if (rank == MASTER_NODE && kind_recording == MainVariables) {
      cout << endl << "-------------------------------------------------------------------------" << endl;
      cout << "Direct iteration to store the primal computational graph." << endl;
      cout << "Compute residuals to check the convergence of the direct problem." << endl;
    }
    iteration->RegisterInput(solver_container, geometry_container, config_container, ZONE_0, INST_0, kind_recording);

  }

  /*--- Set the dependencies of the iteration ---*/

  iteration->SetDependencies(solver_container, geometry_container, numerics_container, config_container, ZONE_0,
                             INST_0, kind_recording);

  /*--- Do one iteration of the direct solver ---*/

  DirectRun(kind_recording);

  // NOTE: The inverse design calls were moved to DirectRun() - postprocess

  /*--- Store the recording state ---*/

  RecordingState = kind_recording;

  /*--- Register Output of the iteration ---*/

  iteration->RegisterOutput(solver_container, geometry_container, config_container, output_container[ZONE_0], ZONE_0, INST_0);

  /*--- Extract the objective function and store it --- */

  SetObjFunction();

  AD::StopRecording();

}

void CDiscAdjSinglezoneDriver::SetAdj_ObjFunction(){

  bool time_stepping = config->GetTime_Marching() != STEADY;
  unsigned long IterAvg_Obj = config->GetIter_Avg_Objective();
  su2double seeding = 1.0;

  CWindowingTools windowEvaluator = CWindowingTools();

  if (time_stepping){
    if (TimeIter < IterAvg_Obj){
      // Default behavior (in case no specific window is chosen) is to use Square-Windowing, i.e. the numerator equals 1.0
      seeding = windowEvaluator.GetWndWeight(config->GetKindWindow(),TimeIter, IterAvg_Obj-1)/ (static_cast<su2double>(IterAvg_Obj));
    }
    else{
      seeding = 0.0;
    }
  }

  if (rank == MASTER_NODE){
    SU2_TYPE::SetDerivative(ObjFunc, SU2_TYPE::GetValue(seeding));
  } else {
    SU2_TYPE::SetDerivative(ObjFunc, 0.0);
  }
}

void CDiscAdjSinglezoneDriver::SetObjFunction(){

  bool heat         = (config->GetWeakly_Coupled_Heat());
  bool turbo        = (config->GetBoolTurbomachinery());

  ObjFunc = 0.0;

  direct_output->SetHistory_Output(geometry, solver, config,
                                   config->GetTimeIter(),
                                   config->GetOuterIter(),
                                   config->GetInnerIter());

  /*--- Specific scalar objective functions ---*/

  switch (config->GetKind_Solver()) {
  case DISC_ADJ_INC_EULER:       case DISC_ADJ_INC_NAVIER_STOKES:      case DISC_ADJ_INC_RANS:
  case DISC_ADJ_EULER:           case DISC_ADJ_NAVIER_STOKES:          case DISC_ADJ_RANS:
  case DISC_ADJ_FEM_EULER:       case DISC_ADJ_FEM_NS:                 case DISC_ADJ_FEM_RANS:

    solver[FLOW_SOL]->SetTotal_ComboObj(0.0);

//    if (config->GetnMarker_Analyze() != 0)
//      output->SpecialOutput_AnalyzeSurface(solver[FLOW_SOL], geometry, config, false);

//    if ((config->GetnMarker_Analyze() != 0) && compressible)
//      output->SpecialOutput_Distortion(solver[FLOW_SOL], geometry, config, false);

//    if (config->GetnMarker_NearFieldBound() != 0)
//      output->SpecialOutput_SonicBoom(solver[FLOW_SOL], geometry, config, false);

//    if (config->GetPlot_Section_Forces())
//      output->SpecialOutput_SpanLoad(solver[FLOW_SOL], geometry, config, false);

    /*--- Surface based obj. function ---*/

    solver[FLOW_SOL]->Evaluate_ObjFunc(config);
    ObjFunc += solver[FLOW_SOL]->GetTotal_ComboObj();
    if (heat){
      if (config->GetKind_ObjFunc() == TOTAL_HEATFLUX) {
        ObjFunc += solver[HEAT_SOL]->GetTotal_HeatFlux();
      }
      else if (config->GetKind_ObjFunc() == TOTAL_AVG_TEMPERATURE) {
        ObjFunc += solver[HEAT_SOL]->GetTotal_AvgTemperature();
      }
    }

    /*--- This calls to be moved to a generic framework at a next stage         ---*/
    /*--- Some things that are currently hacked into output must be reorganized ---*/
    if (turbo){

      solver[FLOW_SOL]->SetTotal_ComboObj(0.0);
      output_legacy->ComputeTurboPerformance(solver[FLOW_SOL], geometry, config);

      unsigned short nMarkerTurboPerf = config->GetnMarker_TurboPerformance();
      unsigned short nSpanSections = config->GetnSpanWiseSections();

      switch (config_container[ZONE_0]->GetKind_ObjFunc()){
      case ENTROPY_GENERATION:
        solver[FLOW_SOL]->AddTotal_ComboObj(output_legacy->GetEntropyGen(nMarkerTurboPerf-1, nSpanSections));
        break;
      case FLOW_ANGLE_OUT:
        solver[FLOW_SOL]->AddTotal_ComboObj(output_legacy->GetFlowAngleOut(nMarkerTurboPerf-1, nSpanSections));
        break;
      case MASS_FLOW_IN:
        solver[FLOW_SOL]->AddTotal_ComboObj(output_legacy->GetMassFlowIn(nMarkerTurboPerf-1, nSpanSections));
        break;
      default:
        break;
      }

      ObjFunc = solver[FLOW_SOL]->GetTotal_ComboObj();

    }

    break;
  case DISC_ADJ_FEM:
    switch (config->GetKind_ObjFunc()){
    case REFERENCE_GEOMETRY:
      ObjFunc = solver[FEA_SOL]->GetTotal_OFRefGeom();
      break;
    case REFERENCE_NODE:
      ObjFunc = solver[FEA_SOL]->GetTotal_OFRefNode();
      break;
    case TOPOL_COMPLIANCE:
      ObjFunc = solver[FEA_SOL]->GetTotal_OFCompliance();
      break;
    case VOLUME_FRACTION:
    case TOPOL_DISCRETENESS:
      ObjFunc = solver[FEA_SOL]->GetTotal_OFVolFrac();
      break;
    default:
      ObjFunc = 0.0;  // If the objective function is computed in a different physical problem
      break;
    }
    break;
  }

  if (rank == MASTER_NODE){
    AD::RegisterOutput(ObjFunc);
  }

}

void CDiscAdjSinglezoneDriver::DirectRun(unsigned short kind_recording){

  /*--- Mesh movement ---*/

  direct_iteration->SetMesh_Deformation(geometry_container[ZONE_0][INST_0], solver, numerics, config, kind_recording);

  /*--- Zone preprocessing ---*/

  direct_iteration->Preprocess(direct_output, integration_container, geometry_container, solver_container, numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

  /*--- Iterate the direct solver ---*/

  direct_iteration->Iterate(direct_output, integration_container, geometry_container, solver_container, numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

  /*--- Postprocess the direct solver ---*/

  direct_iteration->Postprocess(direct_output, integration_container, geometry_container, solver_container, numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

  /*--- Print the direct residual to screen ---*/

  Print_DirectResidual(kind_recording);

}

void CDiscAdjSinglezoneDriver::Print_DirectResidual(unsigned short kind_recording){

  /*--- Print the residuals of the direct iteration that we just recorded ---*/
  /*--- This routine should be moved to the output, once the new structure is in place ---*/
  if ((rank == MASTER_NODE) && (kind_recording == MainVariables)){

    switch (config->GetKind_Solver()) {

    case DISC_ADJ_EULER: case DISC_ADJ_NAVIER_STOKES: case DISC_ADJ_RANS:
    case DISC_ADJ_INC_EULER: case DISC_ADJ_INC_NAVIER_STOKES: case DISC_ADJ_INC_RANS:
    case DISC_ADJ_FEM_EULER : case DISC_ADJ_FEM_NS : case DISC_ADJ_FEM_RANS :
      cout << "log10[U(0)]: "   << log10(solver[FLOW_SOL]->GetRes_RMS(0))
           << ", log10[U(1)]: " << log10(solver[FLOW_SOL]->GetRes_RMS(1))
           << ", log10[U(2)]: " << log10(solver[FLOW_SOL]->GetRes_RMS(2)) << "." << endl;
      cout << "log10[U(3)]: " << log10(solver[FLOW_SOL]->GetRes_RMS(3));
      if (geometry->GetnDim() == 3) cout << ", log10[U(4)]: " << log10(solver[FLOW_SOL]->GetRes_RMS(4));
      cout << "." << endl;
      if ( config->GetKind_Turb_Model() != NONE && !config->GetFrozen_Visc_Disc()) {
        cout << "log10[Turb(0)]: "   << log10(solver[TURB_SOL]->GetRes_RMS(0));
        if (solver[TURB_SOL]->GetnVar() > 1) cout << ", log10[Turb(1)]: " << log10(solver[TURB_SOL]->GetRes_RMS(1));
        cout << "." << endl;
      }
      if (config->GetWeakly_Coupled_Heat()){
        cout << "log10[Heat(0)]: "   << log10(solver[HEAT_SOL]->GetRes_RMS(0)) << "." << endl;
      }
      if ( config->AddRadiation()) {
        cout <<"log10[E(rad)]: " << log10(solver[RAD_SOL]->GetRes_RMS(0)) << endl;
      }
      break;

    case DISC_ADJ_FEM:

      if (config->GetGeometricConditions() == LARGE_DEFORMATIONS){
        cout << "UTOL-A: "   << log10(solver[FEA_SOL]->GetRes_FEM(0))
             << ", RTOL-A: " << log10(solver[FEA_SOL]->GetRes_FEM(1))
             << ", ETOL-A: " << log10(solver[FEA_SOL]->GetRes_FEM(2)) << "." << endl;
      }
      else{
        if (geometry->GetnDim() == 2){
          cout << "log10[RMS Ux]: "   << log10(solver[FEA_SOL]->GetRes_RMS(0))
               << ", log10[RMS Uy]: " << log10(solver[FEA_SOL]->GetRes_RMS(1)) << "." << endl;
        }
        else{
          cout << "log10[RMS Ux]: "   << log10(solver[FEA_SOL]->GetRes_RMS(0))
               << ", log10[RMS Uy]: " << log10(solver[FEA_SOL]->GetRes_RMS(1))
               << ", log10[RMS Uz]: " << log10(solver[FEA_SOL]->GetRes_RMS(2))<< "." << endl;
        }
      }

      break;

    case DISC_ADJ_HEAT:
      cout << "log10[Cons(0)]: "   << log10(solver[HEAT_SOL]->GetRes_RMS(0)) << "." << endl;
      break;

    }

    cout << "-------------------------------------------------------------------------" << endl << endl;
  }
  else if ((rank == MASTER_NODE) && (kind_recording == SecondaryVariables) && (SecondaryVariables != NONE)){
    cout << endl << "Recording the computational graph with respect to the ";
    switch (SecondaryVariables){
      case MESH_COORDS: cout << "mesh coordinates." << endl;    break;
      default:          cout << "secondary variables." << endl; break;
     }
  }

}

void CDiscAdjSinglezoneDriver::MainRecording(){

  /*--- SetRecording stores the computational graph on one iteration of the direct problem. Calling it with NONE
   *    as argument ensures that all information from a previous recording is removed. ---*/

  SetRecording(NONE);

  /*--- Store the computational graph of one direct iteration with the conservative variables as input. ---*/

  SetRecording(MainVariables);

}

void CDiscAdjSinglezoneDriver::SecondaryRecording(){

  /*--- SetRecording stores the computational graph on one iteration of the direct problem. Calling it with NONE
   *    as argument ensures that all information from a previous recording is removed. ---*/

  SetRecording(NONE);

  /*--- Store the computational graph of one direct iteration with the secondary variables as input. ---*/

  SetRecording(SecondaryVariables);

  /*--- Initialize the adjoint of the output variables of the iteration with the adjoint solution
   *    of the current iteration. The values are passed to the AD tool. ---*/

  iteration->InitializeAdjoint(solver_container, geometry_container, config_container, ZONE_0, INST_0);

  /*--- Initialize the adjoint of the objective function with 1.0. ---*/

  SetAdj_ObjFunction();

  /*--- Interpret the stored information by calling the corresponding routine of the AD tool. ---*/

  AD::ComputeAdjoint();

  /*--- Extract the computed sensitivity values. ---*/

  int IDX_SOL;

  if (config->GetKind_Solver() == DISC_ADJ_FEM) {
    IDX_SOL = ADJFEA_SOL;
  } else if(SecondaryVariables == MESH_COORDS) {
    IDX_SOL = ADJFLOW_SOL;
  } else if(SecondaryVariables == MESH_DEFORM) {
    IDX_SOL = ADJMESH_SOL;
  } else {
    IDX_SOL = -1;
  }

  if(IDX_SOL >= 0)
    solver[IDX_SOL]->SetSensitivity(geometry, solver, config);

  /*--- Clear the stored adjoint information to be ready for a new evaluation. ---*/

  AD::ClearAdjoints();

}
