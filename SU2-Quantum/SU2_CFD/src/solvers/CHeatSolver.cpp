/*!
 * \file CHeatSolver.cpp
 * \brief Main subrotuines for solving the heat equation
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


#include "../../include/solvers/CHeatSolver.hpp"

CHeatSolver::CHeatSolver(void) : CSolver() {

  ConjugateVar = nullptr;
  HeatFlux     = nullptr;
}

CHeatSolver::CHeatSolver(CGeometry *geometry, CConfig *config, unsigned short iMesh) : CSolver() {

  unsigned short iVar, iDim, nLineLets, iMarker;
  unsigned long iVertex;

  bool multizone = config->GetMultizone_Problem();

  int rank = MASTER_NODE;

  bool heat_equation = ((config->GetKind_Solver() == HEAT_EQUATION) ||
                        (config->GetKind_Solver() == DISC_ADJ_HEAT));

  /* A grid is defined as dynamic if there's rigid grid movement or grid deformation AND the problem is time domain */
  dynamic_grid = config->GetDynamic_Grid();

#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  /*--- Dimension of the problem --> temperature is the only conservative variable ---*/

  nVar = 1;
  nPoint = geometry->GetnPoint();
  nPointDomain = geometry->GetnPointDomain();

  /*--- Initialize nVarGrad for deallocation ---*/

  nVarGrad = nVar;

  /*--- Define geometry constants in the solver structure ---*/

  nDim = geometry->GetnDim();
  nMarker = config->GetnMarker_All();

  CurrentMesh = iMesh;
  /*--- Define some auxiliar vector related with the residual ---*/

  Residual      = new su2double[nVar];  for (iVar = 0; iVar < nVar; iVar++) Residual[iVar]      = 0.0;
  Residual_RMS  = new su2double[nVar];  for (iVar = 0; iVar < nVar; iVar++) Residual_RMS[iVar]  = 0.0;
  Residual_i    = new su2double[nVar];  for (iVar = 0; iVar < nVar; iVar++) Residual_i[iVar]    = 0.0;
  Residual_j    = new su2double[nVar];  for (iVar = 0; iVar < nVar; iVar++) Residual_j[iVar]    = 0.0;
  Residual_Max  = new su2double[nVar];  for (iVar = 0; iVar < nVar; iVar++) Residual_Max[iVar]  = 0.0;
  Res_Conv      = new su2double[nVar];  for (iVar = 0; iVar < nVar; iVar++) Res_Conv[iVar]      = 0.0;
  Res_Visc      = new su2double[nVar];  for (iVar = 0; iVar < nVar; iVar++) Res_Visc[iVar]      = 0.0;

  /*--- Define some structures for locating max residuals ---*/

  Point_Max = new unsigned long[nVar];
  for (iVar = 0; iVar < nVar; iVar++) Point_Max[iVar] = 0;
  Point_Max_Coord = new su2double*[nVar];
  for (iVar = 0; iVar < nVar; iVar++) {
    Point_Max_Coord[iVar] = new su2double[nDim];
    for (iDim = 0; iDim < nDim; iDim++) Point_Max_Coord[iVar][iDim] = 0.0;
  }

  /*--- Define some auxiliar vector related with the solution ---*/

  Solution = new su2double[nVar];
  Solution_i = new su2double[nVar]; Solution_j = new su2double[nVar];

  /*--- Define some auxiliary vectors related to the geometry ---*/

  Vector   = new su2double[nDim]; for (iDim = 0; iDim < nDim; iDim++) Vector[iDim]   = 0.0;
  Vector_i = new su2double[nDim]; for (iDim = 0; iDim < nDim; iDim++) Vector_i[iDim] = 0.0;
  Vector_j = new su2double[nDim]; for (iDim = 0; iDim < nDim; iDim++) Vector_j[iDim] = 0.0;

  /*--- Define some auxiliary vectors related to the primitive flow solution ---*/

  Primitive_Flow_i = new su2double[nDim+1]; for (iVar = 0; iVar < nDim+1; iVar++) Primitive_Flow_i[iVar] = 0.0;
  Primitive_Flow_j = new su2double[nDim+1]; for (iVar = 0; iVar < nDim+1; iVar++) Primitive_Flow_j[iVar] = 0.0;

  /*--- Jacobians and vector structures for implicit computations ---*/

  Jacobian_i = new su2double* [nVar];
  Jacobian_j = new su2double* [nVar];
  for (iVar = 0; iVar < nVar; iVar++) {
    Jacobian_i[iVar] = new su2double [nVar];
    Jacobian_j[iVar] = new su2double [nVar];
  }

  /*--- Initialization of the structure of the whole Jacobian ---*/

  if (rank == MASTER_NODE) cout << "Initialize Jacobian structure (heat equation) MG level: " << iMesh << "." << endl;
  Jacobian.Initialize(nPoint, nPointDomain, nVar, nVar, true, geometry, config);

  if (config->GetKind_Linear_Solver_Prec() == LINELET) {
    nLineLets = Jacobian.BuildLineletPreconditioner(geometry, config);
    if (rank == MASTER_NODE) cout << "Compute linelet structure. " << nLineLets << " elements in each line (average)." << endl;
  }

  LinSysSol.Initialize(nPoint, nPointDomain, nVar, 0.0);
  LinSysRes.Initialize(nPoint, nPointDomain, nVar, 0.0);

  if (config->GetExtraOutput()) {
    if (nDim == 2) { nOutputVariables = 13; }
    else if (nDim == 3) { nOutputVariables = 19; }
    OutputVariables.Initialize(nPoint, nPointDomain, nOutputVariables, 0.0);
    OutputHeadingNames = new string[nOutputVariables];
  }

  HeatFlux_per_Marker = new su2double[nMarker];
  AverageT_per_Marker = new su2double[nMarker];
  Surface_Areas       = new su2double[config->GetnMarker_HeatFlux()];

  for(iMarker = 0; iMarker < nMarker; iMarker++) {
    HeatFlux_per_Marker[iMarker]        = 0.0;
    AverageT_per_Marker[iMarker]  = 0.0;
  }
  for(iMarker = 0; iMarker < config->GetnMarker_HeatFlux(); iMarker++) {
    Surface_Areas[iMarker] = 0.0;
  }

  Set_Heatflux_Areas(geometry, config);

  /*--- Set the reference values for temperature ---*/

  su2double Temperature_FreeStream = config->GetInc_Temperature_Init();
  config->SetTemperature_FreeStream(Temperature_FreeStream);
  su2double Temperature_Ref = 0.0;

  if (config->GetRef_Inc_NonDim() == DIMENSIONAL) {
    Temperature_Ref = 1.0;
  }
  else if (config->GetRef_Inc_NonDim() == INITIAL_VALUES) {
    Temperature_Ref = Temperature_FreeStream;
  }
  else if (config->GetRef_Inc_NonDim() == REFERENCE_VALUES) {
    Temperature_Ref = config->GetInc_Temperature_Ref();
  }
  config->SetTemperature_Ref(Temperature_Ref);

  /*--- Set the reference values for heat fluxes. If the heat solver runs stand-alone,
   *    thermal conductivity is read directly from config file ---*/

  if (heat_equation) {

    su2double rho_cp = config->GetDensity_Solid()*config->GetSpecific_Heat_Cp();
    config->SetThermalDiffusivity_Solid(config->GetThermalConductivity_Solid() / rho_cp);

    config->SetTemperature_FreeStreamND(config->GetTemperature_Initial_Solid()/config->GetTemperature_Ref());
    config->SetHeat_Flux_Ref(rho_cp*Temperature_Ref);
  }
  else {

    config->SetTemperature_FreeStreamND(config->GetTemperature_FreeStream()/config->GetTemperature_Ref());
    config->SetHeat_Flux_Ref(config->GetViscosity_Ref()*config->GetSpecific_Heat_Cp());
  }

  /*--- Store the value of the temperature and the heat flux density at the boundaries,
   used for communications with donor cells ---*/

  unsigned short nConjVariables = 4;

  ConjugateVar = new su2double** [nMarker];
  for (iMarker = 0; iMarker < nMarker; iMarker++) {
    ConjugateVar[iMarker] = new su2double* [geometry->nVertex[iMarker]];
    for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

      ConjugateVar[iMarker][iVertex] = new su2double [nConjVariables];
      for (iVar = 0; iVar < nConjVariables ; iVar++) {
        ConjugateVar[iMarker][iVertex][iVar] = 0.0;
      }
      ConjugateVar[iMarker][iVertex][0] = config->GetTemperature_FreeStreamND();
    }
  }

  /*--- Heat flux in all the markers ---*/

  HeatFlux = new su2double* [nMarker];
  for (iMarker = 0; iMarker < nMarker; iMarker++) {
    HeatFlux[iMarker] = new su2double [geometry->nVertex[iMarker]];
    for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {
      HeatFlux[iMarker][iVertex] = 0.0;
    }
  }

  if (multizone){
    /*--- Initialize the BGS residuals. ---*/
    Residual_BGS      = new su2double[nVar];         for (iVar = 0; iVar < nVar; iVar++) Residual_BGS[iVar]  = 1.0;
    Residual_Max_BGS  = new su2double[nVar];         for (iVar = 0; iVar < nVar; iVar++) Residual_Max_BGS[iVar]  = 1.0;

    /*--- Define some structures for locating max residuals ---*/

    Point_Max_BGS       = new unsigned long[nVar];  for (iVar = 0; iVar < nVar; iVar++) Point_Max_BGS[iVar]  = 0;
    Point_Max_Coord_BGS = new su2double*[nVar];
    for (iVar = 0; iVar < nVar; iVar++) {
      Point_Max_Coord_BGS[iVar] = new su2double[nDim];
      for (iDim = 0; iDim < nDim; iDim++) Point_Max_Coord_BGS[iVar][iDim] = 0.0;
    }
  }

  /*--- Initialize the nodes vector. ---*/

  nodes = new CHeatVariable(config->GetTemperature_FreeStreamND(), nPoint, nDim, nVar, config);

  SetBaseClassPointerToNodes();

  /*--- MPI solution ---*/

  InitiateComms(geometry, config, SOLUTION);
  CompleteComms(geometry, config, SOLUTION);

  /*--- Add the solver name (max 8 characters) ---*/

  SolverName = "HEAT";
}

CHeatSolver::~CHeatSolver(void) {

  unsigned short iMarker;

  if (HeatFlux != nullptr) {
    for (iMarker = 0; iMarker < nMarker; iMarker++) {
      delete [] HeatFlux[iMarker];
    }
    delete [] HeatFlux;
  }

  delete nodes;
}

void CHeatSolver::Preprocessing(CGeometry *geometry, CSolver **solver_container, CConfig *config, unsigned short iMesh, unsigned short iRKStep, unsigned short RunTime_EqSystem, bool Output) {

  unsigned long iPoint;
  bool center = (config->GetKind_ConvNumScheme_Heat() == SPACE_CENTERED);

  if (center) {
    SetUndivided_Laplacian(geometry, config);
  }

  for (iPoint = 0; iPoint < nPoint; iPoint ++) {

    /*--- Initialize the residual vector ---*/

    LinSysRes.SetBlock_Zero(iPoint);
  }

  /*--- Initialize the Jacobian matrices ---*/

  Jacobian.SetValZero();

  if (config->GetReconstructionGradientRequired()) {
    if (config->GetKind_Gradient_Method_Recon() == GREEN_GAUSS)
      SetSolution_Gradient_GG(geometry, config, true);
    if (config->GetKind_Gradient_Method_Recon() == LEAST_SQUARES)
      SetSolution_Gradient_LS(geometry, config, true);
    if (config->GetKind_Gradient_Method_Recon() == WEIGHTED_LEAST_SQUARES)
      SetSolution_Gradient_LS(geometry, config, true);
  }
  if (config->GetKind_Gradient_Method() == GREEN_GAUSS) SetSolution_Gradient_GG(geometry, config);
  if (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES) SetSolution_Gradient_LS(geometry, config);
}

void CHeatSolver::Postprocessing(CGeometry *geometry, CSolver **solver_container, CConfig *config, unsigned short iMesh) { }

void CHeatSolver::LoadRestart(CGeometry **geometry, CSolver ***solver, CConfig *config, int val_iter, bool val_update_geo) {

  /*--- Restart the solution from file information ---*/

  unsigned short iDim, iVar, iMesh;
  unsigned long iPoint, index, iChildren, Point_Fine;

  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));

  bool heat_equation = ((config->GetKind_Solver() == HEAT_EQUATION) ||
                        (config->GetKind_Solver() == DISC_ADJ_HEAT));

  su2double Area_Children, Area_Parent, *Coord, *Solution_Fine;

  string restart_filename = config->GetFilename(config->GetSolution_FileName(), "", val_iter);

  Coord = new su2double [nDim];
  for (iDim = 0; iDim < nDim; iDim++)
    Coord[iDim] = 0.0;

  int rank = MASTER_NODE;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  int counter = 0;
  long iPoint_Local = 0; unsigned long iPoint_Global = 0;
  unsigned long iPoint_Global_Local = 0;
  unsigned short rbuf_NotMatching = 0, sbuf_NotMatching = 0;

  /*--- Skip coordinates ---*/

  unsigned short skipVars = 0;

  if (flow) {

    if (nDim == 2) skipVars += 5;
    if (nDim == 3) skipVars += 7;
  }
  else if (heat_equation) {

    if (nDim == 2) skipVars += 2;
    if (nDim == 3) skipVars += 3;
  }
  else {
    cout << "WARNING: Finite volume heat solver's restart routine could not load data." << endl;
  }

  /*--- Read the restart data from either an ASCII or binary SU2 file. ---*/

  if (config->GetRead_Binary_Restart()) {
    Read_SU2_Restart_Binary(geometry[MESH_0], config, restart_filename);
  } else {
    Read_SU2_Restart_ASCII(geometry[MESH_0], config, restart_filename);
  }

  /*--- Load data from the restart into correct containers. ---*/

  counter = 0;
  for (iPoint_Global = 0; iPoint_Global < geometry[MESH_0]->GetGlobal_nPointDomain(); iPoint_Global++ ) {

    /*--- Retrieve local index. If this node from the restart file lives
     on the current processor, we will load and instantiate the vars. ---*/

    iPoint_Local = geometry[MESH_0]->GetGlobal_to_Local_Point(iPoint_Global);

    if (iPoint_Local > -1) {

      /*--- We need to store this point's data, so jump to the correct
       offset in the buffer of data from the restart file and load it. ---*/

      index = counter*Restart_Vars[1] + skipVars;
      for (iVar = 0; iVar < nVar; iVar++) Solution[iVar] = Restart_Data[index+iVar];
      nodes->SetSolution(iPoint_Local,Solution);
      iPoint_Global_Local++;

      /*--- Increment the overall counter for how many points have been loaded. ---*/
      counter++;
    }

  }

  /*--- Detect a wrong solution file ---*/

  if (iPoint_Global_Local < nPointDomain) { sbuf_NotMatching = 1; }

#ifndef HAVE_MPI
  rbuf_NotMatching = sbuf_NotMatching;
#else
  SU2_MPI::Allreduce(&sbuf_NotMatching, &rbuf_NotMatching, 1, MPI_UNSIGNED_SHORT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (rbuf_NotMatching != 0) {
    if (rank == MASTER_NODE) {
      cout << endl << "The solution file " << restart_filename.data() << " doesn't match with the mesh file!" << endl;
      cout << "It could be empty lines at the end of the file." << endl << endl;
    }
#ifndef HAVE_MPI
    exit(EXIT_FAILURE);
#else
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD,1);
    MPI_Finalize();
#endif
  }

  /*--- Communicate the loaded solution on the fine grid before we transfer
   it down to the coarse levels. We alo call the preprocessing routine
   on the fine level in order to have all necessary quantities updated,
   especially if this is a turbulent simulation (eddy viscosity). ---*/

  solver[MESH_0][HEAT_SOL]->InitiateComms(geometry[MESH_0], config, SOLUTION);
  solver[MESH_0][HEAT_SOL]->CompleteComms(geometry[MESH_0], config, SOLUTION);

  solver[MESH_0][HEAT_SOL]->Preprocessing(geometry[MESH_0], solver[MESH_0], config, MESH_0, NO_RK_ITER, RUNTIME_HEAT_SYS, false);

  /*--- Interpolate the solution down to the coarse multigrid levels ---*/

  for (iMesh = 1; iMesh <= config->GetnMGLevels(); iMesh++) {
    for (iPoint = 0; iPoint < geometry[iMesh]->GetnPoint(); iPoint++) {
      Area_Parent = geometry[iMesh]->nodes->GetVolume(iPoint);
      for (iVar = 0; iVar < nVar; iVar++) Solution[iVar] = 0.0;
      for (iChildren = 0; iChildren < geometry[iMesh]->nodes->GetnChildren_CV(iPoint); iChildren++) {
        Point_Fine = geometry[iMesh]->nodes->GetChildren_CV(iPoint, iChildren);
        Area_Children = geometry[iMesh-1]->nodes->GetVolume(Point_Fine);
        Solution_Fine = solver[iMesh-1][HEAT_SOL]->GetNodes()->GetSolution(Point_Fine);
        for (iVar = 0; iVar < nVar; iVar++) {
          Solution[iVar] += Solution_Fine[iVar]*Area_Children/Area_Parent;
        }
      }
      solver[iMesh][HEAT_SOL]->GetNodes()->SetSolution(iPoint,Solution);
    }
    solver[iMesh][HEAT_SOL]->InitiateComms(geometry[iMesh], config, SOLUTION);
    solver[iMesh][HEAT_SOL]->CompleteComms(geometry[iMesh], config, SOLUTION);
    solver[iMesh][HEAT_SOL]->Preprocessing(geometry[iMesh], solver[iMesh], config, iMesh, NO_RK_ITER, RUNTIME_HEAT_SYS, false);
  }

  delete [] Coord;

  /*--- Delete the class memory that is used to load the restart. ---*/

  delete [] Restart_Vars;
  delete [] Restart_Data;
  Restart_Vars = nullptr; Restart_Data = nullptr;

}


void CHeatSolver::SetUndivided_Laplacian(CGeometry *geometry, CConfig *config) {

  unsigned long iPoint, jPoint, iEdge;
  su2double *Diff;
  unsigned short iVar;
  bool boundary_i, boundary_j;

  Diff = new su2double[nVar];

  nodes->SetUnd_LaplZero();

  for (iEdge = 0; iEdge < geometry->GetnEdge(); iEdge++) {

    iPoint = geometry->edges->GetNode(iEdge,0);
    jPoint = geometry->edges->GetNode(iEdge,1);

    /*--- Solution differences ---*/

    for (iVar = 0; iVar < nVar; iVar++)
      Diff[iVar] = nodes->GetSolution(iPoint,iVar) - nodes->GetSolution(jPoint,iVar);

    boundary_i = geometry->nodes->GetPhysicalBoundary(iPoint);
    boundary_j = geometry->nodes->GetPhysicalBoundary(jPoint);

    /*--- Both points inside the domain, or both in the boundary ---*/

    if ((!boundary_i && !boundary_j) || (boundary_i && boundary_j)) {
      if (geometry->nodes->GetDomain(iPoint)) nodes->SubtractUnd_Lapl(iPoint,Diff);
      if (geometry->nodes->GetDomain(jPoint)) nodes->AddUnd_Lapl(jPoint,Diff);
    }

    /*--- iPoint inside the domain, jPoint on the boundary ---*/

    if (!boundary_i && boundary_j)
      if (geometry->nodes->GetDomain(iPoint)) nodes->SubtractUnd_Lapl(iPoint,Diff);

    /*--- jPoint inside the domain, iPoint on the boundary ---*/

    if (boundary_i && !boundary_j)
      if (geometry->nodes->GetDomain(jPoint)) nodes->AddUnd_Lapl(jPoint,Diff);

  }

  /*--- MPI parallelization ---*/

  InitiateComms(geometry, config, UNDIVIDED_LAPLACIAN);
  CompleteComms(geometry, config, UNDIVIDED_LAPLACIAN);

  delete [] Diff;

}

void CHeatSolver::Centered_Residual(CGeometry *geometry, CSolver **solver_container,  CNumerics **numerics_container,
                                    CConfig *config, unsigned short iMesh, unsigned short iRKStep) {

  CNumerics* numerics = numerics_container[CONV_TERM];

  su2double *V_i, *V_j, Temp_i, Temp_j;
  unsigned long iEdge, iPoint, jPoint;
  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));

  if(flow) {

    nVarFlow = solver_container[FLOW_SOL]->GetnVar();

    for (iEdge = 0; iEdge < geometry->GetnEdge(); iEdge++) {

      /*--- Points in edge ---*/
      iPoint = geometry->edges->GetNode(iEdge,0);
      jPoint = geometry->edges->GetNode(iEdge,1);
      numerics->SetNormal(geometry->edges->GetNormal(iEdge));

      /*--- Primitive variables w/o reconstruction ---*/
      V_i = solver_container[FLOW_SOL]->GetNodes()->GetPrimitive(iPoint);
      V_j = solver_container[FLOW_SOL]->GetNodes()->GetPrimitive(jPoint);

      Temp_i = nodes->GetSolution(iPoint,0);
      Temp_j = nodes->GetSolution(jPoint,0);

      numerics->SetUndivided_Laplacian(nodes->GetUndivided_Laplacian(iPoint), nodes->GetUndivided_Laplacian(jPoint));
      numerics->SetNeighbor(geometry->nodes->GetnNeighbor(iPoint), geometry->nodes->GetnNeighbor(jPoint));

      numerics->SetPrimitive(V_i, V_j);
      numerics->SetTemperature(Temp_i, Temp_j);

      numerics->ComputeResidual(Residual, Jacobian_i, Jacobian_j, config);

      LinSysRes.AddBlock(iPoint, Residual);
      LinSysRes.SubtractBlock(jPoint, Residual);

      /*--- Implicit part ---*/

      Jacobian.UpdateBlocks(iEdge, iPoint, jPoint, Jacobian_i, Jacobian_j);
    }
  }
}

void CHeatSolver::Upwind_Residual(CGeometry *geometry, CSolver **solver_container,
                                  CNumerics **numerics_container, CConfig *config, unsigned short iMesh) {

  CNumerics* numerics = numerics_container[CONV_TERM];

  su2double *V_i, *V_j, Temp_i, Temp_i_Corrected, Temp_j, Temp_j_Corrected, **Gradient_i, **Gradient_j, Project_Grad_i, Project_Grad_j,
            **Temp_i_Grad, **Temp_j_Grad, Project_Temp_i_Grad, Project_Temp_j_Grad;
  unsigned short iDim, iVar;
  unsigned long iEdge, iPoint, jPoint;
  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));
  bool muscl = (config->GetMUSCL_Heat());

  if(flow) {

    nVarFlow = solver_container[FLOW_SOL]->GetnVar();

    for (iEdge = 0; iEdge < geometry->GetnEdge(); iEdge++) {

      /*--- Points in edge ---*/
      iPoint = geometry->edges->GetNode(iEdge,0);
      jPoint = geometry->edges->GetNode(iEdge,1);
      numerics->SetNormal(geometry->edges->GetNormal(iEdge));

      /*--- Primitive variables w/o reconstruction ---*/
      V_i = solver_container[FLOW_SOL]->GetNodes()->GetPrimitive(iPoint);
      V_j = solver_container[FLOW_SOL]->GetNodes()->GetPrimitive(jPoint);

      Temp_i_Grad = nodes->GetGradient(iPoint);
      Temp_j_Grad = nodes->GetGradient(jPoint);
      numerics->SetConsVarGradient(Temp_i_Grad, Temp_j_Grad);

      Temp_i = nodes->GetSolution(iPoint,0);
      Temp_j = nodes->GetSolution(jPoint,0);

      /* Second order reconstruction */
      if (muscl) {

        for (iDim = 0; iDim < nDim; iDim++) {
          Vector_i[iDim] = 0.5*(geometry->nodes->GetCoord(jPoint, iDim) - geometry->nodes->GetCoord(iPoint, iDim));
          Vector_j[iDim] = 0.5*(geometry->nodes->GetCoord(iPoint, iDim) - geometry->nodes->GetCoord(jPoint, iDim));
        }

        Gradient_i = solver_container[FLOW_SOL]->GetNodes()->GetGradient_Reconstruction(iPoint);
        Gradient_j = solver_container[FLOW_SOL]->GetNodes()->GetGradient_Reconstruction(jPoint);
        Temp_i_Grad = nodes->GetGradient_Reconstruction(iPoint);
        Temp_j_Grad = nodes->GetGradient_Reconstruction(jPoint);

        /*Loop to correct the flow variables*/
        for (iVar = 0; iVar < nVarFlow; iVar++) {

          /*Apply the Gradient to get the right temperature value on the edge */
          Project_Grad_i = 0.0; Project_Grad_j = 0.0;
          for (iDim = 0; iDim < nDim; iDim++) {
              Project_Grad_i += Vector_i[iDim]*Gradient_i[iVar][iDim];
              Project_Grad_j += Vector_j[iDim]*Gradient_j[iVar][iDim];
          }

          Primitive_Flow_i[iVar] = V_i[iVar] + Project_Grad_i;
          Primitive_Flow_j[iVar] = V_j[iVar] + Project_Grad_j;
        }

        /* Correct the temperature variables */
        Project_Temp_i_Grad = 0.0; Project_Temp_j_Grad = 0.0;
        for (iDim = 0; iDim < nDim; iDim++) {
            Project_Temp_i_Grad += Vector_i[iDim]*Temp_i_Grad[0][iDim];
            Project_Temp_j_Grad += Vector_j[iDim]*Temp_j_Grad[0][iDim];
        }

        Temp_i_Corrected = Temp_i + Project_Temp_i_Grad;
        Temp_j_Corrected = Temp_j + Project_Temp_j_Grad;

        numerics->SetPrimitive(Primitive_Flow_i, Primitive_Flow_j);
        numerics->SetTemperature(Temp_i_Corrected, Temp_j_Corrected);
      }

      else {

        numerics->SetPrimitive(V_i, V_j);
        numerics->SetTemperature(Temp_i, Temp_j);
      }

      numerics->ComputeResidual(Residual, Jacobian_i, Jacobian_j, config);

      LinSysRes.AddBlock(iPoint, Residual);
      LinSysRes.SubtractBlock(jPoint, Residual);

      /*--- Implicit part ---*/

      Jacobian.UpdateBlocks(iEdge, iPoint, jPoint, Jacobian_i, Jacobian_j);
    }
  }

}

void CHeatSolver::Viscous_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container,
                                   CConfig *config, unsigned short iMesh, unsigned short iRKStep) {

  CNumerics* numerics = numerics_container[VISC_TERM];

  su2double laminar_viscosity, Prandtl_Lam, Prandtl_Turb, eddy_viscosity_i, eddy_viscosity_j,
      thermal_diffusivity_i, thermal_diffusivity_j, Temp_i, Temp_j, **Temp_i_Grad, **Temp_j_Grad;
  unsigned long iEdge, iPoint, jPoint;

  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));

  bool turb = ((config->GetKind_Solver() == INC_RANS) || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));

  eddy_viscosity_i = 0.0;
  eddy_viscosity_j = 0.0;
  laminar_viscosity = config->GetMu_ConstantND();
  Prandtl_Lam = config->GetPrandtl_Lam();
  Prandtl_Turb = config->GetPrandtl_Turb();

  for (iEdge = 0; iEdge < geometry->GetnEdge(); iEdge++) {

    iPoint = geometry->edges->GetNode(iEdge,0);
    jPoint = geometry->edges->GetNode(iEdge,1);

    /*--- Points coordinates, and normal vector ---*/

    numerics->SetCoord(geometry->nodes->GetCoord(iPoint),
                       geometry->nodes->GetCoord(jPoint));
    numerics->SetNormal(geometry->edges->GetNormal(iEdge));

    Temp_i_Grad = nodes->GetGradient(iPoint);
    Temp_j_Grad = nodes->GetGradient(jPoint);
    numerics->SetConsVarGradient(Temp_i_Grad, Temp_j_Grad);

    /*--- Primitive variables w/o reconstruction ---*/
    Temp_i = nodes->GetSolution(iPoint,0);
    Temp_j = nodes->GetSolution(jPoint,0);
    numerics->SetTemperature(Temp_i, Temp_j);

    /*--- Eddy viscosity to compute thermal conductivity ---*/
    if (flow) {
      if (turb) {
        eddy_viscosity_i = solver_container[TURB_SOL]->GetNodes()->GetmuT(iPoint);
        eddy_viscosity_j = solver_container[TURB_SOL]->GetNodes()->GetmuT(jPoint);
      }
      thermal_diffusivity_i = (laminar_viscosity/Prandtl_Lam) + (eddy_viscosity_i/Prandtl_Turb);
      thermal_diffusivity_j = (laminar_viscosity/Prandtl_Lam) + (eddy_viscosity_j/Prandtl_Turb);
    }
    else {
      thermal_diffusivity_i = config->GetThermalDiffusivity_Solid();
      thermal_diffusivity_j = config->GetThermalDiffusivity_Solid();
    }

    numerics->SetThermalDiffusivity(thermal_diffusivity_i,thermal_diffusivity_j);

    /*--- Compute residual, and Jacobians ---*/

    numerics->ComputeResidual(Residual, Jacobian_i, Jacobian_j, config);

    /*--- Add and subtract residual, and update Jacobians ---*/

    LinSysRes.SubtractBlock(iPoint, Residual);
    LinSysRes.AddBlock(jPoint, Residual);

    Jacobian.UpdateBlocksSub(iEdge, iPoint, jPoint, Jacobian_i, Jacobian_j);
  }
}

void CHeatSolver::Set_Heatflux_Areas(CGeometry *geometry, CConfig *config) {

  unsigned short iMarker, iMarker_HeatFlux, Monitoring, iDim;
  unsigned long iPoint, iVertex;
  string HeatFlux_Tag, Marker_Tag;

  su2double *Local_Surface_Areas, Local_HeatFlux_Areas_Monitor, Area, *Normal;
  Local_Surface_Areas = new su2double[config->GetnMarker_HeatFlux()];

  for ( iMarker_HeatFlux = 0; iMarker_HeatFlux < config->GetnMarker_HeatFlux(); iMarker_HeatFlux++ ) {
    Local_Surface_Areas[iMarker_HeatFlux] = 0.0;
  }
  Local_HeatFlux_Areas_Monitor = 0.0;

  for (iMarker = 0; iMarker < nMarker; iMarker++) {

    Monitoring = config->GetMarker_All_Monitoring(iMarker);

    for ( iMarker_HeatFlux = 0; iMarker_HeatFlux < config->GetnMarker_HeatFlux(); iMarker_HeatFlux++ ) {

      HeatFlux_Tag = config->GetMarker_HeatFlux_TagBound(iMarker_HeatFlux);
      Marker_Tag = config->GetMarker_All_TagBound(iMarker);

      if (Marker_Tag == HeatFlux_Tag) {

        Local_Surface_Areas[iMarker_HeatFlux] = 0.0;

        for( iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++ ) {

          iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

          if(geometry->nodes->GetDomain(iPoint)) {

            Normal = geometry->vertex[iMarker][iVertex]->GetNormal();
            Area = 0.0;
            for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim];
            Area = sqrt(Area);

            Local_Surface_Areas[iMarker_HeatFlux] += Area;

            if(Monitoring == YES) {
              Local_HeatFlux_Areas_Monitor += Area;
            }
          }
        }
      }
    }
  }

  SU2_MPI::Allreduce(Local_Surface_Areas, Surface_Areas, config->GetnMarker_HeatFlux(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  SU2_MPI::Allreduce(&Local_HeatFlux_Areas_Monitor, &Total_HeatFlux_Areas_Monitor, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  Total_HeatFlux_Areas = 0.0;
  for( iMarker_HeatFlux = 0; iMarker_HeatFlux < config->GetnMarker_HeatFlux(); iMarker_HeatFlux++ ) {
    Total_HeatFlux_Areas += Surface_Areas[iMarker_HeatFlux];
  }

  delete[] Local_Surface_Areas;
}

void CHeatSolver::BC_Isothermal_Wall(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config,
                                       unsigned short val_marker) {

  unsigned long iPoint, iVertex, Point_Normal;
  unsigned short iDim;
  su2double *Normal, *Coord_i, *Coord_j, Area, dist_ij, laminar_viscosity, thermal_diffusivity, Twall, dTdn, Prandtl_Lam;
  //su2double Prandtl_Turb;
  bool implicit = (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT);

  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));

  Prandtl_Lam = config->GetPrandtl_Lam();
//  Prandtl_Turb = config->GetPrandtl_Turb();
  laminar_viscosity = config->GetMu_ConstantND();
  //Prandtl_Turb = config->GetPrandtl_Turb();
  //laminar_viscosity = config->GetViscosity_FreeStreamND(); // TDE check for consistency for CHT

  string Marker_Tag = config->GetMarker_All_TagBound(val_marker);

  Twall = config->GetIsothermal_Temperature(Marker_Tag)/config->GetTemperature_Ref();

  for (iVertex = 0; iVertex < geometry->nVertex[val_marker]; iVertex++) {

    iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    if (geometry->nodes->GetDomain(iPoint)) {

        Point_Normal = geometry->vertex[val_marker][iVertex]->GetNormal_Neighbor();

        Normal = geometry->vertex[val_marker][iVertex]->GetNormal();
        Area = 0.0;
        for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim];
        Area = sqrt (Area);

        Coord_i = geometry->nodes->GetCoord(iPoint);
        Coord_j = geometry->nodes->GetCoord(Point_Normal);
        dist_ij = 0;
        for (iDim = 0; iDim < nDim; iDim++)
          dist_ij += (Coord_j[iDim]-Coord_i[iDim])*(Coord_j[iDim]-Coord_i[iDim]);
        dist_ij = sqrt(dist_ij);

        dTdn = -(nodes->GetSolution(Point_Normal,0) - Twall)/dist_ij;

        if(flow) {
          thermal_diffusivity = laminar_viscosity/Prandtl_Lam;
        }
        else
          thermal_diffusivity = config->GetThermalDiffusivity_Solid();

        Res_Visc[0] = thermal_diffusivity*dTdn*Area;

        if(implicit) {

          Jacobian_i[0][0] = -thermal_diffusivity/dist_ij * Area;
        }

        LinSysRes.SubtractBlock(iPoint, Res_Visc);
        Jacobian.SubtractBlock2Diag(iPoint, Jacobian_i);
    }
  }
}

void CHeatSolver::BC_HeatFlux_Wall(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config,
                                                     unsigned short val_marker) {

  unsigned short iDim;
  unsigned long iVertex, iPoint;
  su2double Wall_HeatFlux, Area, *Normal;

  string Marker_Tag = config->GetMarker_All_TagBound(val_marker);
  Wall_HeatFlux = config->GetWall_HeatFlux(Marker_Tag);

  if(config->GetIntegrated_HeatFlux()) {

    unsigned short iMarker_HeatFlux;
    string HeatFlux_Tag, Marker_Tag;

    // Find out which heat flux wall to get the right surface area

    for ( iMarker_HeatFlux = 0; iMarker_HeatFlux < config->GetnMarker_HeatFlux(); iMarker_HeatFlux++ ) {

      HeatFlux_Tag = config->GetMarker_HeatFlux_TagBound(iMarker_HeatFlux);
      Marker_Tag = config->GetMarker_All_TagBound(val_marker);

      if (Marker_Tag == HeatFlux_Tag) {
        Wall_HeatFlux = Wall_HeatFlux / Surface_Areas[iMarker_HeatFlux];
      }
    }
  }

  Wall_HeatFlux = Wall_HeatFlux/config->GetHeat_Flux_Ref();

  for (iVertex = 0; iVertex < geometry->nVertex[val_marker]; iVertex++) {

    iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    if (geometry->nodes->GetDomain(iPoint)) {

      Normal = geometry->vertex[val_marker][iVertex]->GetNormal();
      Area = 0.0;
      for (iDim = 0; iDim < nDim; iDim++)
        Area += Normal[iDim]*Normal[iDim];
      Area = sqrt (Area);

      Res_Visc[0] = 0.0;

      Res_Visc[0] = Wall_HeatFlux * Area;

      /*--- Viscous contribution to the residual at the wall ---*/

      LinSysRes.SubtractBlock(iPoint, Res_Visc);
    }

  }
}

void CHeatSolver::BC_Inlet(CGeometry *geometry, CSolver **solver_container,
                            CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  unsigned short iDim;
  unsigned long iVertex, iPoint, Point_Normal;
  su2double *Flow_Dir,  Vel_Mag;
  su2double *V_inlet, *V_domain;

  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));

  bool viscous              = config->GetViscous();
  bool implicit             = (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT);
  string Marker_Tag         = config->GetMarker_All_TagBound(val_marker);

  su2double *Normal = new su2double[nDim];

  su2double *Coord_i, *Coord_j, Area, dist_ij, laminar_viscosity, thermal_diffusivity, Twall, dTdn, Prandtl_Lam;
  //su2double Prandtl_Turb;
  Prandtl_Lam = config->GetPrandtl_Lam();
//  Prandtl_Turb = config->GetPrandtl_Turb();
  laminar_viscosity = config->GetMu_ConstantND();
  //laminar_viscosity = config->GetViscosity_FreeStreamND(); //TDE check for consistency with CHT

  Twall = config->GetTemperature_FreeStreamND();

  for (iVertex = 0; iVertex < geometry->nVertex[val_marker]; iVertex++) {

    iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    if (geometry->nodes->GetDomain(iPoint)) {

      geometry->vertex[val_marker][iVertex]->GetNormal(Normal);
      for (iDim = 0; iDim < nDim; iDim++) Normal[iDim] = -Normal[iDim];

      if(flow) {

        /*--- Normal vector for this vertex (negate for outward convention) ---*/

        conv_numerics->SetNormal(Normal);

        /*--- Retrieve solution at this boundary node ---*/

        V_domain = solver_container[FLOW_SOL]->GetNodes()->GetPrimitive(iPoint);

        /*--- Retrieve the specified velocity for the inlet. ---*/

        Vel_Mag  = config->GetInlet_Ptotal(Marker_Tag)/config->GetVelocity_Ref();
        Flow_Dir = config->GetInlet_FlowDir(Marker_Tag);

        V_inlet = solver_container[FLOW_SOL]->GetCharacPrimVar(val_marker, iVertex);

        for (iDim = 0; iDim < nDim; iDim++)
          V_inlet[iDim+1] = Vel_Mag*Flow_Dir[iDim];

        conv_numerics->SetPrimitive(V_domain, V_inlet);

        if (dynamic_grid)
          conv_numerics->SetGridVel(geometry->nodes->GetGridVel(iPoint), geometry->nodes->GetGridVel(iPoint));

        conv_numerics->SetTemperature(nodes->GetSolution(iPoint,0), config->GetInlet_Ttotal(Marker_Tag)/config->GetTemperature_Ref());

        /*--- Compute the residual using an upwind scheme ---*/

        conv_numerics->ComputeResidual(Residual, Jacobian_i, Jacobian_j, config);

        /*--- Update residual value ---*/

        LinSysRes.AddBlock(iPoint, Residual);

        /*--- Jacobian contribution for implicit integration ---*/

        if (implicit)
          Jacobian.AddBlock2Diag(iPoint, Jacobian_i);
      }

      /*--- Viscous contribution ---*/

      if (viscous) {

        Point_Normal = geometry->vertex[val_marker][iVertex]->GetNormal_Neighbor();

        geometry->vertex[val_marker][iVertex]->GetNormal(Normal);
        Area = 0.0;
        for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim];
        Area = sqrt (Area);

        Coord_i = geometry->nodes->GetCoord(iPoint);
        Coord_j = geometry->nodes->GetCoord(Point_Normal);
        dist_ij = 0;
        for (iDim = 0; iDim < nDim; iDim++)
          dist_ij += (Coord_j[iDim]-Coord_i[iDim])*(Coord_j[iDim]-Coord_i[iDim]);
        dist_ij = sqrt(dist_ij);

        dTdn = -(nodes->GetSolution(Point_Normal,0) - Twall)/dist_ij;

        thermal_diffusivity = laminar_viscosity/Prandtl_Lam;

        Res_Visc[0] = thermal_diffusivity*dTdn*Area;

        if(implicit) {

          Jacobian_i[0][0] = -thermal_diffusivity/dist_ij * Area;
        }
        /*--- Viscous contribution to the residual at the wall ---*/

        LinSysRes.SubtractBlock(iPoint, Res_Visc);
        Jacobian.SubtractBlock2Diag(iPoint, Jacobian_i);
      }
    }
  }

  /*--- Free locally allocated memory ---*/
  delete [] Normal;

}

void CHeatSolver::BC_Outlet(CGeometry *geometry, CSolver **solver_container,
                             CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  unsigned short iDim;
  unsigned long iVertex, iPoint, Point_Normal;
  su2double *V_outlet, *V_domain;

  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));
  bool implicit             = (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT);

  su2double *Normal = new su2double[nDim];

  for (iVertex = 0; iVertex < geometry->nVertex[val_marker]; iVertex++) {

    iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    if (geometry->nodes->GetDomain(iPoint)) {

      Point_Normal = geometry->vertex[val_marker][iVertex]->GetNormal_Neighbor();

      /*--- Normal vector for this vertex (negate for outward convention) ---*/

      geometry->vertex[val_marker][iVertex]->GetNormal(Normal);
      for (iDim = 0; iDim < nDim; iDim++) Normal[iDim] = -Normal[iDim];

      if(flow) {
          conv_numerics->SetNormal(Normal);

          /*--- Retrieve solution at this boundary node ---*/

          V_domain = solver_container[FLOW_SOL]->GetNodes()->GetPrimitive(iPoint);

          /*--- Retrieve the specified velocity for the inlet. ---*/

          V_outlet = solver_container[FLOW_SOL]->GetCharacPrimVar(val_marker, iVertex);
          for (iDim = 0; iDim < nDim; iDim++)
            V_outlet[iDim+1] = solver_container[FLOW_SOL]->GetNodes()->GetVelocity(Point_Normal, iDim);

          conv_numerics->SetPrimitive(V_domain, V_outlet);

          if (dynamic_grid)
            conv_numerics->SetGridVel(geometry->nodes->GetGridVel(iPoint), geometry->nodes->GetGridVel(iPoint));

          conv_numerics->SetTemperature(nodes->GetSolution(iPoint,0), nodes->GetSolution(Point_Normal,0));

          /*--- Compute the residual using an upwind scheme ---*/

          conv_numerics->ComputeResidual(Residual, Jacobian_i, Jacobian_j, config);

          /*--- Update residual value ---*/

          LinSysRes.AddBlock(iPoint, Residual);

          /*--- Jacobian contribution for implicit integration ---*/

          if (implicit)
            Jacobian.AddBlock2Diag(iPoint, Jacobian_i);
      }
    }
  }

  /*--- Free locally allocated memory ---*/
  delete [] Normal;

}

void CHeatSolver::BC_ConjugateHeat_Interface(CGeometry *geometry, CSolver **solver_container, CNumerics *numerics, CConfig *config, unsigned short val_marker) {

  unsigned long iVertex, iPoint, total_index;
  unsigned short iDim, iVar;

  su2double thermal_diffusivity, rho_cp_solid, Temperature_Ref, T_Conjugate, Tinterface,
      Tnormal_Conjugate, HeatFluxDensity, HeatFlux, Area;

  bool implicit = (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT);
  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));

  su2double *Normal = new su2double[nDim];

  Temperature_Ref       = config->GetTemperature_Ref();
  rho_cp_solid          = config->GetDensity_Solid()*config->GetSpecific_Heat_Cp();

  if (flow) {

    for (iVertex = 0; iVertex < geometry->nVertex[val_marker]; iVertex++) {

      iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

      if (geometry->nodes->GetDomain(iPoint)) {

        Normal = geometry->vertex[val_marker][iVertex]->GetNormal();
        Area = 0.0;
        for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim];
        Area = sqrt (Area);

        T_Conjugate = GetConjugateHeatVariable(val_marker, iVertex, 0)/Temperature_Ref;

        nodes->SetSolution_Old(iPoint,&T_Conjugate);
        LinSysRes.SetBlock_Zero(iPoint, 0);
        nodes->SetRes_TruncErrorZero(iPoint);

        if (implicit) {
          for (iVar = 0; iVar < nVar; iVar++) {
            total_index = iPoint*nVar+iVar;
            Jacobian.DeleteValsRowi(total_index);
          }
        }
      }
    }
  }
  else {

    for (iVertex = 0; iVertex < geometry->nVertex[val_marker]; iVertex++) {

      iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

      if (geometry->nodes->GetDomain(iPoint)) {

        Normal = geometry->vertex[val_marker][iVertex]->GetNormal();
        Area = 0.0;
        for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim];
        Area = sqrt(Area);

        thermal_diffusivity = GetConjugateHeatVariable(val_marker, iVertex, 2)/rho_cp_solid;

        if ((config->GetKind_CHT_Coupling() == DIRECT_TEMPERATURE_ROBIN_HEATFLUX) ||
            (config->GetKind_CHT_Coupling() == AVERAGED_TEMPERATURE_ROBIN_HEATFLUX)) {

          Tinterface        = nodes->GetSolution(iPoint,0);
          Tnormal_Conjugate = GetConjugateHeatVariable(val_marker, iVertex, 3)/Temperature_Ref;

          HeatFluxDensity   = thermal_diffusivity*(Tinterface - Tnormal_Conjugate);
          HeatFlux          = HeatFluxDensity * Area;
        }
        else {

          HeatFluxDensity = GetConjugateHeatVariable(val_marker, iVertex, 1)/config->GetHeat_Flux_Ref();
          HeatFlux        = HeatFluxDensity*Area;
        }

        Res_Visc[0] = -HeatFlux;
        LinSysRes.SubtractBlock(iPoint, Res_Visc);

        if (implicit) {

          Jacobian_i[0][0] = thermal_diffusivity*Area;
          Jacobian.SubtractBlock2Diag(iPoint, Jacobian_i);
        }
      }
    }
  }
}

void CHeatSolver::Heat_Fluxes(CGeometry *geometry, CSolver **solver_container, CConfig *config) {

  unsigned long iVertex, iPoint, iPointNormal;
  unsigned short Boundary, Monitoring, iMarker, iDim;
  su2double *Coord, *Coord_Normal, *Normal, Area, dist, Twall, dTdn, thermal_diffusivity;
  string Marker_Tag, HeatFlux_Tag;

  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));

#ifdef HAVE_MPI
  su2double MyAllBound_HeatFlux, MyAllBound_AverageT;
#endif

  AllBound_HeatFlux = 0.0;
  AllBound_AverageT = 0.0;

  for ( iMarker = 0; iMarker < nMarker; iMarker++ ) {

    AverageT_per_Marker[iMarker]  = 0.0;
    HeatFlux_per_Marker[iMarker]  = 0.0;

    Boundary = config->GetMarker_All_KindBC(iMarker);
    Marker_Tag = config->GetMarker_All_TagBound(iMarker);
    Monitoring = config->GetMarker_All_Monitoring(iMarker);

    if ( Boundary == ISOTHERMAL ) {

      Twall = config->GetIsothermal_Temperature(Marker_Tag)/config->GetTemperature_Ref();

      for( iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++ ) {

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

        if(geometry->nodes->GetDomain(iPoint)) {

          iPointNormal = geometry->vertex[iMarker][iVertex]->GetNormal_Neighbor();

          Coord = geometry->nodes->GetCoord(iPoint);
          Coord_Normal = geometry->nodes->GetCoord(iPointNormal);

          Normal = geometry->vertex[iMarker][iVertex]->GetNormal();
          Area = 0.0;
          for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim];
          Area = sqrt(Area);

          dist = 0.0;
          for (iDim = 0; iDim < nDim; iDim++) dist += (Coord_Normal[iDim]-Coord[iDim])*(Coord_Normal[iDim]-Coord[iDim]);
          dist = sqrt(dist);

          dTdn = (Twall - nodes->GetSolution(iPointNormal,0))/dist;

          if(flow) {
            thermal_diffusivity = config->GetViscosity_FreeStreamND()/config->GetPrandtl_Lam();
          }
          else {
            thermal_diffusivity = config->GetThermalDiffusivity_Solid();
          }

          HeatFlux[iMarker][iVertex] = thermal_diffusivity*dTdn*config->GetHeat_Flux_Ref();

          HeatFlux_per_Marker[iMarker] += HeatFlux[iMarker][iVertex]*Area;

        }
      }
    }
    else if ( Boundary == CHT_WALL_INTERFACE || Boundary == HEAT_FLUX ) {

      for( iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++ ) {

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

        if(geometry->nodes->GetDomain(iPoint)) {

          iPointNormal = geometry->vertex[iMarker][iVertex]->GetNormal_Neighbor();

          Twall = nodes->GetSolution(iPoint,0);

          Coord = geometry->nodes->GetCoord(iPoint);
          Coord_Normal = geometry->nodes->GetCoord(iPointNormal);

          Normal = geometry->vertex[iMarker][iVertex]->GetNormal();
          Area = 0.0;
          for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim];
          Area = sqrt(Area);

          dist = 0.0;
          for (iDim = 0; iDim < nDim; iDim++) dist += (Coord_Normal[iDim]-Coord[iDim])*(Coord_Normal[iDim]-Coord[iDim]);
          dist = sqrt(dist);

          dTdn = (Twall - nodes->GetSolution(iPointNormal,0))/dist;

          if(flow) {
            thermal_diffusivity = config->GetViscosity_FreeStreamND()/config->GetPrandtl_Lam();
          }
          else {
            thermal_diffusivity = config->GetThermalDiffusivity_Solid();
          }

          HeatFlux[iMarker][iVertex] = thermal_diffusivity*dTdn*config->GetHeat_Flux_Ref();

          HeatFlux_per_Marker[iMarker] += HeatFlux[iMarker][iVertex]*Area;

          /*--- We do only aim to compute averaged temperatures on the (interesting) heat flux walls ---*/

          if ( Boundary == HEAT_FLUX ) {

            AverageT_per_Marker[iMarker] += Twall*config->GetTemperature_Ref()*Area;
          }
        }
      }
    }

    if (Monitoring == YES) {

      AllBound_HeatFlux += HeatFlux_per_Marker[iMarker];
      AllBound_AverageT += AverageT_per_Marker[iMarker];
    }
  }

#ifdef HAVE_MPI
  MyAllBound_HeatFlux = AllBound_HeatFlux;
  MyAllBound_AverageT = AllBound_AverageT;
  SU2_MPI::Allreduce(&MyAllBound_HeatFlux, &AllBound_HeatFlux, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  SU2_MPI::Allreduce(&MyAllBound_AverageT, &AllBound_AverageT, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (Total_HeatFlux_Areas_Monitor != 0.0) {
    Total_AverageT = AllBound_AverageT/Total_HeatFlux_Areas_Monitor;
  }
  else {
    Total_AverageT = 0.0;
  }

  Total_HeatFlux = AllBound_HeatFlux;
}

void CHeatSolver::SetTime_Step(CGeometry *geometry, CSolver **solver_container, CConfig *config,
                               unsigned short iMesh, unsigned long Iteration) {

  unsigned short iDim, iMarker;
  unsigned long iEdge, iVertex, iPoint = 0, jPoint = 0;
  su2double Area, Vol, laminar_viscosity, eddy_viscosity, thermal_diffusivity, Prandtl_Lam, Prandtl_Turb, Mean_ProjVel, Mean_BetaInc2, Mean_DensityInc, Mean_SoundSpeed, Lambda;
  su2double Global_Delta_Time = 0.0, Global_Delta_UnstTimeND = 0.0, Local_Delta_Time = 0.0, Local_Delta_Time_Inv, Local_Delta_Time_Visc, CFL_Reduction, K_v = 0.25;
  const su2double* Normal;

  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));

  bool turb = ((config->GetKind_Solver() == INC_RANS) || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));
  bool dual_time = ((config->GetTime_Marching() == DT_STEPPING_1ST) ||
                    (config->GetTime_Marching() == DT_STEPPING_2ND));
  bool implicit = (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT);

  eddy_viscosity    = 0.0;
  laminar_viscosity = config->GetMu_ConstantND();
  Prandtl_Lam = config->GetPrandtl_Lam();
  Prandtl_Turb = config->GetPrandtl_Turb();

  thermal_diffusivity = config->GetThermalDiffusivity_Solid();

  /*--- Compute spectral radius based on thermal conductivity ---*/

  Min_Delta_Time = 1.E30; Max_Delta_Time = 0.0;
  CFL_Reduction = config->GetCFLRedCoeff_Turb();

  for (iPoint = 0; iPoint < nPointDomain; iPoint++) {
    nodes->SetMax_Lambda_Inv(iPoint,0.0);
    nodes->SetMax_Lambda_Visc(iPoint,0.0);
  }

  /*--- Loop interior edges ---*/

  for (iEdge = 0; iEdge < geometry->GetnEdge(); iEdge++) {

    iPoint = geometry->edges->GetNode(iEdge,0);
    jPoint = geometry->edges->GetNode(iEdge,1);

    /*--- get the edge's normal vector to compute the edge's area ---*/
    Normal = geometry->edges->GetNormal(iEdge);
    Area = 0; for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim]; Area = sqrt(Area);

    /*--- Inviscid contribution ---*/

    if (flow) {
      Mean_ProjVel = 0.5 * (solver_container[FLOW_SOL]->GetNodes()->GetProjVel(iPoint,Normal) + solver_container[FLOW_SOL]->GetNodes()->GetProjVel(jPoint,Normal));
      Mean_BetaInc2 = 0.5 * (solver_container[FLOW_SOL]->GetNodes()->GetBetaInc2(iPoint) + solver_container[FLOW_SOL]->GetNodes()->GetBetaInc2(jPoint));
      Mean_DensityInc = 0.5 * (solver_container[FLOW_SOL]->GetNodes()->GetDensity(iPoint) + solver_container[FLOW_SOL]->GetNodes()->GetDensity(jPoint));
      Mean_SoundSpeed = sqrt(Mean_ProjVel*Mean_ProjVel + (Mean_BetaInc2/Mean_DensityInc)*Area*Area);

      Lambda = fabs(Mean_ProjVel) + Mean_SoundSpeed;
      if (geometry->nodes->GetDomain(iPoint)) nodes->AddMax_Lambda_Inv(iPoint, Lambda);
      if (geometry->nodes->GetDomain(jPoint)) nodes->AddMax_Lambda_Inv(jPoint, Lambda);
    }

    /*--- Viscous contribution ---*/

    thermal_diffusivity = config->GetThermalDiffusivity_Solid();
    if(flow) {
      if(turb) {
        eddy_viscosity = solver_container[TURB_SOL]->GetNodes()->GetmuT(iPoint);
      }

      thermal_diffusivity = laminar_viscosity/Prandtl_Lam + eddy_viscosity/Prandtl_Turb;
    }

    Lambda = thermal_diffusivity*Area*Area;
    if (geometry->nodes->GetDomain(iPoint)) nodes->AddMax_Lambda_Visc(iPoint, Lambda);
    if (geometry->nodes->GetDomain(jPoint)) nodes->AddMax_Lambda_Visc(jPoint, Lambda);

  }

  /*--- Loop boundary edges ---*/

  for (iMarker = 0; iMarker < geometry->GetnMarker(); iMarker++) {
    for (iVertex = 0; iVertex < geometry->GetnVertex(iMarker); iVertex++) {

      /*--- Point identification, Normal vector and area ---*/

      iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
      Normal = geometry->vertex[iMarker][iVertex]->GetNormal();
      Area = 0.0; for (iDim = 0; iDim < nDim; iDim++) Area += Normal[iDim]*Normal[iDim]; Area = sqrt(Area);

      /*--- Inviscid contribution ---*/

      if (flow) {
        Mean_ProjVel = solver_container[FLOW_SOL]->GetNodes()->GetProjVel(iPoint, Normal);
        Mean_BetaInc2 = solver_container[FLOW_SOL]->GetNodes()->GetBetaInc2(iPoint);
        Mean_DensityInc = solver_container[FLOW_SOL]->GetNodes()->GetDensity(iPoint);
        Mean_SoundSpeed = sqrt(Mean_ProjVel*Mean_ProjVel + (Mean_BetaInc2/Mean_DensityInc)*Area*Area);

        Lambda = fabs(Mean_ProjVel) + Mean_SoundSpeed;
        if (geometry->nodes->GetDomain(iPoint)) nodes->AddMax_Lambda_Inv(iPoint, Lambda);
      }

      /*--- Viscous contribution ---*/

      thermal_diffusivity = config->GetThermalDiffusivity_Solid();
      if(flow) {
        if(turb) {
          eddy_viscosity = solver_container[TURB_SOL]->GetNodes()->GetmuT(iPoint);
        }

        thermal_diffusivity = laminar_viscosity/Prandtl_Lam + eddy_viscosity/Prandtl_Turb;
      }

      Lambda = thermal_diffusivity*Area*Area;
      if (geometry->nodes->GetDomain(iPoint)) nodes->AddMax_Lambda_Visc(iPoint, Lambda);

    }
  }

  /*--- Each element uses their own speed, steady state simulation ---*/

  for (iPoint = 0; iPoint < nPointDomain; iPoint++) {

    Vol = geometry->nodes->GetVolume(iPoint);

    if (Vol != 0.0) {

      if(flow) {
        Local_Delta_Time_Inv = config->GetCFL(iMesh)*Vol / nodes->GetMax_Lambda_Inv(iPoint);
        Local_Delta_Time_Visc = config->GetCFL(iMesh)*K_v*Vol*Vol/ nodes->GetMax_Lambda_Visc(iPoint);
      }
      else {
        Local_Delta_Time_Inv = config->GetMax_DeltaTime();
        Local_Delta_Time_Visc = config->GetCFL(iMesh)*K_v*Vol*Vol/ nodes->GetMax_Lambda_Visc(iPoint);
        //Local_Delta_Time_Visc = 100.0*K_v*Vol*Vol/ nodes->GetMax_Lambda_Visc(iPoint);
      }

      /*--- Time step setting method ---*/

      if (config->GetKind_TimeStep_Heat() == BYFLOW && flow) {
        Local_Delta_Time = solver_container[FLOW_SOL]->GetNodes()->GetDelta_Time(iPoint);
      }
      else if (config->GetKind_TimeStep_Heat() == MINIMUM) {
        Local_Delta_Time = min(Local_Delta_Time_Inv, Local_Delta_Time_Visc);
      }
      else if (config->GetKind_TimeStep_Heat() == CONVECTIVE) {
        Local_Delta_Time = Local_Delta_Time_Inv;
      }
      else if (config->GetKind_TimeStep_Heat() == VISCOUS) {
        Local_Delta_Time = Local_Delta_Time_Visc;
      }

      /*--- Min-Max-Logic ---*/

      Global_Delta_Time = min(Global_Delta_Time, Local_Delta_Time);
      Min_Delta_Time = min(Min_Delta_Time, Local_Delta_Time);
      Max_Delta_Time = max(Max_Delta_Time, Local_Delta_Time);
      if (Local_Delta_Time > config->GetMax_DeltaTime())
        Local_Delta_Time = config->GetMax_DeltaTime();

      nodes->SetDelta_Time(iPoint,CFL_Reduction*Local_Delta_Time);
    }
    else {
      nodes->SetDelta_Time(iPoint,0.0);
    }
  }

  /*--- Compute the max and the min dt (in parallel) ---*/
  if (config->GetComm_Level() == COMM_FULL) {
#ifdef HAVE_MPI
    su2double rbuf_time, sbuf_time;
    sbuf_time = Min_Delta_Time;
    SU2_MPI::Reduce(&sbuf_time, &rbuf_time, 1, MPI_DOUBLE, MPI_MIN, MASTER_NODE, MPI_COMM_WORLD);
    SU2_MPI::Bcast(&rbuf_time, 1, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
    Min_Delta_Time = rbuf_time;

    sbuf_time = Max_Delta_Time;
    SU2_MPI::Reduce(&sbuf_time, &rbuf_time, 1, MPI_DOUBLE, MPI_MAX, MASTER_NODE, MPI_COMM_WORLD);
    SU2_MPI::Bcast(&rbuf_time, 1, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
    Max_Delta_Time = rbuf_time;
#endif
  }

  /*--- For exact time solution use the minimum delta time of the whole mesh ---*/
  if (config->GetTime_Marching() == TIME_STEPPING) {
#ifdef HAVE_MPI
    su2double rbuf_time, sbuf_time;
    sbuf_time = Global_Delta_Time;
    SU2_MPI::Reduce(&sbuf_time, &rbuf_time, 1, MPI_DOUBLE, MPI_MIN, MASTER_NODE, MPI_COMM_WORLD);
    SU2_MPI::Bcast(&rbuf_time, 1, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
    Global_Delta_Time = rbuf_time;
#endif
    for (iPoint = 0; iPoint < nPointDomain; iPoint++)
      nodes->SetDelta_Time(iPoint,Global_Delta_Time);
  }

  /*--- Recompute the unsteady time step for the dual time strategy
   if the unsteady CFL is diferent from 0 ---*/
  if ((dual_time) && (Iteration == 0) && (config->GetUnst_CFL() != 0.0) && (iMesh == MESH_0)) {
    Global_Delta_UnstTimeND = config->GetUnst_CFL()*Global_Delta_Time/config->GetCFL(iMesh);

#ifdef HAVE_MPI
    su2double rbuf_time, sbuf_time;
    sbuf_time = Global_Delta_UnstTimeND;
    SU2_MPI::Reduce(&sbuf_time, &rbuf_time, 1, MPI_DOUBLE, MPI_MIN, MASTER_NODE, MPI_COMM_WORLD);
    SU2_MPI::Bcast(&rbuf_time, 1, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
    Global_Delta_UnstTimeND = rbuf_time;
#endif
    config->SetDelta_UnstTimeND(Global_Delta_UnstTimeND);
  }

  /*--- The pseudo local time (explicit integration) cannot be greater than the physical time ---*/
  if (dual_time)
    for (iPoint = 0; iPoint < nPointDomain; iPoint++) {
      if (!implicit) {
        cout << "Using unsteady time: " << config->GetDelta_UnstTimeND() << endl;
        Local_Delta_Time = min((2.0/3.0)*config->GetDelta_UnstTimeND(), nodes->GetDelta_Time(iPoint));
        nodes->SetDelta_Time(iPoint,Local_Delta_Time);
      }
  }
}

void CHeatSolver::ExplicitEuler_Iteration(CGeometry *geometry, CSolver **solver_container, CConfig *config) {

  su2double *local_Residual, *local_Res_TruncError, Vol, Delta, Res;
  unsigned short iVar;
  unsigned long iPoint;

  bool adjoint = config->GetContinuous_Adjoint();

  for (iVar = 0; iVar < nVar; iVar++) {
    SetRes_RMS(iVar, 0.0);
    SetRes_Max(iVar, 0.0, 0);
  }

  /*--- Update the solution ---*/

  for (iPoint = 0; iPoint < nPointDomain; iPoint++) {
    Vol = geometry->nodes->GetVolume(iPoint);
    Delta = nodes->GetDelta_Time(iPoint) / Vol;

    local_Res_TruncError = nodes->GetResTruncError(iPoint);
    local_Residual = LinSysRes.GetBlock(iPoint);

    if (!adjoint) {
      for (iVar = 0; iVar < nVar; iVar++) {
        Res = local_Residual[iVar] + local_Res_TruncError[iVar];
        nodes->AddSolution(iPoint,iVar, -Res*Delta);
        AddRes_RMS(iVar, Res*Res);
        AddRes_Max(iVar, fabs(Res), geometry->nodes->GetGlobalIndex(iPoint), geometry->nodes->GetCoord(iPoint));
      }
    }

  }

  /*--- MPI solution ---*/

  InitiateComms(geometry, config, SOLUTION);
  CompleteComms(geometry, config, SOLUTION);

  /*--- Compute the root mean square residual ---*/

  SetResidual_RMS(geometry, config);

}


void CHeatSolver::ImplicitEuler_Iteration(CGeometry *geometry, CSolver **solver_container, CConfig *config) {

  unsigned short iVar;
  unsigned long iPoint, total_index;
  su2double Delta, Vol, *local_Res_TruncError;
  bool flow = ((config->GetKind_Solver() == INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == INC_RANS)
               || (config->GetKind_Solver() == DISC_ADJ_INC_NAVIER_STOKES)
               || (config->GetKind_Solver() == DISC_ADJ_INC_RANS));


  /*--- Set maximum residual to zero ---*/

  for (iVar = 0; iVar < nVar; iVar++) {
    SetRes_RMS(iVar, 0.0);
    SetRes_Max(iVar, 0.0, 0);
  }

  /*--- Build implicit system ---*/

  for (iPoint = 0; iPoint < nPointDomain; iPoint++) {

    /*--- Read the residual ---*/

    local_Res_TruncError = nodes->GetResTruncError(iPoint);

    /*--- Read the volume ---*/

    Vol = geometry->nodes->GetVolume(iPoint);

    /*--- Modify matrix diagonal to assure diagonal dominance ---*/

    if (nodes->GetDelta_Time(iPoint) != 0.0) {

      if(flow) {
        Delta = Vol / nodes->GetDelta_Time(iPoint);
        Jacobian.AddVal2Diag(iPoint, Delta);
      }
      else {
        Delta = Vol / nodes->GetDelta_Time(iPoint);
        Jacobian.AddVal2Diag(iPoint, Delta);
      }

    } else {
      Jacobian.SetVal2Diag(iPoint, 1.0);
      for (iVar = 0; iVar < nVar; iVar++) {
        total_index = iPoint*nVar + iVar;
        LinSysRes[total_index] = 0.0;
        local_Res_TruncError[iVar] = 0.0;
      }
    }

    /*--- Right hand side of the system (-Residual) and initial guess (x = 0) ---*/

    for (iVar = 0; iVar < nVar; iVar++) {
      total_index = iPoint*nVar+iVar;
      LinSysRes[total_index] = - (LinSysRes[total_index] + local_Res_TruncError[iVar]);
      LinSysSol[total_index] = 0.0;
      AddRes_RMS(iVar, LinSysRes[total_index]*LinSysRes[total_index]);
      AddRes_Max(iVar, fabs(LinSysRes[total_index]), geometry->nodes->GetGlobalIndex(iPoint), geometry->nodes->GetCoord(iPoint));
    }
  }

  /*--- Initialize residual and solution at the ghost points ---*/

  for (iPoint = nPointDomain; iPoint < nPoint; iPoint++) {
    for (iVar = 0; iVar < nVar; iVar++) {
      total_index = iPoint*nVar + iVar;
      LinSysRes[total_index] = 0.0;
      LinSysSol[total_index] = 0.0;
    }
  }

  /*--- Solve or smooth the linear system ---*/

  System.Solve(Jacobian, LinSysRes, LinSysSol, geometry, config);

  for (iPoint = 0; iPoint < nPointDomain; iPoint++) {
    for (iVar = 0; iVar < nVar; iVar++) {
      nodes->AddSolution(iPoint,iVar, LinSysSol[iPoint*nVar+iVar]);
    }
  }

  /*--- MPI solution ---*/

  InitiateComms(geometry, config, SOLUTION);
  CompleteComms(geometry, config, SOLUTION);

  /*--- Compute the root mean square residual ---*/

  SetResidual_RMS(geometry, config);

}

void CHeatSolver::SetInitialCondition(CGeometry **geometry, CSolver ***solver_container, CConfig *config, unsigned long TimeIter) {

  unsigned long iPoint, Point_Fine;
  unsigned short iMesh, iChildren, iVar;
  su2double Area_Children, Area_Parent, *Solution_Fine, *Solution;

  bool restart   = (config->GetRestart() || config->GetRestart_Flow());
  bool dual_time = ((config->GetTime_Marching() == DT_STEPPING_1ST) ||
                    (config->GetTime_Marching() == DT_STEPPING_2ND));

  /*--- If restart solution, then interpolate the flow solution to
   all the multigrid levels, this is important with the dual time strategy ---*/

  if (restart && (TimeIter == 0)) {

    Solution = new su2double[nVar];
    for (iMesh = 1; iMesh <= config->GetnMGLevels(); iMesh++) {
      for (iPoint = 0; iPoint < geometry[iMesh]->GetnPoint(); iPoint++) {
        Area_Parent = geometry[iMesh]->nodes->GetVolume(iPoint);
        for (iVar = 0; iVar < nVar; iVar++) Solution[iVar] = 0.0;
        for (iChildren = 0; iChildren < geometry[iMesh]->nodes->GetnChildren_CV(iPoint); iChildren++) {
          Point_Fine = geometry[iMesh]->nodes->GetChildren_CV(iPoint, iChildren);
          Area_Children = geometry[iMesh-1]->nodes->GetVolume(Point_Fine);
          Solution_Fine = solver_container[iMesh-1][HEAT_SOL]->GetNodes()->GetSolution(Point_Fine);
          for (iVar = 0; iVar < nVar; iVar++) {
            Solution[iVar] += Solution_Fine[iVar]*Area_Children/Area_Parent;
          }
        }
        solver_container[iMesh][HEAT_SOL]->GetNodes()->SetSolution(iPoint,Solution);
      }
      solver_container[iMesh][HEAT_SOL]->InitiateComms(geometry[iMesh], config, SOLUTION);
      solver_container[iMesh][HEAT_SOL]->CompleteComms(geometry[iMesh], config, SOLUTION);
    }
    delete [] Solution;
  }

  /*--- The value of the solution for the first iteration of the dual time ---*/

  if (dual_time && (TimeIter == 0 || (restart && (long)TimeIter == (long)config->GetRestart_Iter()))) {

    /*--- Push back the initial condition to previous solution containers
     for a 1st-order restart or when simply intitializing to freestream. ---*/

    for (iMesh = 0; iMesh <= config->GetnMGLevels(); iMesh++) {
      solver_container[iMesh][HEAT_SOL]->GetNodes()->Set_Solution_time_n();
      solver_container[iMesh][HEAT_SOL]->GetNodes()->Set_Solution_time_n1();
    }

    if ((restart && (long)TimeIter == (long)config->GetRestart_Iter()) &&
        (config->GetTime_Marching() == DT_STEPPING_2ND)) {

      /*--- Load an additional restart file for a 2nd-order restart ---*/

      solver_container[MESH_0][HEAT_SOL]->LoadRestart(geometry, solver_container, config, SU2_TYPE::Int(config->GetRestart_Iter()-1), true);

      /*--- Push back this new solution to time level N. ---*/

      for (iMesh = 0; iMesh <= config->GetnMGLevels(); iMesh++) {
        solver_container[iMesh][HEAT_SOL]->GetNodes()->Set_Solution_time_n();
      }
    }
  }
}

void CHeatSolver::SetResidual_DualTime(CGeometry *geometry, CSolver **solver_container, CConfig *config,
                                        unsigned short iRKStep, unsigned short iMesh, unsigned short RunTime_EqSystem) {

  /*--- Local variables ---*/

  unsigned short iVar, jVar;
  unsigned long iPoint;

  su2double *U_time_n, *U_time_nP1, *U_time_nM1;
  su2double Volume_nP1, TimeStep;

  bool implicit       = (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT);

  /*--- Store the physical time step ---*/

  TimeStep = config->GetDelta_UnstTimeND();

  /*--- Compute the dual time-stepping source term for static meshes ---*/

  if (!dynamic_grid) {

    /*--- Loop over all nodes (excluding halos) ---*/

    for (iPoint = 0; iPoint < nPointDomain; iPoint++) {

      /*--- Retrieve the solution at time levels n-1, n, and n+1. Note that
       we are currently iterating on U^n+1 and that U^n & U^n-1 are fixed,
       previous solutions that are stored in memory. ---*/

      U_time_nM1 = nodes->GetSolution_time_n1(iPoint);
      U_time_n   = nodes->GetSolution_time_n(iPoint);
      U_time_nP1 = nodes->GetSolution(iPoint);

      /*--- CV volume at time n+1. As we are on a static mesh, the volume
       of the CV will remained fixed for all time steps. ---*/

      Volume_nP1 = geometry->nodes->GetVolume(iPoint);

      /*--- Compute the dual time-stepping source term based on the chosen
       time discretization scheme (1st- or 2nd-order).---*/

      for (iVar = 0; iVar < nVar; iVar++) {
        if (config->GetTime_Marching() == DT_STEPPING_1ST)
          Residual[iVar] = (U_time_nP1[iVar] - U_time_n[iVar])*Volume_nP1 / TimeStep;
        if (config->GetTime_Marching() == DT_STEPPING_2ND)
          Residual[iVar] = ( 3.0*U_time_nP1[iVar] - 4.0*U_time_n[iVar]
                            +1.0*U_time_nM1[iVar])*Volume_nP1 / (2.0*TimeStep);
      }

      /*--- Store the residual and compute the Jacobian contribution due
       to the dual time source term. ---*/

      LinSysRes.AddBlock(iPoint, Residual);
      if (implicit) {
        for (iVar = 0; iVar < nVar; iVar++) {
          for (jVar = 0; jVar < nVar; jVar++) Jacobian_i[iVar][jVar] = 0.0;
          if (config->GetTime_Marching() == DT_STEPPING_1ST)
            Jacobian_i[iVar][iVar] = Volume_nP1 / TimeStep;
          if (config->GetTime_Marching() == DT_STEPPING_2ND)
            Jacobian_i[iVar][iVar] = (Volume_nP1*3.0)/(2.0*TimeStep);
        }

        Jacobian.AddBlock2Diag(iPoint, Jacobian_i);
      }
    }
  }
}
