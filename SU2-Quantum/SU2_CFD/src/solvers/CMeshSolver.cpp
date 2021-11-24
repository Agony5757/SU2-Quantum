/*!
 * \file CMeshSolver.cpp
 * \brief Main subroutines to solve moving meshes using a pseudo-linear elastic approach.
 * \author Ruben Sanchez
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

#include "../../../Common/include/adt_structure.hpp"
#include "../../../Common/include/omp_structure.hpp"
#include "../../include/solvers/CMeshSolver.hpp"
#include "../../include/variables/CMeshBoundVariable.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"

using namespace GeometryToolbox;


CMeshSolver::CMeshSolver(CGeometry *geometry, CConfig *config) : CFEASolver(true) {

  /*--- Initialize some booleans that determine the kind of problem at hand. ---*/

  time_domain = config->GetTime_Domain();
  multizone = config->GetMultizone_Problem();

  /*--- Determine if the stiffness per-element is set ---*/
  switch (config->GetDeform_Stiffness_Type()) {
  case INVERSE_VOLUME:
  case SOLID_WALL_DISTANCE:
    stiffness_set = false;
    break;
  case CONSTANT_STIFFNESS:
    stiffness_set = true;
    break;
  }

  /*--- Initialize the number of spatial dimensions, length of the state
   vector (same as spatial dimensions for grid deformation), and grid nodes. ---*/

  unsigned short iDim;
  unsigned long iPoint, iElem;

  nDim         = geometry->GetnDim();
  nVar         = geometry->GetnDim();
  nPoint       = geometry->GetnPoint();
  nPointDomain = geometry->GetnPointDomain();
  nElement     = geometry->GetnElem();

  MinVolume_Ref = 0.0;
  MinVolume_Curr = 0.0;

  MaxVolume_Ref = 0.0;
  MaxVolume_Curr = 0.0;

  /*--- Initialize the node structure ---*/

  nodes = new CMeshBoundVariable(nPoint, nDim, config);
  SetBaseClassPointerToNodes();

  /*--- Set which points are vertices and allocate boundary data. ---*/

  for (iPoint = 0; iPoint < nPoint; iPoint++) {

    for (iDim = 0; iDim < nDim; ++iDim)
      nodes->SetMesh_Coord(iPoint, iDim, geometry->nodes->GetCoord(iPoint, iDim));

    for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
      long iVertex = geometry->nodes->GetVertex(iPoint, iMarker);
      if (iVertex >= 0) {
        nodes->Set_isVertex(iPoint,true);
        break;
      }
    }
  }
  static_cast<CMeshBoundVariable*>(nodes)->AllocateBoundaryVariables(config);

  /*--- Initialize the element structure ---*/

  element.resize(nElement);

  /*--- Initialize matrix, solution, and r.h.s. structures for the linear solver. ---*/

  if (rank == MASTER_NODE) cout << "Initialize Jacobian structure (Mesh Deformation)." << endl;

  LinSysSol.Initialize(nPoint, nPointDomain, nVar, 0.0);
  LinSysRes.Initialize(nPoint, nPointDomain, nVar, 0.0);
  Jacobian.Initialize(nPoint, nPointDomain, nVar, nVar, false, geometry, config);
  System.SetToleranceType(LinearToleranceType::ABSOLUTE);

  /*--- Initialize structures for hybrid-parallel mode. ---*/

  HybridParallelInitialization(geometry);

  /*--- Element container structure. ---*/

  if (nDim == 2) {
    for(int thread = 0; thread < omp_get_max_threads(); ++thread) {

      const int offset = thread*MAX_FE_KINDS;
      element_container[FEA_TERM][EL_TRIA+offset] = new CTRIA1();
      element_container[FEA_TERM][EL_QUAD+offset] = new CQUAD4();
    }
  }
  else {
    for(int thread = 0; thread < omp_get_max_threads(); ++thread) {

      const int offset = thread*MAX_FE_KINDS;
      element_container[FEA_TERM][EL_TETRA+offset] = new CTETRA1();
      element_container[FEA_TERM][EL_HEXA +offset] = new CHEXA8 ();
      element_container[FEA_TERM][EL_PYRAM+offset] = new CPYRAM5();
      element_container[FEA_TERM][EL_PRISM+offset] = new CPRISM6();
    }
  }

  unsigned short iVar;

  /*--- Initialize the BGS residuals in multizone problems. ---*/
  if (config->GetMultizone_Residual()){

    Residual_BGS      = new su2double[nVar]();
    Residual_Max_BGS  = new su2double[nVar]();

    /*--- Define some structures for locating max residuals ---*/

    Point_Max_BGS       = new unsigned long[nVar]();
    Point_Max_Coord_BGS = new su2double*[nVar];
    for (iVar = 0; iVar < nVar; iVar++) {
      Point_Max_Coord_BGS[iVar] = new su2double[nDim]();
    }
  }


  /*--- Allocate element properties - only the index, to allow further integration with CFEASolver on a later stage ---*/
  element_properties = new CProperty*[nElement];
  for (iElem = 0; iElem < nElement; iElem++){
    element_properties[iElem] = new CProperty(iElem);
  }

  /*--- Compute the element volumes using the reference coordinates ---*/
  SU2_OMP_PARALLEL {
    SetMinMaxVolume(geometry, config, false);
  }

  /*--- Compute the wall distance using the reference coordinates ---*/
  SetWallDistance(geometry, config);

  if (size != SINGLE_NODE) {
    vector<unsigned short> essentialMarkers;
    /*--- Markers types covered in SetBoundaryDisplacements. ---*/
    for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
      if (((config->GetMarker_All_KindBC(iMarker) != SEND_RECEIVE) &&
           (config->GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY)) ||
           (config->GetMarker_All_Deform_Mesh(iMarker) == YES) ||
           (config->GetMarker_All_Moving(iMarker) == YES)) {
        essentialMarkers.push_back(iMarker);
      }
    }
    Set_VertexEliminationSchedule(geometry, essentialMarkers);
  }
}

void CMeshSolver::SetMinMaxVolume(CGeometry *geometry, CConfig *config, bool updated) {

  /*--- This routine is for post processing, it does not need to be recorded. ---*/
  const bool wasActive = AD::BeginPassive();

  /*--- Initialize shared reduction variables. ---*/
  SU2_OMP_BARRIER
  SU2_OMP_MASTER {
    MaxVolume = -1E22; MinVolume = 1E22;
    ElemCounter = 0;
  }

  /*--- Local min/max, final reduction outside loop. ---*/
  su2double maxVol = -1E22, minVol = 1E22;
  unsigned long elCount = 0;

  /*--- Loop over the elements in the domain. ---*/

  SU2_OMP_FOR_DYN(omp_chunk_size)
  for (unsigned long iElem = 0; iElem < nElement; iElem++) {

    int thread = omp_get_thread_num();

    int EL_KIND;
    unsigned short iNode, nNodes, iDim;

    GetElemKindAndNumNodes(geometry->elem[iElem]->GetVTK_Type(), EL_KIND, nNodes);

    CElement* fea_elem = element_container[FEA_TERM][EL_KIND + thread*MAX_FE_KINDS];

    /*--- For the number of nodes, we get the coordinates from
     *    the connectivity matrix and the geometry structure. ---*/

    for (iNode = 0; iNode < nNodes; iNode++) {

      auto indexNode = geometry->elem[iElem]->GetNode(iNode);

      /*--- Compute the volume with the reference or current coordinates. ---*/
      for (iDim = 0; iDim < nDim; iDim++) {
        su2double val_Coord = nodes->GetMesh_Coord(indexNode,iDim);
        if (updated)
          val_Coord += nodes->GetSolution(indexNode,iDim);

        fea_elem->SetRef_Coord(iNode, iDim, val_Coord);
      }
    }

    /*--- Compute the volume of the element (or the area in 2D cases ). ---*/

    su2double ElemVolume;
    if (nDim == 2) ElemVolume = fea_elem->ComputeArea();
    else           ElemVolume = fea_elem->ComputeVolume();

    maxVol = max(maxVol, ElemVolume);
    minVol = min(minVol, ElemVolume);

    if (updated) element[iElem].SetCurr_Volume(ElemVolume);
    else element[iElem].SetRef_Volume(ElemVolume);

    /*--- Count distorted elements. ---*/
    if (ElemVolume <= 0.0) elCount++;
  }
  SU2_OMP_CRITICAL
  {
    MaxVolume = max(MaxVolume, maxVol);
    MinVolume = min(MinVolume, minVol);
    ElemCounter += elCount;
  }
  SU2_OMP_BARRIER

  SU2_OMP_MASTER
  {
    elCount = ElemCounter; maxVol = MaxVolume; minVol = MinVolume;
    SU2_MPI::Allreduce(&elCount, &ElemCounter, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    SU2_MPI::Allreduce(&maxVol, &MaxVolume, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    SU2_MPI::Allreduce(&minVol, &MinVolume, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  }
  SU2_OMP_BARRIER

  /*--- Volume from 0 to 1 ---*/

  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iElem = 0; iElem < nElement; iElem++) {
    if (updated) {
      su2double ElemVolume = element[iElem].GetCurr_Volume()/MaxVolume;
      element[iElem].SetCurr_Volume(ElemVolume);
    }
    else {
      su2double ElemVolume = element[iElem].GetRef_Volume()/MaxVolume;
      element[iElem].SetRef_Volume(ElemVolume);
    }
  }

  /*--- Store the maximum and minimum volume. ---*/
  SU2_OMP_MASTER {
  if (updated) {
    MaxVolume_Curr = MaxVolume;
    MinVolume_Curr = MinVolume;
  }
  else {
    MaxVolume_Ref = MaxVolume;
    MinVolume_Ref = MinVolume;
  }

  if ((ElemCounter != 0) && (rank == MASTER_NODE))
    cout <<"There are " << ElemCounter << " elements with negative volume.\n" << endl;

  } SU2_OMP_BARRIER

  AD::EndPassive(wasActive);
}

void CMeshSolver::SetWallDistance(CGeometry *geometry, CConfig *config) {

  /*--- Initialize min and max distance ---*/

  MaxDistance = -1E22; MinDistance = 1E22;

  /*--- Compute the total number of nodes on no-slip boundaries ---*/

  unsigned long nVertex_SolidWall = 0;
  for(auto iMarker=0u; iMarker<config->GetnMarker_All(); ++iMarker) {
    if(config->GetSolid_Wall(iMarker)) {
      nVertex_SolidWall += geometry->GetnVertex(iMarker);
    }
  }

  /*--- Allocate the vectors to hold boundary node coordinates
   and its local ID. ---*/

  vector<su2double>     Coord_bound(nDim*nVertex_SolidWall);
  vector<unsigned long> PointIDs(nVertex_SolidWall);

  /*--- Retrieve and store the coordinates of the no-slip boundary nodes
   and their local point IDs. ---*/


  for (unsigned long iMarker=0, ii=0, jj=0; iMarker<config->GetnMarker_All(); ++iMarker) {

    if (!config->GetSolid_Wall(iMarker)) continue;

    for (auto iVertex=0u; iVertex<geometry->GetnVertex(iMarker); ++iVertex) {
      auto iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
      PointIDs[jj++] = iPoint;
      for (auto iDim=0u; iDim<nDim; ++iDim){
        Coord_bound[ii++] = nodes->GetMesh_Coord(iPoint,iDim);
      }
    }
  }

  /*--- Build the ADT of the boundary nodes. ---*/

  CADTPointsOnlyClass WallADT(nDim, nVertex_SolidWall, Coord_bound.data(),
                              PointIDs.data(), true);

  SU2_OMP_PARALLEL
  {
  /*--- Loop over all interior mesh nodes and compute the distances to each
   of the no-slip boundary nodes. Store the minimum distance to the wall
   for each interior mesh node. ---*/

  if( WallADT.IsEmpty() ) {

    /*--- No solid wall boundary nodes in the entire mesh. Set the
     wall distance to MaxDistance so we get stiffness of 1. ---*/

    SU2_OMP_FOR_STAT(omp_chunk_size)
    for (auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
      nodes->SetWallDistance(iPoint, MaxDistance);
    }
  }
  else {
    su2double MaxDistance_Local = -1E22, MinDistance_Local = 1E22;

    /*--- Solid wall boundary nodes are present. Compute the wall
     distance for all nodes. ---*/
    SU2_OMP_FOR_DYN(omp_chunk_size)
    for(auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
      su2double dist;
      unsigned long pointID;
      int rankID;
      WallADT.DetermineNearestNode(nodes->GetMesh_Coord(iPoint), dist,
                                   pointID, rankID);
      nodes->SetWallDistance(iPoint,dist);

      MaxDistance_Local = max(MaxDistance_Local, dist);

      /*--- To discard points on the surface we use > EPS ---*/

      if (dist > EPS)  MinDistance_Local = min(MinDistance_Local, dist);

    }
    SU2_OMP_CRITICAL
    {
      MaxDistance = max(MaxDistance, MaxDistance_Local);
      MinDistance = min(MinDistance, MinDistance_Local);
    }
    SU2_OMP_BARRIER

    SU2_OMP_MASTER
    {
      MaxDistance_Local = MaxDistance;
      MinDistance_Local = MinDistance;
      SU2_MPI::Allreduce(&MaxDistance_Local, &MaxDistance, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      SU2_MPI::Allreduce(&MinDistance_Local, &MinDistance, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    }
    SU2_OMP_BARRIER
  }

  /*--- Normalize distance from 0 to 1 ---*/
  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
    su2double nodeDist = nodes->GetWallDistance(iPoint)/MaxDistance;
    nodes->SetWallDistance(iPoint,nodeDist);
  }

  /*--- Compute the element distances ---*/
  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (auto iElem = 0ul; iElem < nElement; iElem++) {

    int EL_KIND;
    unsigned short nNodes = 0;
    GetElemKindAndNumNodes(geometry->elem[iElem]->GetVTK_Type(), EL_KIND, nNodes);

    /*--- Average the distance of the nodes in the element ---*/

    su2double ElemDist = 0.0;
    for (auto iNode = 0u; iNode < nNodes; iNode++) {
      auto iPoint = geometry->elem[iElem]->GetNode(iNode);
      ElemDist += nodes->GetWallDistance(iPoint);
    }
    ElemDist = ElemDist/su2double(nNodes);

    element[iElem].SetWallDistance(ElemDist);
  }

  } // end SU2_OMP_PARALLEL
}

void CMeshSolver::SetMesh_Stiffness(CGeometry **geometry, CNumerics **numerics, CConfig *config){

  if (stiffness_set) return;

  /*--- Use the config option as an upper bound on elasticity modulus.
   *    For RANS meshes the range of element volume or wall distance is
   *    very large and leads to an ill-conditioned stiffness matrix.
   *    Absolute values of elasticity modulus are not important for
   *    mesh deformation, since linear elasticity is used and all
   *    boundary conditions are essential (Dirichlet). ---*/
  const su2double maxE = config->GetDeform_ElasticityMod();

  /*--- All threads must execute the entire loop (no worksharing),
   *    each sets the stiffnesses for its numerics instance. ---*/
  SU2_OMP_PARALLEL
  {
  CNumerics* myNumerics = numerics[FEA_TERM + omp_get_thread_num()*MAX_TERMS];

  switch (config->GetDeform_Stiffness_Type()) {

    /*--- Stiffness inverse of the volume of the element. ---*/
    case INVERSE_VOLUME:
      for (unsigned long iElem = 0; iElem < nElement; iElem++) {
        su2double E = 1.0 / element[iElem].GetRef_Volume();
        myNumerics->SetMeshElasticProperties(iElem, min(E,maxE));
      }
    break;

    /*--- Stiffness inverse of the distance of the element to the closest wall. ---*/
    case SOLID_WALL_DISTANCE: {
      const su2double offset = config->GetDeform_StiffLayerSize();
      if (fabs(offset) > 0.0) {
        /*--- With prescribed layer of maximum stiffness (reaches max and holds). ---*/
        su2double d0 = offset / MaxDistance;
        su2double dmin = 1.0 / maxE;
        su2double scale = 1.0 / (1.0 - d0);
        for (unsigned long iElem = 0; iElem < nElement; iElem++) {
          su2double E = 1.0 / max(dmin, (element[iElem].GetWallDistance() - d0)*scale);
          myNumerics->SetMeshElasticProperties(iElem, E);
        }
      } else {
        /*--- Without prescribed layer of maximum stiffness (may not reach max). ---*/
        for (unsigned long iElem = 0; iElem < nElement; iElem++) {
          su2double E = 1.0 / element[iElem].GetWallDistance();
          myNumerics->SetMeshElasticProperties(iElem, min(E,maxE));
        }
      }
    }
    break;
  }
  }
  stiffness_set = true;

}

void CMeshSolver::DeformMesh(CGeometry **geometry, CNumerics **numerics, CConfig *config){

  if (multizone) nodes->Set_BGSSolution_k();

  /*--- Capture a few MPI dependencies for AD. ---*/
  geometry[MESH_0]->InitiateComms(geometry[MESH_0], config, COORDINATES);
  geometry[MESH_0]->CompleteComms(geometry[MESH_0], config, COORDINATES);

  InitiateComms(geometry[MESH_0], config, SOLUTION);
  CompleteComms(geometry[MESH_0], config, SOLUTION);

  InitiateComms(geometry[MESH_0], config, MESH_DISPLACEMENTS);
  CompleteComms(geometry[MESH_0], config, MESH_DISPLACEMENTS);

  /*--- Compute the stiffness matrix, no point recording because we clear the residual. ---*/

  const bool wasActive = AD::BeginPassive();

  Compute_StiffMatrix(geometry[MESH_0], numerics, config);

  AD::EndPassive(wasActive);

  /*--- Clear residual (loses AD info), we do not want an incremental solution. ---*/
  SU2_OMP_PARALLEL {
    LinSysRes.SetValZero();
  }

  /*--- Impose boundary conditions (all of them are ESSENTIAL BC's - displacements). ---*/
  SetBoundaryDisplacements(geometry[MESH_0], numerics[FEA_TERM], config);

  /*--- Solve the linear system. ---*/
  Solve_System(geometry[MESH_0], config);

  SU2_OMP_PARALLEL {

  /*--- Update the grid coordinates and cell volumes using the solution
     of the linear system (usol contains the x, y, z displacements). ---*/
  UpdateGridCoord(geometry[MESH_0], config);

  /*--- Update the dual grid. ---*/
  UpdateDualGrid(geometry[MESH_0], config);

  /*--- The Grid Velocity is only computed if the problem is time domain ---*/
  if (time_domain) ComputeGridVelocity(geometry[MESH_0], config);

  /*--- Update the multigrid structure. ---*/
  UpdateMultiGrid(geometry, config);

  /*--- Check for failed deformation (negative volumes). ---*/
  SetMinMaxVolume(geometry[MESH_0], config, true);

  } // end parallel

}

void CMeshSolver::UpdateGridCoord(CGeometry *geometry, CConfig *config){

  /*--- Update the grid coordinates using the solution of the linear system ---*/

  /*--- LinSysSol contains the absolute x, y, z displacements. ---*/
  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++){
    for (unsigned short iDim = 0; iDim < nDim; iDim++) {
      /*--- Retrieve the displacement from the solution of the linear system ---*/
      su2double val_disp = LinSysSol(iPoint, iDim);
      /*--- Store the displacement of the mesh node ---*/
      nodes->SetSolution(iPoint, iDim, val_disp);
      /*--- Compute the current coordinate as Mesh_Coord + Displacement ---*/
      su2double val_coord = nodes->GetMesh_Coord(iPoint,iDim) + val_disp;
      /*--- Update the geometry container ---*/
      geometry->nodes->SetCoord(iPoint, iDim, val_coord);
    }
  }

  /*--- Communicate the updated displacements and mesh coordinates. ---*/
  geometry->InitiateComms(geometry, config, COORDINATES);
  geometry->CompleteComms(geometry, config, COORDINATES);

  InitiateComms(geometry, config, SOLUTION);
  CompleteComms(geometry, config, SOLUTION);

}

void CMeshSolver::UpdateDualGrid(CGeometry *geometry, CConfig *config){

  /*--- After moving all nodes, update the dual mesh. Recompute the edges and
   dual mesh control volumes in the domain and on the boundaries. ---*/

  geometry->SetCoord_CG();
  geometry->SetControlVolume(config, UPDATE);
  geometry->SetBoundControlVolume(config, UPDATE);
  geometry->SetMaxLength(config);

}

void CMeshSolver::ComputeGridVelocity(CGeometry *geometry, CConfig *config){

  /*--- Compute the velocity of each node in the domain of the current rank
   (halo nodes are not computed as the grid velocity is later communicated). ---*/

  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++) {

    /*--- Coordinates of the current point at n+1, n, & n-1 time levels. ---*/

    const su2double* Disp_nM1 = nodes->GetSolution_time_n1(iPoint);
    const su2double* Disp_n   = nodes->GetSolution_time_n(iPoint);
    const su2double* Disp_nP1 = nodes->GetSolution(iPoint);

    /*--- Unsteady time step ---*/

    su2double TimeStep = config->GetDelta_UnstTimeND();

    /*--- Compute mesh velocity with 1st or 2nd-order approximation. ---*/

    for (unsigned short iDim = 0; iDim < nDim; iDim++) {

      su2double GridVel = 0.0;

      if (config->GetTime_Marching() == DT_STEPPING_1ST)
        GridVel = ( Disp_nP1[iDim] - Disp_n[iDim] ) / TimeStep;
      if (config->GetTime_Marching() == DT_STEPPING_2ND)
        GridVel = ( 3.0*Disp_nP1[iDim] - 4.0*Disp_n[iDim] +
                    1.0*Disp_nM1[iDim] ) / (2.0*TimeStep);

      /*--- Store grid velocity for this point ---*/

      geometry->nodes->SetGridVel(iPoint, iDim, GridVel);

    }
  }

  /*--- The velocity was computed for nPointDomain, now we communicate it. ---*/
  geometry->InitiateComms(geometry, config, GRID_VELOCITY);
  geometry->CompleteComms(geometry, config, GRID_VELOCITY);

}

void CMeshSolver::UpdateMultiGrid(CGeometry **geometry, CConfig *config) const{

  /*--- Update the multigrid structure after moving the finest grid,
   including computing the grid velocities on the coarser levels
   when the problem is solved in unsteady conditions. ---*/

  for (auto iMGlevel = 1u; iMGlevel <= config->GetnMGLevels(); iMGlevel++) {
    const auto iMGfine = iMGlevel-1;
    geometry[iMGlevel]->SetControlVolume(config, geometry[iMGfine], UPDATE);
    geometry[iMGlevel]->SetBoundControlVolume(config, geometry[iMGfine],UPDATE);
    geometry[iMGlevel]->SetCoord(geometry[iMGfine]);
    if (time_domain)
      geometry[iMGlevel]->SetRestricted_GridVelocity(geometry[iMGfine], config);
  }

}

void CMeshSolver::SetBoundaryDisplacements(CGeometry *geometry, CNumerics *numerics, CConfig *config){

  /* Surface motions are not applied during discrete adjoint runs as the corresponding
   * boundary displacements are computed when loading the primal solution, and it
   * would be complex to account for the incremental nature of these motions.
   * The derivatives are still correct since the motion does not depend on the solution,
   * but this means that (for now) we cannot get derivatives w.r.t. motion parameters. */

  if (config->GetSurface_Movement(DEFORMING) && !config->GetDiscrete_Adjoint()) {
    if (rank == MASTER_NODE)
      cout << endl << " Updating surface positions." << endl;

    Surface_Translating(geometry, config, config->GetTimeIter());
    Surface_Plunging(geometry, config, config->GetTimeIter());
    Surface_Pitching(geometry, config, config->GetTimeIter());
    Surface_Rotating(geometry, config, config->GetTimeIter());
  }

  unsigned short iMarker;

  /*--- Impose zero displacements of all non-moving surfaces (also at nodes in multiple moving/non-moving boundaries). ---*/
  /*--- Exceptions: symmetry plane, the receive boundaries and periodic boundaries should get a different treatment. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == NO) &&
        (config->GetMarker_All_Moving(iMarker) == NO) &&
        (config->GetMarker_All_KindBC(iMarker) != SYMMETRY_PLANE) &&
        (config->GetMarker_All_KindBC(iMarker) != SEND_RECEIVE) &&
        (config->GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY)) {

      BC_Clamped(geometry, numerics, config, iMarker);
    }
  }

  /*--- Symmetry plane is clamped, for now. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == NO) &&
        (config->GetMarker_All_Moving(iMarker) == NO) &&
        (config->GetMarker_All_KindBC(iMarker) == SYMMETRY_PLANE)) {

      BC_Clamped(geometry, numerics, config, iMarker);
    }
  }

  /*--- Impose displacement boundary conditions. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == YES) ||
        (config->GetMarker_All_Moving(iMarker) == YES)) {

      BC_Deforming(geometry, numerics, config, iMarker);
    }
  }

  /*--- Clamp far away nodes according to deform limit. ---*/
  if ((config->GetDeform_Stiffness_Type() == SOLID_WALL_DISTANCE) &&
      (config->GetDeform_Limit() < MaxDistance)) {

    const su2double limit = config->GetDeform_Limit() / MaxDistance;

    for (auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
      if (nodes->GetWallDistance(iPoint) <= limit) continue;

      su2double zeros[MAXNVAR] = {0.0};
      nodes->SetSolution(iPoint, zeros);
      LinSysSol.SetBlock(iPoint, zeros);
      Jacobian.EnforceSolutionAtNode(iPoint, zeros, LinSysRes);
    }
  }

}

void CMeshSolver::SetDualTime_Mesh(void){

  nodes->Set_Solution_time_n1();
  nodes->Set_Solution_time_n();
}

void CMeshSolver::LoadRestart(CGeometry **geometry, CSolver ***solver, CConfig *config, int val_iter, bool val_update_geo) {

  /*--- Read the restart data from either an ASCII or binary SU2 file. ---*/

  string filename = config->GetFilename(config->GetSolution_FileName(), "", val_iter);

  if (config->GetRead_Binary_Restart()) {
    Read_SU2_Restart_Binary(geometry[MESH_0], config, filename);
  } else {
    Read_SU2_Restart_ASCII(geometry[MESH_0], config, filename);
  }

  /*--- Load data from the restart into correct containers. ---*/

  unsigned long iPoint_Global, counter = 0;

  for (iPoint_Global = 0; iPoint_Global < geometry[MESH_0]->GetGlobal_nPointDomain(); iPoint_Global++) {

    /*--- Retrieve local index. If this node from the restart file lives
     on the current processor, we will load and instantiate the vars. ---*/

    auto iPoint_Local = geometry[MESH_0]->GetGlobal_to_Local_Point(iPoint_Global);

    if (iPoint_Local >= 0) {

      /*--- We need to store this point's data, so jump to the correct
       offset in the buffer of data from the restart file and load it. ---*/

      auto index = counter*Restart_Vars[1];

      for (unsigned short iDim = 0; iDim < nDim; iDim++){
        /*--- Update the coordinates of the mesh ---*/
        su2double curr_coord = Restart_Data[index+iDim];
        /// TODO: "Double deformation" in multizone adjoint if this is set here?
        ///       In any case it should not be needed as deformation is called before other solvers
        ///geometry[MESH_0]->nodes->SetCoord(iPoint_Local, iDim, curr_coord);

        /*--- Store the displacements computed as the current coordinates
         minus the coordinates of the reference mesh file ---*/
        su2double displ = curr_coord - nodes->GetMesh_Coord(iPoint_Local, iDim);
        nodes->SetSolution(iPoint_Local, iDim, displ);
      }

      /*--- Increment the overall counter for how many points have been loaded. ---*/
      counter++;
    }

  }

  /*--- Detect a wrong solution file ---*/

  if (counter != nPointDomain) {
    SU2_MPI::Error(string("The solution file ") + filename + string(" doesn't match with the mesh file!\n") +
                   string("It could be empty lines at the end of the file."), CURRENT_FUNCTION);
  }

  /*--- Communicate the loaded displacements. ---*/
  solver[MESH_0][MESH_SOL]->InitiateComms(geometry[MESH_0], config, SOLUTION);
  solver[MESH_0][MESH_SOL]->CompleteComms(geometry[MESH_0], config, SOLUTION);

  /*--- Communicate the new coordinates at the halos ---*/
  geometry[MESH_0]->InitiateComms(geometry[MESH_0], config, COORDINATES);
  geometry[MESH_0]->CompleteComms(geometry[MESH_0], config, COORDINATES);

  /*--- Init the linear system solution. ---*/
  for (unsigned long iPoint = 0; iPoint < nPoint; ++iPoint) {
    for (unsigned short iDim = 0; iDim < nDim; ++iDim) {
      LinSysSol(iPoint, iDim) = nodes->GetSolution(iPoint, iDim);
    }
  }

  /*--- Recompute the edges and dual mesh control volumes in the
   domain and on the boundaries. ---*/
  UpdateDualGrid(geometry[MESH_0], config);

  /*--- For time-domain problems, we need to compute the grid velocities ---*/
  if (time_domain){
    /*--- Update the old geometry (coordinates n and n-1) ---*/
    Restart_OldGeometry(geometry[MESH_0], config);
    /*--- Once Displacement_n and Displacement_n1 are filled,
     we can compute the Grid Velocity ---*/
    ComputeGridVelocity(geometry[MESH_0], config);
  }

  /*--- Update the multigrid structure after setting up the finest grid,
   including computing the grid velocities on the coarser levels
   when the problem is unsteady. ---*/
  UpdateMultiGrid(geometry, config);

  /*--- Store the boundary displacements at the Bound_Disp variable. ---*/

  for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {

    if ((config->GetMarker_All_Deform_Mesh(iMarker) == YES) ||
        (config->GetMarker_All_Moving(iMarker) == YES)) {

      for (unsigned long iVertex = 0; iVertex < geometry[MESH_0]->nVertex[iMarker]; iVertex++) {

        /*--- Get node index. ---*/
        auto iNode = geometry[MESH_0]->vertex[iMarker][iVertex]->GetNode();

        /*--- Set boundary solution. ---*/
        nodes->SetBound_Disp(iNode, nodes->GetSolution(iNode));
      }
    }
  }

  /*--- Delete the class memory that is used to load the restart. ---*/

  delete [] Restart_Vars; Restart_Vars = nullptr;
  delete [] Restart_Data; Restart_Data = nullptr;

}

void CMeshSolver::Restart_OldGeometry(CGeometry *geometry, CConfig *config) {

  /*--- This function is intended for dual time simulations ---*/

  unsigned short iZone = config->GetiZone();
  unsigned short nZone = geometry->GetnZone();
  string filename = config->GetSolution_FileName();

  /*--- Multizone problems require the number of the zone to be appended. ---*/

  if (nZone > 1)
    filename = config->GetMultizone_FileName(filename, iZone, "");

  /*--- Determine how many files need to be read. ---*/

  unsigned short nSteps = (config->GetTime_Marching() == DT_STEPPING_2ND) ? 2 : 1;

  for(unsigned short iStep = 1; iStep <= nSteps; ++iStep) {

    unsigned short CommType = (iStep == 1) ? SOLUTION_TIME_N : SOLUTION_TIME_N1;

    /*--- Modify file name for an unsteady restart ---*/
    int Unst_RestartIter;
    if (config->GetRestart()) Unst_RestartIter = SU2_TYPE::Int(config->GetRestart_Iter()) - iStep;
    else Unst_RestartIter = SU2_TYPE::Int(config->GetUnst_AdjointIter()) - SU2_TYPE::Int(config->GetTimeIter())-iStep-1;

    if (Unst_RestartIter < 0) {

      if (rank == MASTER_NODE) cout << "Requested mesh restart filename is negative. Setting known solution" << endl;

      /*--- Set loaded solution into correct previous time containers. ---*/
      if(iStep==1) nodes->Set_Solution_time_n();
      else nodes->Set_Solution_time_n1();
    }
    else {
      string filename_n = config->GetUnsteady_FileName(filename, Unst_RestartIter, "");

      /*--- Read the restart data from either an ASCII or binary SU2 file. ---*/

      if (config->GetRead_Binary_Restart()) {
        Read_SU2_Restart_Binary(geometry, config, filename_n);
      } else {
        Read_SU2_Restart_ASCII(geometry, config, filename_n);
      }

      /*--- Load data from the restart into correct containers. ---*/

      unsigned long iPoint_Global, counter = 0;

      for (iPoint_Global = 0; iPoint_Global < geometry->GetGlobal_nPointDomain(); iPoint_Global++) {

        /*--- Retrieve local index. If this node from the restart file lives
         on the current processor, we will load and instantiate the vars. ---*/

        auto iPoint_Local = geometry->GetGlobal_to_Local_Point(iPoint_Global);

        if (iPoint_Local >= 0) {

          /*--- We need to store this point's data, so jump to the correct
           offset in the buffer of data from the restart file and load it. ---*/

          auto index = counter*Restart_Vars[1];

          for (unsigned short iDim = 0; iDim < nDim; iDim++) {
            su2double curr_coord = Restart_Data[index+iDim];
            su2double displ = curr_coord - nodes->GetMesh_Coord(iPoint_Local,iDim);

            if(iStep==1)
              nodes->Set_Solution_time_n(iPoint_Local, iDim, displ);
            else
              nodes->Set_Solution_time_n1(iPoint_Local, iDim, displ);
          }

          /*--- Increment the overall counter for how many points have been loaded. ---*/
          counter++;
        }
      }


      /*--- Detect a wrong solution file. ---*/

      if (counter != nPointDomain) {
        SU2_MPI::Error(string("The solution file ") + filename_n + string(" doesn't match with the mesh file!\n") +
                       string("It could be empty lines at the end of the file."), CURRENT_FUNCTION);
      }
    }

    /*--- Delete the class memory that is used to load the restart. ---*/

    delete [] Restart_Vars; Restart_Vars = nullptr;
    delete [] Restart_Data; Restart_Data = nullptr;

    InitiateComms(geometry, config, CommType);
    CompleteComms(geometry, config, CommType);

  } // iStep

}

void CMeshSolver::Surface_Pitching(CGeometry *geometry, CConfig *config, unsigned long iter) {

  su2double deltaT, time_new, time_old, Lref;
  const su2double* Coord = nullptr;
  su2double Center[3] = {0.0}, VarCoord[3] = {0.0}, Omega[3] = {0.0}, Ampl[3] = {0.0}, Phase[3] = {0.0};
  su2double VarCoordAbs[3] = {0.0};
  su2double rotCoord[3] = {0.0}, r[3] = {0.0};
  su2double rotMatrix[3][3] = {{0.0}};
  su2double dtheta, dphi, dpsi;
  const su2double DEG2RAD = PI_NUMBER/180.0;
  unsigned short iMarker, jMarker, iDim;
  unsigned long iPoint, iVertex;
  string Marker_Tag, Moving_Tag;

  /*--- Retrieve values from the config file ---*/

  deltaT = config->GetDelta_UnstTimeND();
  Lref   = config->GetLength_Ref();

  /*--- Compute delta time based on physical time step ---*/

  time_new = iter*deltaT;
  if (iter == 0) time_old = time_new;
  else time_old = (iter-1)*deltaT;

  /*--- Store displacement of each node on the pitching surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to pitch ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

      /*--- Pitching origin, frequency, and amplitude from config. ---*/

      for (iDim = 0; iDim < 3; iDim++){
        Ampl[iDim]   = config->GetMarkerPitching_Ampl(jMarker, iDim)*DEG2RAD;
        Omega[iDim]  = config->GetMarkerPitching_Omega(jMarker, iDim)/config->GetOmega_Ref();
        Phase[iDim]  = config->GetMarkerPitching_Phase(jMarker, iDim)*DEG2RAD;
        Center[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);
      }
      /*--- Print some information to the console. Be verbose at the first
       iteration only (mostly for debugging purposes). ---*/
      // Note that the MASTER_NODE might not contain all the markers being moved.

      if (rank == MASTER_NODE) {
        cout << " Storing pitching displacement for marker: ";
        cout << Marker_Tag << "." << endl;
        if (iter == 0) {
          cout << " Pitching frequency: (" << Omega[0] << ", " << Omega[1];
          cout << ", " << Omega[2] << ") rad/s about origin: (" << Center[0];
          cout << ", " << Center[1] << ", " << Center[2] << ")." << endl;
          cout << " Pitching amplitude about origin: (" << Ampl[0]/DEG2RAD;
          cout << ", " << Ampl[1]/DEG2RAD << ", " << Ampl[2]/DEG2RAD;
          cout << ") degrees."<< endl;
          cout << " Pitching phase lag about origin: (" << Phase[0]/DEG2RAD;
          cout << ", " << Phase[1]/DEG2RAD <<", "<< Phase[2]/DEG2RAD;
          cout << ") degrees."<< endl;
        }
      }

      /*--- Compute delta change in the angle about the x, y, & z axes. ---*/

      dtheta = -Ampl[0]*(sin(Omega[0]*time_new + Phase[0])
                       - sin(Omega[0]*time_old + Phase[0]));
      dphi   = -Ampl[1]*(sin(Omega[1]*time_new + Phase[1])
                       - sin(Omega[1]*time_old + Phase[1]));
      dpsi   = -Ampl[2]*(sin(Omega[2]*time_new + Phase[2])
                       - sin(Omega[2]*time_old + Phase[2]));

      /*--- Compute rotation matrix. ---*/

      RotationMatrix(dtheta, dphi, dpsi, rotMatrix);

      /*--- Apply rotation to the vertices. ---*/

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Index and coordinates of the current point ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
        Coord  = geometry->nodes->GetCoord(iPoint);

        /*--- Calculate non-dim. position from rotation center ---*/

        for (iDim = 0; iDim < nDim; iDim++)
          r[iDim] = (Coord[iDim]-Center[iDim])/Lref;

        /*--- Compute transformed point coordinates ---*/

        Rotate(rotMatrix, Center, r, rotCoord);

        /*--- Calculate delta change in the x, y, & z directions ---*/
        for (iDim = 0; iDim < nDim; iDim++)
          VarCoord[iDim] = (rotCoord[iDim]-Coord[iDim])/Lref;

        /*--- Set node displacement for volume deformation ---*/

        for (iDim = 0; iDim < nDim; iDim++)
          VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + VarCoord[iDim];

        nodes->SetBound_Disp(iPoint, VarCoordAbs);
      }
    }
  }
  /*--- For pitching we don't update the motion origin and moment reference origin. ---*/

}

void CMeshSolver::Surface_Rotating(CGeometry *geometry, CConfig *config, unsigned long iter) {

  su2double deltaT, time_new, time_old, Lref;
  const su2double* Coord = nullptr;
  su2double VarCoordAbs[3] = {0.0};
  su2double Center[3] = {0.0}, VarCoord[3] = {0.0}, Omega[3] = {0.0},
  rotCoord[3] = {0.0}, r[3] = {0.0}, Center_Aux[3] = {0.0};
  su2double rotMatrix[3][3] = {{0.0}};
  su2double dtheta, dphi, dpsi;
  unsigned short iMarker, jMarker, iDim;
  unsigned long iPoint, iVertex;
  string Marker_Tag, Moving_Tag;

  /*--- Retrieve values from the config file ---*/

  deltaT = config->GetDelta_UnstTimeND();
  Lref   = config->GetLength_Ref();

  /*--- Compute delta time based on physical time step ---*/

  time_new = iter*deltaT;
  if (iter == 0) time_old = time_new;
  else time_old = (iter-1)*deltaT;

  /*--- Store displacement of each node on the rotating surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to rotate ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

      /*--- Rotation origin and angular velocity from config. ---*/

      for (iDim = 0; iDim < 3; iDim++){
        Omega[iDim]  = config->GetMarkerRotationRate(jMarker, iDim)/config->GetOmega_Ref();
        Center[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);
      }

      /*--- Print some information to the console. Be verbose at the first
       iteration only (mostly for debugging purposes). ---*/
      // Note that the MASTER_NODE might not contain all the markers being moved.

      if (rank == MASTER_NODE) {
        cout << " Storing rotating displacement for marker: ";
        cout << Marker_Tag << "." << endl;
        if (iter == 0) {
          cout << " Angular velocity: (" << Omega[0] << ", " << Omega[1];
          cout << ", " << Omega[2] << ") rad/s about origin: (" << Center[0];
          cout << ", " << Center[1] << ", " << Center[2] << ")." << endl;
        }
      }

      /*--- Compute delta change in the angle about the x, y, & z axes. ---*/

      dtheta = Omega[0]*(time_new-time_old);
      dphi   = Omega[1]*(time_new-time_old);
      dpsi   = Omega[2]*(time_new-time_old);

      /*--- Compute rotation matrix. ---*/

      RotationMatrix(dtheta, dphi, dpsi, rotMatrix);

      /*--- Apply rotation to the vertices. ---*/

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Index and coordinates of the current point ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
        Coord  = geometry->nodes->GetCoord(iPoint);

        /*--- Calculate non-dim. position from rotation center ---*/

        for (iDim = 0; iDim < nDim; iDim++)
          r[iDim] = (Coord[iDim]-Center[iDim])/Lref;

        /*--- Compute transformed point coordinates ---*/

        Rotate(rotMatrix, Center, r, rotCoord);

        /*--- Calculate delta change in the x, y, & z directions ---*/
        for (iDim = 0; iDim < nDim; iDim++)
          VarCoord[iDim] = (rotCoord[iDim]-Coord[iDim])/Lref;

        /*--- Set node displacement for volume deformation ---*/
        for (iDim = 0; iDim < nDim; iDim++)
          VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + VarCoord[iDim];

        nodes->SetBound_Disp(iPoint, VarCoordAbs);
      }
    }
  }

  /*--- When updating the origins it is assumed that all markers have the
   same rotation movement, because we use the last markers rotation matrix and center ---*/

  /*--- Set the mesh motion center to the new location after
   incrementing the position with the rotation. This new
   location will be used for subsequent mesh motion for the given marker.---*/

  for (jMarker=0; jMarker < config->GetnMarker_Moving(); jMarker++) {

    /*-- Check if we want to update the motion origin for the given marker ---*/

    if (config->GetMoveMotion_Origin(jMarker) != YES) continue;

    for (iDim = 0; iDim < 3; iDim++)
      Center_Aux[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);

    /*--- Calculate non-dim. position from rotation center ---*/

    for (iDim = 0; iDim < nDim; iDim++)
      r[iDim] = (Center_Aux[iDim]-Center[iDim])/Lref;

    /*--- Compute transformed point coordinates ---*/

    Rotate(rotMatrix, Center, r, rotCoord);

    /*--- Calculate delta change in the x, y, & z directions ---*/
    for (iDim = 0; iDim < nDim; iDim++)
      VarCoord[iDim] = (rotCoord[iDim]-Center_Aux[iDim])/Lref;

    for (iDim = 0; iDim < 3; iDim++)
      Center_Aux[iDim] += VarCoord[iDim];

    config->SetMarkerMotion_Origin(Center_Aux, jMarker);
  }

  /*--- Set the moment computation center to the new location after
   incrementing the position with the rotation. ---*/

  for (jMarker=0; jMarker<config->GetnMarker_Monitoring(); jMarker++) {

    Center_Aux[0] = config->GetRefOriginMoment_X(jMarker);
    Center_Aux[1] = config->GetRefOriginMoment_Y(jMarker);
    Center_Aux[2] = config->GetRefOriginMoment_Z(jMarker);

    /*--- Calculate non-dim. position from rotation center ---*/

    for (iDim = 0; iDim < nDim; iDim++)
      r[iDim] = (Center_Aux[iDim]-Center[iDim])/Lref;

    /*--- Compute transformed point coordinates ---*/

    Rotate(rotMatrix, Center, r, rotCoord);

    /*--- Calculate delta change in the x, y, & z directions ---*/
    for (iDim = 0; iDim < nDim; iDim++)
      VarCoord[iDim] = (rotCoord[iDim]-Center_Aux[iDim])/Lref;

    config->SetRefOriginMoment_X(jMarker, Center_Aux[0]+VarCoord[0]);
    config->SetRefOriginMoment_Y(jMarker, Center_Aux[1]+VarCoord[1]);
    config->SetRefOriginMoment_Z(jMarker, Center_Aux[2]+VarCoord[2]);
  }
}

void CMeshSolver::Surface_Plunging(CGeometry *geometry, CConfig *config, unsigned long iter) {

  su2double deltaT, time_new, time_old, Lref;
  su2double Center[3] = {0.0}, VarCoord[3] = {0.0}, Omega[3] = {0.0}, Ampl[3] = {0.0};
  su2double VarCoordAbs[3] = {0.0};
  const su2double DEG2RAD = PI_NUMBER/180.0;
  unsigned short iMarker, jMarker;
  unsigned long iPoint, iVertex;
  string Marker_Tag, Moving_Tag;
  unsigned short iDim;

  /*--- Retrieve values from the config file ---*/

  deltaT = config->GetDelta_UnstTimeND();
  Lref   = config->GetLength_Ref();

  /*--- Compute delta time based on physical time step ---*/

  time_new = iter*deltaT;
  if (iter == 0) time_old = time_new;
  else time_old = (iter-1)*deltaT;

  /*--- Store displacement of each node on the plunging surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to plunge ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

      /*--- Plunging frequency and amplitude from config. ---*/

      for (iDim = 0; iDim < 3; iDim++){
        Ampl[iDim]   = config->GetMarkerPlunging_Ampl(jMarker, iDim)/Lref;
        Omega[iDim]  = config->GetMarkerPlunging_Omega(jMarker, iDim)/config->GetOmega_Ref();
        Center[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);
      }

      /*--- Print some information to the console. Be verbose at the first
       iteration only (mostly for debugging purposes). ---*/
      // Note that the MASTER_NODE might not contain all the markers being moved.

      if (rank == MASTER_NODE) {
        cout << " Storing plunging displacement for marker: ";
        cout << Marker_Tag << "." << endl;
        if (iter == 0) {
          cout << " Plunging frequency: (" << Omega[0] << ", " << Omega[1];
          cout << ", " << Omega[2] << ") rad/s." << endl;
          cout << " Plunging amplitude: (" << Ampl[0]/DEG2RAD;
          cout << ", " << Ampl[1]/DEG2RAD << ", " << Ampl[2]/DEG2RAD;
          cout << ") degrees."<< endl;
        }
      }

      /*--- Compute delta change in the position in the x, y, & z directions. ---*/

      VarCoord[0] = -Ampl[0]*(sin(Omega[0]*time_new) - sin(Omega[0]*time_old));
      VarCoord[1] = -Ampl[1]*(sin(Omega[1]*time_new) - sin(Omega[1]*time_old));
      VarCoord[2] = -Ampl[2]*(sin(Omega[2]*time_new) - sin(Omega[2]*time_old));

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Set node displacement for volume deformation ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

        for (iDim = 0; iDim < nDim; iDim++)
          VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + VarCoord[iDim];

        nodes->SetBound_Disp(iPoint, VarCoordAbs);

      }
    }
  }

  /*--- When updating the origins it is assumed that all markers have the
   same plunging movement, because we use the last VarCoord set ---*/

  /*--- Set the mesh motion center to the new location after
   incrementing the position with the translation. This new
   location will be used for subsequent mesh motion for the given marker.---*/

  for (jMarker=0; jMarker<config->GetnMarker_Moving(); jMarker++) {

    /*-- Check if we want to update the motion origin for the given marker ---*/

    if (config->GetMoveMotion_Origin(jMarker) == YES) {
      for (iDim = 0; iDim < 3; iDim++)
        Center[iDim] += VarCoord[iDim];

      config->SetMarkerMotion_Origin(Center, jMarker);
    }
  }

  /*--- Set the moment computation center to the new location after
   incrementing the position with the plunging. ---*/

  for (jMarker=0; jMarker < config->GetnMarker_Monitoring(); jMarker++) {
    Center[0] = config->GetRefOriginMoment_X(jMarker) + VarCoord[0];
    Center[1] = config->GetRefOriginMoment_Y(jMarker) + VarCoord[1];
    Center[2] = config->GetRefOriginMoment_Z(jMarker) + VarCoord[2];
    config->SetRefOriginMoment_X(jMarker, Center[0]);
    config->SetRefOriginMoment_Y(jMarker, Center[1]);
    config->SetRefOriginMoment_Z(jMarker, Center[2]);
  }
}

void CMeshSolver::Surface_Translating(CGeometry *geometry, CConfig *config, unsigned long iter) {

  su2double deltaT, time_new, time_old;
  su2double Center[3] = {0.0}, VarCoord[3] = {0.0};
  su2double VarCoordAbs[3] = {0.0};
  su2double xDot[3] = {0.0};
  unsigned short iMarker, jMarker;
  unsigned long iPoint, iVertex;
  string Marker_Tag, Moving_Tag;
  unsigned short iDim;

  /*--- Retrieve values from the config file ---*/

  deltaT = config->GetDelta_UnstTimeND();

  /*--- Compute delta time based on physical time step ---*/

  time_new = iter*deltaT;
  if (iter == 0) time_old = time_new;
  else time_old = (iter-1)*deltaT;

  /*--- Store displacement of each node on the translating surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to translate ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

      for (iDim = 0; iDim < 3; iDim++) {
        xDot[iDim]   = config->GetMarkerTranslationRate(jMarker, iDim);
        Center[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);
      }

      /*--- Print some information to the console. Be verbose at the first
       iteration only (mostly for debugging purposes). ---*/
      // Note that the MASTER_NODE might not contain all the markers being moved.

      if (rank == MASTER_NODE) {
        cout << " Storing translating displacement for marker: ";
        cout << Marker_Tag << "." << endl;
        if (iter == 0) {
          cout << " Translational velocity: (" << xDot[0]*config->GetVelocity_Ref() << ", " << xDot[1]*config->GetVelocity_Ref();
          cout << ", " << xDot[2]*config->GetVelocity_Ref();
          if (config->GetSystemMeasurements() == SI) cout << ") m/s." << endl;
          else cout << ") ft/s." << endl;
        }
      }

      /*--- Compute delta change in the position in the x, y, & z directions. ---*/

      VarCoord[0] = xDot[0]*(time_new-time_old);
      VarCoord[1] = xDot[1]*(time_new-time_old);
      VarCoord[2] = xDot[2]*(time_new-time_old);

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Set node displacement for volume deformation ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

        for (iDim = 0; iDim < nDim; iDim++)
          VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + VarCoord[iDim];

        nodes->SetBound_Disp(iPoint, VarCoordAbs);
      }
    }
  }

  /*--- When updating the origins it is assumed that all markers have the
        same translational velocity, because we use the last VarCoord set ---*/

  /*--- Set the mesh motion center to the new location after
   incrementing the position with the translation. This new
   location will be used for subsequent mesh motion for the given marker.---*/

  for (jMarker=0; jMarker < config->GetnMarker_Moving(); jMarker++) {

    /*-- Check if we want to update the motion origin for the given marker ---*/

    if (config->GetMoveMotion_Origin(jMarker) == YES) {
      for (iDim = 0; iDim < 3; iDim++)
        Center[iDim] += VarCoord[iDim];

      config->SetMarkerMotion_Origin(Center, jMarker);
    }
  }

  /*--- Set the moment computation center to the new location after
   incrementing the position with the translation. ---*/

  for (jMarker=0; jMarker < config->GetnMarker_Monitoring(); jMarker++) {
    Center[0] = config->GetRefOriginMoment_X(jMarker) + VarCoord[0];
    Center[1] = config->GetRefOriginMoment_Y(jMarker) + VarCoord[1];
    Center[2] = config->GetRefOriginMoment_Z(jMarker) + VarCoord[2];
    config->SetRefOriginMoment_X(jMarker, Center[0]);
    config->SetRefOriginMoment_Y(jMarker, Center[1]);
    config->SetRefOriginMoment_Z(jMarker, Center[2]);
  }
}
