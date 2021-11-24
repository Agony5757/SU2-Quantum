/*!
 * \file CMultiGridGeometry.cpp
 * \brief Implementation of the multigrid geometry class.
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

#include "../../include/geometry/CMultiGridGeometry.hpp"
#include "../../include/CMultiGridQueue.hpp"
#include "../../include/toolboxes/printing_toolbox.hpp"


CMultiGridGeometry::CMultiGridGeometry(CGeometry **geometry, CConfig *config_container, unsigned short iMesh) : CGeometry() {

  /*--- CGeometry & CConfig pointers to the fine grid level for clarity. We may
   need access to the other zones in the mesh for zone boundaries. ---*/

  CGeometry *fine_grid = geometry[iMesh-1];
  CConfig *config = config_container;

  /*--- Local variables ---*/

  unsigned long iPoint, Index_CoarseCV, CVPoint, iElem, iVertex, jPoint, iteration, nVertexS, nVertexR,
                nBufferS_Vector, nBufferR_Vector, iParent, jVertex,Local_nPointCoarse, Local_nPointFine, Global_nPointCoarse, Global_nPointFine,
                *Buffer_Receive_Parent = nullptr, *Buffer_Send_Parent = nullptr, *Buffer_Receive_Children = nullptr, *Buffer_Send_Children = nullptr,
                *Parent_Remote = nullptr,         *Children_Remote = nullptr,    *Parent_Local = nullptr,            *Children_Local = nullptr;
  short marker_seed;
  bool agglomerate_seed = true;
  unsigned short nChildren, iNode, counter, iMarker, jMarker, priority, MarkerS, MarkerR, *nChildren_MPI;
  vector<unsigned long> Suitable_Indirect_Neighbors, Aux_Parent;
  vector<unsigned long>::iterator it;

  unsigned short nMarker_Max = config->GetnMarker_Max();

  unsigned short *copy_marker = new unsigned short [nMarker_Max];

#ifdef HAVE_MPI
  int send_to, receive_from;
  SU2_MPI::Status status;
#endif

  nDim = fine_grid->GetnDim(); // Write the number of dimensions of the coarse grid.

  /*--- Create a queue system to deo the agglomeration
   1st) More than two markers ---> Vertices (never agglomerate)
   2nd) Two markers ---> Edges (agglomerate if same BC, never agglomerate if different BC)
   3rd) One marker ---> Surface (always agglomarate)
   4th) No marker ---> Internal Volume (always agglomarate) ---*/

  /*--- Set a marker to indicate indirect agglomeration ---*/

  if (iMesh == MESH_1) {

    for (iPoint = 0; iPoint < fine_grid->GetnPoint(); iPoint ++)
      fine_grid->nodes->SetAgglomerate_Indirect(iPoint, false);

    for (iElem = 0; iElem < fine_grid->GetnElem(); iElem++) {
      if ((fine_grid->elem[iElem]->GetVTK_Type() == HEXAHEDRON) ||
          (fine_grid->elem[iElem]->GetVTK_Type() == QUADRILATERAL)) {
        for (iNode = 0; iNode < fine_grid->elem[iElem]->GetnNodes(); iNode++) {
          iPoint = fine_grid->elem[iElem]->GetNode(iNode);
          fine_grid->nodes->SetAgglomerate_Indirect(iPoint, true);
        }
      }
    }

  }

  /*--- Create the coarse grid structure using as baseline the fine grid ---*/

  CMultiGridQueue MGQueue_InnerCV(fine_grid->GetnPoint());

  nPointNode = fine_grid->GetnPoint();

  nodes = new CPoint(fine_grid->GetnPoint(), nDim, iMesh, config);

  Index_CoarseCV = 0;

  /*--- The first step is the boundary agglomeration. ---*/

  for (iMarker = 0; iMarker < fine_grid->GetnMarker(); iMarker++) {

    for (iVertex = 0; iVertex < fine_grid->GetnVertex(iMarker); iVertex++) {
      iPoint = fine_grid->vertex[iMarker][iVertex]->GetNode();

      /*--- If the element has not being previously agglomerated and it belongs
       to the physical domain, then the agglomeration is studied ---*/

      if ((fine_grid->nodes->GetAgglomerate(iPoint) == false) &&
          (fine_grid->nodes->GetDomain(iPoint)) &&
          (GeometricalCheck(iPoint, fine_grid, config))) {

        nChildren = 1;

        /*--- We set an index for the parent control volume ---*/

        fine_grid->nodes->SetParent_CV(iPoint, Index_CoarseCV);

        /*--- We add the seed point (child) to the parent control volume ---*/

        nodes->SetChildren_CV(Index_CoarseCV, 0, iPoint);
        agglomerate_seed = true; counter = 0; marker_seed = iMarker;

        /*--- For a particular point in the fine grid we save all the markers
         that are in that point ---*/

        for (jMarker = 0; jMarker < fine_grid->GetnMarker(); jMarker ++)
          if (fine_grid->nodes->GetVertex(iPoint, jMarker) != -1) {
            copy_marker[counter] = jMarker;
            counter++;
          }

        /*--- To aglomerate a vertex it must have only one physical bc!!
         This can be improved. If there is only a marker, it is a good
         candidate for agglomeration ---*/

        if (counter == 1) agglomerate_seed = true;

        /*--- If there are two markers, we will aglomerate if one of the
         marker is SEND_RECEIVE ---*/

        if (counter == 2) {
          if ((config->GetMarker_All_KindBC(copy_marker[0]) == SEND_RECEIVE) ||
              (config->GetMarker_All_KindBC(copy_marker[1]) == SEND_RECEIVE)) agglomerate_seed = true;
          else agglomerate_seed = false;
        }

        /*--- If there are more than 2 markers, the aglomeration will be discarted ---*/

        if (counter > 2) agglomerate_seed = false;

        /*--- If the seed can be agglomerated, we try to agglomerate more points ---*/

        if (agglomerate_seed) {

          /*--- Now we do a sweep over all the nodes that surround the seed point ---*/

          for (iNode = 0; iNode < fine_grid->nodes->GetnPoint(iPoint); iNode ++) {

            CVPoint = fine_grid->nodes->GetPoint(iPoint, iNode);

            /*--- The new point can be agglomerated ---*/

            if (SetBoundAgglomeration(CVPoint, marker_seed, fine_grid, config)) {

              /*--- We set the value of the parent ---*/

              fine_grid->nodes->SetParent_CV(CVPoint, Index_CoarseCV);

              /*--- We set the value of the child ---*/

              nodes->SetChildren_CV(Index_CoarseCV, nChildren, CVPoint);
              nChildren++;
            }

          }

          Suitable_Indirect_Neighbors.clear();

          if (fine_grid->nodes->GetAgglomerate_Indirect(iPoint))
            SetSuitableNeighbors(&Suitable_Indirect_Neighbors, iPoint, Index_CoarseCV, fine_grid);

          /*--- Now we do a sweep over all the indirect nodes that can be added ---*/

          for (iNode = 0; iNode < Suitable_Indirect_Neighbors.size(); iNode ++) {

            CVPoint = Suitable_Indirect_Neighbors[iNode];

            /*--- The new point can be agglomerated ---*/

            if (SetBoundAgglomeration(CVPoint, marker_seed, fine_grid, config)) {

              /*--- We set the value of the parent ---*/

              fine_grid->nodes->SetParent_CV(CVPoint, Index_CoarseCV);

              /*--- We set the indirect agglomeration information ---*/

              if (fine_grid->nodes->GetAgglomerate_Indirect(CVPoint))
                nodes->SetAgglomerate_Indirect(Index_CoarseCV, true);

              /*--- We set the value of the child ---*/

              nodes->SetChildren_CV(Index_CoarseCV, nChildren, CVPoint);
              nChildren++;
            }
          }


        }

        /*--- Update the number of child of the control volume ---*/

        nodes->SetnChildren_CV(Index_CoarseCV, nChildren);
        Index_CoarseCV++;
      }
    }
  }

  /*--- Agglomerate all the nodes that have more than one physical boundary condition,
   Maybe here we can add the posibility of merging the vertex that have the same number,
   and kind  of markers---*/

  for (iMarker = 0; iMarker < fine_grid->GetnMarker(); iMarker++)
    for (iVertex = 0; iVertex < fine_grid->GetnVertex(iMarker); iVertex++) {
      iPoint = fine_grid->vertex[iMarker][iVertex]->GetNode();
      if ((fine_grid->nodes->GetAgglomerate(iPoint) == false) &&
          (fine_grid->nodes->GetDomain(iPoint))) {
        fine_grid->nodes->SetParent_CV(iPoint, Index_CoarseCV);
        nodes->SetChildren_CV(Index_CoarseCV, 0, iPoint);
        nodes->SetnChildren_CV(Index_CoarseCV, 1);
        Index_CoarseCV++;
      }
    }

  /*--- Update the queue with the results from the boundary agglomeration ---*/

  for (iPoint = 0; iPoint < fine_grid->GetnPoint(); iPoint ++) {

    /*--- The CV has been agglomerated, remove form the list ---*/

    if (fine_grid->nodes->GetAgglomerate(iPoint) == true) {

      MGQueue_InnerCV.RemoveCV(iPoint);

    }

    else {

      /*--- Count the number of agglomerated neighbors, and modify the queue ---*/

      priority = 0;
      for (iNode = 0; iNode < fine_grid->nodes->GetnPoint(iPoint); iNode ++) {
        jPoint = fine_grid->nodes->GetPoint(iPoint, iNode);
        if (fine_grid->nodes->GetAgglomerate(jPoint) == true) priority++;
      }
      MGQueue_InnerCV.MoveCV(iPoint, priority);
    }
  }

  /*--- Agglomerate the domain nodes ---*/

  iteration = 0;
  while (!MGQueue_InnerCV.EmptyQueue() && (iteration < fine_grid->GetnPoint())) {

    iPoint = MGQueue_InnerCV.NextCV();
    iteration ++;

    /*--- If the element has not being previously agglomerated, belongs to the physical domain,
     and satisfies several geometrical criteria then the seed CV is acepted for agglomeration ---*/

    if ((fine_grid->nodes->GetAgglomerate(iPoint) == false) &&
        (fine_grid->nodes->GetDomain(iPoint)) &&
        (GeometricalCheck(iPoint, fine_grid, config))) {

      nChildren = 1;

      /*--- We set an index for the parent control volume ---*/

      fine_grid->nodes->SetParent_CV(iPoint, Index_CoarseCV);

      /*--- We add the seed point (child) to the parent control volume ---*/

      nodes->SetChildren_CV(Index_CoarseCV, 0, iPoint);

      /*--- Update the queue with the seed point (remove the seed and
       increase the priority of the neighbors) ---*/

      MGQueue_InnerCV.Update(iPoint, fine_grid);

      /*--- Now we do a sweep over all the nodes that surround the seed point ---*/

      for (iNode = 0; iNode < fine_grid->nodes->GetnPoint(iPoint); iNode ++) {

        CVPoint = fine_grid->nodes->GetPoint(iPoint, iNode);

        /*--- Determine if the CVPoint can be agglomerated ---*/

        if ((fine_grid->nodes->GetAgglomerate(CVPoint) == false) &&
            (fine_grid->nodes->GetDomain(CVPoint)) &&
            (GeometricalCheck(CVPoint, fine_grid, config))) {

          /*--- We set the value of the parent ---*/

          fine_grid->nodes->SetParent_CV(CVPoint, Index_CoarseCV);

          /*--- We set the value of the child ---*/

          nodes->SetChildren_CV(Index_CoarseCV, nChildren, CVPoint);
          nChildren++;

          /*--- Update the queue with the new control volume (remove the CV and
           increase the priority of the neighbors) ---*/

          MGQueue_InnerCV.Update(CVPoint, fine_grid);

        }

      }

      /*--- Subrotuine to identify the indirect neighbors ---*/

      Suitable_Indirect_Neighbors.clear();
      if (fine_grid->nodes->GetAgglomerate_Indirect(iPoint))
        SetSuitableNeighbors(&Suitable_Indirect_Neighbors, iPoint, Index_CoarseCV, fine_grid);

      /*--- Now we do a sweep over all the indirect nodes that can be added ---*/

      for (iNode = 0; iNode < Suitable_Indirect_Neighbors.size(); iNode ++) {

        CVPoint = Suitable_Indirect_Neighbors[iNode];

        /*--- The new point can be agglomerated ---*/

        if ((fine_grid->nodes->GetAgglomerate(CVPoint) == false) &&
            (fine_grid->nodes->GetDomain(CVPoint))) {

          /*--- We set the value of the parent ---*/

          fine_grid->nodes->SetParent_CV(CVPoint, Index_CoarseCV);

          /*--- We set the indirect agglomeration information ---*/

          if (fine_grid->nodes->GetAgglomerate_Indirect(CVPoint))
            nodes->SetAgglomerate_Indirect(Index_CoarseCV, true);

          /*--- We set the value of the child ---*/

          nodes->SetChildren_CV(Index_CoarseCV, nChildren, CVPoint);
          nChildren++;

          /*--- Update the queue with the new control volume (remove the CV and
           increase the priority of the neighbors) ---*/

          MGQueue_InnerCV.Update(CVPoint, fine_grid);

        }
      }

      /*--- Update the number of control of childrens ---*/

      nodes->SetnChildren_CV(Index_CoarseCV, nChildren);
      Index_CoarseCV++;
    }
    else {

      /*--- The seed point can not be agglomerated because of size, domain, streching, etc.
       move the point to the lowest priority ---*/

      MGQueue_InnerCV.MoveCV(iPoint, -1);
    }

  }

  /*--- Add all the elements that have not being agglomerated, in the previous stage ---*/

  for (iPoint = 0; iPoint < fine_grid->GetnPoint(); iPoint ++) {
    if ((fine_grid->nodes->GetAgglomerate(iPoint) == false) && (fine_grid->nodes->GetDomain(iPoint))) {

      nChildren = 1;
      fine_grid->nodes->SetParent_CV(iPoint, Index_CoarseCV);
      if (fine_grid->nodes->GetAgglomerate_Indirect(iPoint))
        nodes->SetAgglomerate_Indirect(Index_CoarseCV, true);
      nodes->SetChildren_CV(Index_CoarseCV, 0, iPoint);
      nodes->SetnChildren_CV(Index_CoarseCV, nChildren);
      Index_CoarseCV++;

    }
  }

  nPointDomain = Index_CoarseCV;

  /*--- Check that there are no hanging nodes ---*/

  unsigned long iFinePoint, iFinePoint_Neighbor, iCoarsePoint, iCoarsePoint_Complete;
  unsigned short iChildren;

  /*--- Find the point surrounding a point ---*/
  {
    /*--- Temporary, CPoint (nodes) then compresses the information ---*/
    vector<vector<unsigned long> > points(fine_grid->GetnPoint());

    for (iCoarsePoint = 0; iCoarsePoint < nPointDomain; iCoarsePoint ++) {
      for (iChildren = 0; iChildren <  nodes->GetnChildren_CV(iCoarsePoint); iChildren ++) {
        iFinePoint = nodes->GetChildren_CV(iCoarsePoint, iChildren);
        for (iNode = 0; iNode < fine_grid->nodes->GetnPoint(iFinePoint); iNode ++) {
          iFinePoint_Neighbor = fine_grid->nodes->GetPoint(iFinePoint, iNode);
          iParent = fine_grid->nodes->GetParent_CV(iFinePoint_Neighbor);
          if (iParent != iCoarsePoint) {
            auto End = points[iCoarsePoint].end();
            if (find(points[iCoarsePoint].begin(), End, iParent) == End)
              points[iCoarsePoint].push_back(iParent);
          }
        }
      }
    }
    nodes->SetPoints(points);
  }

  /*--- Detect isolated points and merge them with its correct neighbor ---*/

  for (iCoarsePoint = 0; iCoarsePoint < nPointDomain; iCoarsePoint ++) {

    if (nodes->GetnPoint(iCoarsePoint) == 1) {

      /*--- Find the neighbor of the isolated point. This neighbor is the right control volume ---*/

      iCoarsePoint_Complete = nodes->GetPoint(iCoarsePoint, 0);

      /*--- Add the children to the connected control volume (and modify it parent indexing).
       Identify the child CV from the finest grid and added to the correct control volume.
       Set the parent CV of iFinePoint. Instead of using the original
       (iCoarsePoint) one use the new one (iCoarsePoint_Complete) ---*/

      nChildren = nodes->GetnChildren_CV(iCoarsePoint_Complete);

      for (iChildren = 0; iChildren <  nodes->GetnChildren_CV(iCoarsePoint); iChildren ++) {
        iFinePoint = nodes->GetChildren_CV(iCoarsePoint, iChildren);
        nodes->SetChildren_CV(iCoarsePoint_Complete, nChildren, iFinePoint);
        nChildren++;
        fine_grid->nodes->SetParent_CV(iFinePoint, iCoarsePoint_Complete);
      }

      /*--- Update the number of children control volumes ---*/

      nodes->SetnChildren_CV(iCoarsePoint_Complete, nChildren);
      nodes->SetnChildren_CV(iCoarsePoint, 0);

    }
  }

  //  unsigned long iPointFree = nPointDomain-1;
  //  iCoarsePoint = 0;
  //
  //  do {
  //
  //    if (nodes->GetnChildren_CV(iCoarsePoint) == 0) {
  //
  //      while (nodes->GetnChildren_CV(iPointFree) == 0) {
  //        Index_CoarseCV--;
  //        iPointFree--;
  //      }
  //
  //      nChildren = nodes->GetnChildren_CV(iPointFree);
  //      for (iChildren = 0; iChildren <  nChildren; iChildren ++) {
  //        iFinePoint = nodes->GetChildren_CV(iPointFree, iChildren);
  //        nodes->SetChildren_CV(iCoarsePoint, iChildren, iFinePoint);
  //        fine_grid->nodes->SetParent_CV(iFinePoint, iCoarsePoint);
  //      }
  //      nodes->SetnChildren_CV(iCoarsePoint, nChildren);
  //      nodes->SetnChildren_CV(iPointFree, 0);
  //
  //      Index_CoarseCV--;
  //      iPointFree--;
  //
  //    }
  //
  //    iCoarsePoint++;
  //
  //  } while ((iCoarsePoint-1) < Index_CoarseCV);
  //
  //  nPointDomain = Index_CoarseCV;

  /*--- Reset the point surrounding a point ---*/

  nodes->ResetPoints();

  /*--- Dealing with MPI parallelization, the objective is that the received nodes must be agglomerated
   in the same way as the donor nodes. Send the node agglomeration information of the donor
   (parent and children), Sending only occurs with MPI ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {

    if ((config->GetMarker_All_KindBC(iMarker) == SEND_RECEIVE) &&
        (config->GetMarker_All_SendRecv(iMarker) > 0)) {

      MarkerS = iMarker;  MarkerR = iMarker+1;

#ifdef HAVE_MPI
      send_to = config->GetMarker_All_SendRecv(MarkerS)-1;
      receive_from = abs(config->GetMarker_All_SendRecv(MarkerR))-1;
#endif

      nVertexS = fine_grid->nVertex[MarkerS];   nVertexR = fine_grid->nVertex[MarkerR];
      nBufferS_Vector = nVertexS;               nBufferR_Vector = nVertexR;

      /*--- Allocate Receive and send buffers  ---*/

      Buffer_Receive_Children = new unsigned long [nBufferR_Vector];
      Buffer_Send_Children = new unsigned long [nBufferS_Vector];

      Buffer_Receive_Parent = new unsigned long [nBufferR_Vector];
      Buffer_Send_Parent = new unsigned long [nBufferS_Vector];

      /*--- Copy the information that should be sended ---*/

      for (iVertex = 0; iVertex < nVertexS; iVertex++) {
        iPoint = fine_grid->vertex[MarkerS][iVertex]->GetNode();
        Buffer_Send_Children[iVertex] = iPoint;
        Buffer_Send_Parent[iVertex] = fine_grid->nodes->GetParent_CV(iPoint);
      }

#ifdef HAVE_MPI
      /*--- Send/Receive information using Sendrecv ---*/
      SU2_MPI::Sendrecv(Buffer_Send_Children, nBufferS_Vector, MPI_UNSIGNED_LONG, send_to,0,
                   Buffer_Receive_Children, nBufferR_Vector, MPI_UNSIGNED_LONG, receive_from,0, MPI_COMM_WORLD, &status);
      SU2_MPI::Sendrecv(Buffer_Send_Parent, nBufferS_Vector, MPI_UNSIGNED_LONG, send_to,1,
                   Buffer_Receive_Parent, nBufferR_Vector, MPI_UNSIGNED_LONG, receive_from,1, MPI_COMM_WORLD, &status);
#else
      /*--- Receive information without MPI ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {
        Buffer_Receive_Children[iVertex] = Buffer_Send_Children[iVertex];
        Buffer_Receive_Parent[iVertex] = Buffer_Send_Parent[iVertex];
      }
#endif

      /*--- Deallocate send buffer ---*/

      delete [] Buffer_Send_Children;
      delete [] Buffer_Send_Parent;

      /*--- Create a list of the parent nodes without repeated parents ---*/

      Aux_Parent.clear();
      for (iVertex = 0; iVertex < nVertexR; iVertex++)
        Aux_Parent.push_back (Buffer_Receive_Parent[iVertex]);

      sort(Aux_Parent.begin(), Aux_Parent.end());
      it = unique(Aux_Parent.begin(), Aux_Parent.end());
      Aux_Parent.resize(it - Aux_Parent.begin());

      /*--- Allocate some structures ---*/

      Parent_Remote = new unsigned long[nVertexR];
      Children_Remote = new unsigned long[nVertexR];
      Parent_Local = new unsigned long[nVertexR];
      Children_Local = new unsigned long[nVertexR];

      /*--- Create the local vector and remote for the parents and the children ---*/

      for (iVertex = 0; iVertex < nVertexR; iVertex++) {

        Parent_Remote[iVertex] = Buffer_Receive_Parent[iVertex];

        /*--- We use the same sorting as in the donor domain ---*/

        for (jVertex = 0; jVertex < Aux_Parent.size(); jVertex++) {
          if (Parent_Remote[iVertex] == Aux_Parent[jVertex]) {
            Parent_Local[iVertex] = jVertex + Index_CoarseCV;
            break;
          }
        }

        Children_Remote[iVertex] = Buffer_Receive_Children[iVertex];
        Children_Local[iVertex] = fine_grid->vertex[MarkerR][iVertex]->GetNode();

      }

      Index_CoarseCV += Aux_Parent.size();

      nChildren_MPI = new unsigned short [Index_CoarseCV];
      for (iParent = 0; iParent < Index_CoarseCV; iParent++)
        nChildren_MPI[iParent] = 0;

      /*--- Create the final structure ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {

        /*--- Be careful, it is possible that a node change the agglomeration configuration, the priority
         is always, when receive the information ---*/

        fine_grid->nodes->SetParent_CV(Children_Local[iVertex], Parent_Local[iVertex]);
        nodes->SetChildren_CV(Parent_Local[iVertex], nChildren_MPI[Parent_Local[iVertex]], Children_Local[iVertex]);
        nChildren_MPI[Parent_Local[iVertex]]++;
        nodes->SetnChildren_CV(Parent_Local[iVertex], nChildren_MPI[Parent_Local[iVertex]]);
        nodes->SetDomain(Parent_Local[iVertex], false);

      }

      /*--- Deallocate auxiliar structures ---*/

      delete[] nChildren_MPI;
      delete[] Parent_Remote;
      delete[] Children_Remote;
      delete[] Parent_Local;
      delete[] Children_Local;

      /*--- Deallocate receive buffer ---*/

      delete [] Buffer_Receive_Children;
      delete [] Buffer_Receive_Parent;

    }

  }

  /*--- Update the number of points after the MPI agglomeration ---*/

  nPoint = Index_CoarseCV;

  /*--- Console output with the summary of the agglomeration ---*/

  Local_nPointCoarse = nPoint;
  Local_nPointFine = fine_grid->GetnPoint();

  SU2_MPI::Allreduce(&Local_nPointCoarse, &Global_nPointCoarse, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
  SU2_MPI::Allreduce(&Local_nPointFine, &Global_nPointFine, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

  su2double Coeff = 1.0, CFL = 0.0, factor = 1.5;

  if (iMesh != MESH_0) {
    if (nDim == 2) Coeff = pow(su2double(Global_nPointFine)/su2double(Global_nPointCoarse), 1./2.);
    if (nDim == 3) Coeff = pow(su2double(Global_nPointFine)/su2double(Global_nPointCoarse), 1./3.);
    CFL = factor*config->GetCFL(iMesh-1)/Coeff;
    config->SetCFL(iMesh, CFL);
  }

  su2double ratio = su2double(Global_nPointFine)/su2double(Global_nPointCoarse);

  if (((nDim == 2) && (ratio < 2.5)) ||
      ((nDim == 3) && (ratio < 2.5))) {
    config->SetMGLevels(iMesh-1);
  }
  else {
    if (rank == MASTER_NODE) {
      PrintingToolbox::CTablePrinter MGTable(&std::cout);
      MGTable.AddColumn("MG Level", 10);
      MGTable.AddColumn("CVs", 10);
      MGTable.AddColumn("Aggl. Rate", 10);
      MGTable.AddColumn("CFL", 10);
      MGTable.SetAlign(PrintingToolbox::CTablePrinter::RIGHT);


      if (iMesh == 1){
        MGTable.PrintHeader();
        MGTable << iMesh - 1 << Global_nPointFine << "1/1.00" << config->GetCFL(iMesh -1);
      }
      stringstream ss;
      ss << "1/" << std::setprecision(3) << ratio;
      MGTable << iMesh << Global_nPointCoarse << ss.str() << CFL;
      if (iMesh == config->GetnMGLevels()){
        MGTable.PrintFooter();
      }
    }
  }

  edgeColorGroupSize = config->GetEdgeColoringGroupSize();

  delete [] copy_marker;

}

bool CMultiGridGeometry::SetBoundAgglomeration(unsigned long CVPoint, short marker_seed, CGeometry *fine_grid, CConfig *config) {

  bool agglomerate_CV = false;
  unsigned short counter, jMarker;

  unsigned short nMarker_Max = config->GetnMarker_Max();

  unsigned short *copy_marker = new unsigned short [nMarker_Max];

  /*--- Basic condition, the element has not being previously agglomerated, it belongs to the domain,
   and has passed some basic geometrical check ---*/

  if ((fine_grid->nodes->GetAgglomerate(CVPoint) == false) &&
      (fine_grid->nodes->GetDomain(CVPoint)) &&
      (GeometricalCheck(CVPoint, fine_grid, config))) {

    /*--- If the element belong to the boundary, we must be careful ---*/

    if (fine_grid->nodes->GetBoundary(CVPoint)) {

      /*--- Identify the markers of the vertex that we want to agglomerate ---*/

      counter = 0;
      for (jMarker = 0; jMarker < fine_grid->GetnMarker(); jMarker ++)
        if (fine_grid->nodes->GetVertex(CVPoint, jMarker) != -1) {
          copy_marker[counter] = jMarker;
          counter++;
        }

      /*--- The basic condition is that the aglomerated vertex must have the same physical marker,
       but eventually a send-receive condition ---*/

      /*--- Only one marker in the vertex that is going to be aglomerated ---*/

      if (counter == 1) {

        /*--- We agglomerate if there is only a marker and is the same marker as the seed marker ---*/

        if (copy_marker[0] == marker_seed)
          agglomerate_CV = true;

        /*--- If there is only a marker, but the marker is the SEND_RECEIVE ---*/

        if (config->GetMarker_All_KindBC(copy_marker[0]) == SEND_RECEIVE)
          agglomerate_CV = true;

      }

      /*--- If there are two markers in the vertex that is going to be aglomerated ---*/

      if (counter == 2) {

        /*--- First we verify that the seed is a physical boundary ---*/

        if (config->GetMarker_All_KindBC(marker_seed) != SEND_RECEIVE) {

          /*--- Then we check that one of the marker is equal to the seed marker, and the other is send/receive ---*/

          if (((copy_marker[0] == marker_seed) && (config->GetMarker_All_KindBC(copy_marker[1]) == SEND_RECEIVE)) ||
              ((config->GetMarker_All_KindBC(copy_marker[0]) == SEND_RECEIVE) && (copy_marker[1] == marker_seed)))
            agglomerate_CV = true;
        }

      }

    }

    /*--- If the element belong to the domain, it is allways aglomerated ---*/

    else { agglomerate_CV = true; }

  }

  delete [] copy_marker;

  return agglomerate_CV;

}


bool CMultiGridGeometry::GeometricalCheck(unsigned long iPoint, CGeometry *fine_grid, CConfig *config) {

  su2double max_dimension = 1.2;

  /*--- Evaluate the total size of the element ---*/

  bool Volume = true;
  su2double ratio = pow(fine_grid->nodes->GetVolume(iPoint), 1.0/su2double(nDim))*max_dimension;
  su2double limit = pow(config->GetDomainVolume(), 1.0/su2double(nDim));
  if ( ratio > limit ) Volume = false;

  /*--- Evaluate the stretching of the element ---*/

  bool Stretching = true;

  /* unsigned short iNode, iDim;
   unsigned long jPoint;
   su2double *Coord_i = fine_grid->nodes->GetCoord(iPoint);
   su2double max_dist = 0.0 ; su2double min_dist = 1E20;
   for (iNode = 0; iNode < fine_grid->nodes->GetnPoint(iPoint); iNode ++) {
   jPoint = fine_grid->nodes->GetPoint(iPoint, iNode);
   su2double *Coord_j = fine_grid->nodes->GetCoord(jPoint);
   su2double distance = 0.0;
   for (iDim = 0; iDim < nDim; iDim++)
   distance += (Coord_j[iDim]-Coord_i[iDim])*(Coord_j[iDim]-Coord_i[iDim]);
   distance = sqrt(distance);
   max_dist = max(distance, max_dist);
   min_dist = min(distance, min_dist);
   }
   if ( max_dist/min_dist > 100.0 ) Stretching = false;*/

  return (Stretching && Volume);

}

void CMultiGridGeometry::SetSuitableNeighbors(vector<unsigned long> *Suitable_Indirect_Neighbors, unsigned long iPoint,
                                              unsigned long Index_CoarseCV, CGeometry *fine_grid) {

  unsigned long jPoint, kPoint, lPoint;
  unsigned short iNode, jNode, iNeighbor, jNeighbor, kNode;
  bool SecondNeighborSeed, ThirdNeighborSeed;
  vector<unsigned long>::iterator it;

  /*--- Create a list with the first neighbors, including the seed ---*/

  vector<unsigned long> First_Neighbor_Points;
  First_Neighbor_Points.push_back(iPoint);
  for (iNode = 0; iNode < fine_grid->nodes->GetnPoint(iPoint); iNode ++) {
    jPoint = fine_grid->nodes->GetPoint(iPoint, iNode);
    First_Neighbor_Points.push_back(jPoint);
  }

  /*--- Create a list with the second neighbors, without first, and seed neighbors ---*/

  vector<unsigned long> Second_Neighbor_Points, Second_Origin_Points, Suitable_Second_Neighbors;

  for (iNode = 0; iNode < fine_grid->nodes->GetnPoint(iPoint); iNode ++) {
    jPoint = fine_grid->nodes->GetPoint(iPoint, iNode);

    for (jNode = 0; jNode < fine_grid->nodes->GetnPoint(jPoint); jNode ++) {
      kPoint = fine_grid->nodes->GetPoint(jPoint, jNode);

      /*--- Check that the second neighbor do not belong to the first neighbor or the seed ---*/

      SecondNeighborSeed = true;
      for (iNeighbor = 0; iNeighbor < First_Neighbor_Points.size(); iNeighbor ++)
        if (kPoint == First_Neighbor_Points[iNeighbor]) {
          SecondNeighborSeed = false; break;
        }

      if (SecondNeighborSeed) {
        Second_Neighbor_Points.push_back(kPoint);
        Second_Origin_Points.push_back(jPoint);
      }

    }
  }

  /*---  Identify those second neighbors that are repeated (candidate to be added) ---*/

  for (iNeighbor = 0; iNeighbor < Second_Neighbor_Points.size(); iNeighbor ++)

    for (jNeighbor = 0; jNeighbor < Second_Neighbor_Points.size(); jNeighbor ++)

    /*--- Repeated second neighbor with different origin ---*/

      if ((Second_Neighbor_Points[iNeighbor] == Second_Neighbor_Points[jNeighbor]) &&
          (Second_Origin_Points[iNeighbor] != Second_Origin_Points[jNeighbor]) &&
          (iNeighbor < jNeighbor)) {

        Suitable_Indirect_Neighbors->push_back(Second_Neighbor_Points[iNeighbor]);

        /*--- Create alist with the suitable second neighbor, that we will use
         to compute the third neighbors --*/

        Suitable_Second_Neighbors.push_back(Second_Neighbor_Points[iNeighbor]);

      }


  /*--- Remove repeated from the suitable second neighbors ---*/

  sort(Suitable_Second_Neighbors.begin(), Suitable_Second_Neighbors.end());
  it = unique(Suitable_Second_Neighbors.begin(), Suitable_Second_Neighbors.end());
  Suitable_Second_Neighbors.resize(it - Suitable_Second_Neighbors.begin());

  /*--- Remove repeated from first neighbors ---*/

  sort(First_Neighbor_Points.begin(), First_Neighbor_Points.end());
  it = unique(First_Neighbor_Points.begin(), First_Neighbor_Points.end());
  First_Neighbor_Points.resize(it - First_Neighbor_Points.begin());

  /*--- Create a list with the third neighbors, without first, second, and seed neighbors ---*/

  vector<unsigned long> Third_Neighbor_Points, Third_Origin_Points;

  for (jNode = 0; jNode < Suitable_Second_Neighbors.size(); jNode ++) {
    kPoint = Suitable_Second_Neighbors[jNode];

    for (kNode = 0; kNode < fine_grid->nodes->GetnPoint(kPoint); kNode ++) {
      lPoint = fine_grid->nodes->GetPoint(kPoint, kNode);

      /*--- Check that the third neighbor do not belong to the first neighbors or the seed ---*/

      ThirdNeighborSeed = true;

      for (iNeighbor = 0; iNeighbor < First_Neighbor_Points.size(); iNeighbor ++)
        if (lPoint == First_Neighbor_Points[iNeighbor]) {
          ThirdNeighborSeed = false;
          break;
        }

      /*--- Check that the third neighbor do not belong to the second neighbors ---*/

      for (iNeighbor = 0; iNeighbor < Suitable_Second_Neighbors.size(); iNeighbor ++)
        if (lPoint == Suitable_Second_Neighbors[iNeighbor]) {
          ThirdNeighborSeed = false;
          break;
        }

      if (ThirdNeighborSeed) {
        Third_Neighbor_Points.push_back(lPoint);
        Third_Origin_Points.push_back(kPoint);
      }

    }
  }

  /*---  Identify those third neighbors that are repeated (candidate to be added) ---*/

  for (iNeighbor = 0; iNeighbor < Third_Neighbor_Points.size(); iNeighbor ++)
    for (jNeighbor = 0; jNeighbor < Third_Neighbor_Points.size(); jNeighbor ++)

    /*--- Repeated second neighbor with different origin ---*/

      if ((Third_Neighbor_Points[iNeighbor] == Third_Neighbor_Points[jNeighbor]) &&
          (Third_Origin_Points[iNeighbor] != Third_Origin_Points[jNeighbor]) &&
          (iNeighbor < jNeighbor)) {

        Suitable_Indirect_Neighbors->push_back(Third_Neighbor_Points[iNeighbor]);

      }

  /*--- Remove repeated from Suitable Indirect Neighbors List ---*/

  sort(Suitable_Indirect_Neighbors->begin(), Suitable_Indirect_Neighbors->end());
  it = unique(Suitable_Indirect_Neighbors->begin(), Suitable_Indirect_Neighbors->end());
  Suitable_Indirect_Neighbors->resize(it - Suitable_Indirect_Neighbors->begin());

}

void CMultiGridGeometry::SetPoint_Connectivity(CGeometry *fine_grid) {

  unsigned long iFinePoint, iFinePoint_Neighbor, iParent, iCoarsePoint;
  unsigned short iChildren, iNode;

  /*--- Set the point surrounding a point ---*/

  vector<vector<unsigned long> > points(nPoint);

  for (iCoarsePoint = 0; iCoarsePoint < nPoint; iCoarsePoint ++) {
    for (iChildren = 0; iChildren <  nodes->GetnChildren_CV(iCoarsePoint); iChildren ++) {
      iFinePoint = nodes->GetChildren_CV(iCoarsePoint, iChildren);
      for (iNode = 0; iNode < fine_grid->nodes->GetnPoint(iFinePoint); iNode ++) {
        iFinePoint_Neighbor = fine_grid->nodes->GetPoint(iFinePoint, iNode);
        iParent = fine_grid->nodes->GetParent_CV(iFinePoint_Neighbor);
        if (iParent != iCoarsePoint) {
          auto End = points[iCoarsePoint].end();
          if (find(points[iCoarsePoint].begin(), End, iParent) == End)
            points[iCoarsePoint].push_back(iParent);
        }
      }
    }
  }
  nodes->SetPoints(points);

  /*--- Set the number of neighbors variable, this is
   important for JST and multigrid in parallel ---*/

  for (iCoarsePoint = 0; iCoarsePoint < nPoint; iCoarsePoint ++)
    nodes->SetnNeighbor(iCoarsePoint, nodes->GetnPoint(iCoarsePoint));

}

void CMultiGridGeometry::SetVertex(CGeometry *fine_grid, CConfig *config) {
  unsigned long  iVertex, iFinePoint, iCoarsePoint;
  unsigned short iMarker, iMarker_Tag, iChildren;

  nMarker = fine_grid->GetnMarker();
  unsigned short nMarker_Max = config->GetnMarker_Max();

  /*--- If any children node belong to the boundary then the entire control
   volume will belong to the boundary ---*/
  for (iCoarsePoint = 0; iCoarsePoint < nPoint; iCoarsePoint ++)
    for (iChildren = 0; iChildren < nodes->GetnChildren_CV(iCoarsePoint); iChildren ++) {
      iFinePoint = nodes->GetChildren_CV(iCoarsePoint, iChildren);
      if (fine_grid->nodes->GetBoundary(iFinePoint)) {
        nodes->SetBoundary(iCoarsePoint, nMarker);
        break;
      }
    }

  vertex = new CVertex**[nMarker];
  nVertex = new unsigned long [nMarker];

  Tag_to_Marker = new string [nMarker_Max];
  for (iMarker_Tag = 0; iMarker_Tag < nMarker_Max; iMarker_Tag++)
    Tag_to_Marker[iMarker_Tag] = fine_grid->GetMarker_Tag(iMarker_Tag);

  /*--- Compute the number of vertices to do the dimensionalization ---*/
  for (iMarker = 0; iMarker < nMarker; iMarker++) nVertex[iMarker] = 0;


  for (iCoarsePoint = 0; iCoarsePoint < nPoint; iCoarsePoint ++) {
    if (nodes->GetBoundary(iCoarsePoint)) {
      for (iChildren = 0; iChildren < nodes->GetnChildren_CV(iCoarsePoint); iChildren ++) {
        iFinePoint = nodes->GetChildren_CV(iCoarsePoint, iChildren);
        for (iMarker = 0; iMarker < nMarker; iMarker ++) {
          if ((fine_grid->nodes->GetVertex(iFinePoint, iMarker) != -1) && (nodes->GetVertex(iCoarsePoint, iMarker) == -1)) {
            iVertex = nVertex[iMarker];
            nodes->SetVertex(iCoarsePoint, iVertex, iMarker);
            nVertex[iMarker]++;
          }
        }
      }
    }
  }

  for (iMarker = 0; iMarker < nMarker; iMarker++) {
    vertex[iMarker] = new CVertex* [fine_grid->GetnVertex(iMarker)+1];
    nVertex[iMarker] = 0;
  }

  for (iCoarsePoint = 0; iCoarsePoint < nPoint; iCoarsePoint ++)
    if (nodes->GetBoundary(iCoarsePoint))
      for (iMarker = 0; iMarker < nMarker; iMarker ++)
        nodes->SetVertex(iCoarsePoint, -1, iMarker);

  for (iMarker = 0; iMarker < nMarker; iMarker++) nVertex[iMarker] = 0;

  for (iCoarsePoint = 0; iCoarsePoint < nPoint; iCoarsePoint ++) {
    if (nodes->GetBoundary(iCoarsePoint)) {
      for (iChildren = 0; iChildren < nodes->GetnChildren_CV(iCoarsePoint); iChildren ++) {
        iFinePoint = nodes->GetChildren_CV(iCoarsePoint, iChildren);
        for (iMarker = 0; iMarker < fine_grid->GetnMarker(); iMarker ++) {
          if ((fine_grid->nodes->GetVertex(iFinePoint, iMarker) != -1) && (nodes->GetVertex(iCoarsePoint, iMarker) == -1)) {
            iVertex = nVertex[iMarker];
            vertex[iMarker][iVertex] = new CVertex(iCoarsePoint, nDim);
            nodes->SetVertex(iCoarsePoint, iVertex, iMarker);

            /*--- Set the transformation to apply ---*/
            unsigned long ChildVertex = fine_grid->nodes->GetVertex(iFinePoint, iMarker);
            unsigned short RotationKind = fine_grid->vertex[iMarker][ChildVertex]->GetRotation_Type();
            vertex[iMarker][iVertex]->SetRotation_Type(RotationKind);
            nVertex[iMarker]++;
          }
        }
      }
    }
  }
}

void CMultiGridGeometry::MatchNearField(CConfig *config) {

  unsigned short iMarker;
  unsigned long iVertex, iPoint;
  int iProcessor = size;

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_KindBC(iMarker) == NEARFIELD_BOUNDARY) {
      for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++) {
        iPoint = vertex[iMarker][iVertex]->GetNode();
        if (nodes->GetDomain(iPoint)) {
          vertex[iMarker][iVertex]->SetDonorPoint(iPoint, nodes->GetGlobalIndex(iPoint), iVertex, iMarker, iProcessor);
        }
      }
    }
  }

}

void CMultiGridGeometry::MatchActuator_Disk(CConfig *config) {

  unsigned short iMarker;
  unsigned long iVertex, iPoint;
  int iProcessor = size;

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_KindBC(iMarker) == ACTDISK_INLET) ||
        (config->GetMarker_All_KindBC(iMarker) == ACTDISK_OUTLET)) {
      for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++) {
        iPoint = vertex[iMarker][iVertex]->GetNode();
        if (nodes->GetDomain(iPoint)) {
          vertex[iMarker][iVertex]->SetDonorPoint(iPoint, nodes->GetGlobalIndex(iPoint), iVertex, iMarker, iProcessor);
        }
      }
    }
  }

}

void CMultiGridGeometry::MatchPeriodic(CConfig *config, unsigned short val_periodic) {

  unsigned short iMarker, iPeriodic, nPeriodic;
  unsigned long iVertex, iPoint;
  int iProcessor = rank;

  /*--- Evaluate the number of periodic boundary conditions ---*/

  nPeriodic = config->GetnMarker_Periodic();

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_KindBC(iMarker) == PERIODIC_BOUNDARY) {
      iPeriodic = config->GetMarker_All_PerBound(iMarker);
      if ((iPeriodic == val_periodic) ||
          (iPeriodic == val_periodic + nPeriodic/2)) {
        for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++) {
          iPoint = vertex[iMarker][iVertex]->GetNode();
          if (nodes->GetDomain(iPoint)) {
            vertex[iMarker][iVertex]->SetDonorPoint(iPoint, nodes->GetGlobalIndex(iPoint), iVertex, iMarker, iProcessor);
          }
        }
      }
    }
  }

}

void CMultiGridGeometry::SetControlVolume(CConfig *config, CGeometry *fine_grid, unsigned short action) {

  SU2_OMP_MASTER {

  unsigned long iFinePoint, iFinePoint_Neighbor, iCoarsePoint, iEdge, iParent;
  long FineEdge, CoarseEdge;
  unsigned short iChildren, iNode, iDim;
  bool change_face_orientation;
  su2double Coarse_Volume, Area;

  /*--- Compute the area of the coarse volume ---*/
  for (iCoarsePoint = 0; iCoarsePoint < nPoint; iCoarsePoint ++) {
    nodes->SetVolume(iCoarsePoint, 0.0);
    Coarse_Volume = 0.0;
    for (iChildren = 0; iChildren < nodes->GetnChildren_CV(iCoarsePoint); iChildren ++) {
      iFinePoint = nodes->GetChildren_CV(iCoarsePoint, iChildren);
      Coarse_Volume += fine_grid->nodes->GetVolume(iFinePoint);
    }
    nodes->SetVolume(iCoarsePoint, Coarse_Volume);
  }

  /*--- Update or not the values of faces at the edge ---*/
  if (action != ALLOCATE) {
    edges->SetZeroValues();
  }

  for (iCoarsePoint = 0; iCoarsePoint < nPoint; iCoarsePoint ++)
    for (iChildren = 0; iChildren < nodes->GetnChildren_CV(iCoarsePoint); iChildren ++) {
      iFinePoint = nodes->GetChildren_CV(iCoarsePoint, iChildren);

      for (iNode = 0; iNode < fine_grid->nodes->GetnPoint(iFinePoint); iNode ++) {
        iFinePoint_Neighbor = fine_grid->nodes->GetPoint(iFinePoint, iNode);
        iParent = fine_grid->nodes->GetParent_CV(iFinePoint_Neighbor);
        if ((iParent != iCoarsePoint) && (iParent < iCoarsePoint)) {

          FineEdge = fine_grid->FindEdge(iFinePoint, iFinePoint_Neighbor);

          change_face_orientation = false;
          if (iFinePoint < iFinePoint_Neighbor) change_face_orientation = true;

          CoarseEdge = FindEdge(iParent, iCoarsePoint);

          const auto Normal = fine_grid->edges->GetNormal(FineEdge);

          if (change_face_orientation) {
            edges->SubNormal(CoarseEdge,Normal);
          }
          else {
            edges->AddNormal(CoarseEdge,Normal);
          }
        }
      }
    }

  /*--- Check if there is a normal with null area ---*/

  for (iEdge = 0; iEdge < nEdge; iEdge++) {
    const auto NormalFace = edges->GetNormal(iEdge);
    Area = 0.0; for (iDim = 0; iDim < nDim; iDim++) Area += NormalFace[iDim]*NormalFace[iDim];
    Area = sqrt(Area);
    if (Area == 0.0) {
      su2double DefaultNormal[3] = {EPS*EPS};
      edges->SetNormal(iEdge, DefaultNormal);
    }
  }

  } SU2_OMP_BARRIER
}

void CMultiGridGeometry::SetBoundControlVolume(CConfig *config, CGeometry *fine_grid, unsigned short action) {

  SU2_OMP_MASTER {

  unsigned long iCoarsePoint, iFinePoint, FineVertex, iVertex;
  unsigned short iMarker, iChildren, iDim;
  su2double *Normal, Area, *NormalFace = nullptr;

  Normal = new su2double [nDim];

  if (action != ALLOCATE) {
    for (iMarker = 0; iMarker < nMarker; iMarker++)
      for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++)
        vertex[iMarker][iVertex]->SetZeroValues();
  }

  for (iMarker = 0; iMarker < nMarker; iMarker ++)
    for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++) {
      iCoarsePoint = vertex[iMarker][iVertex]->GetNode();
      for (iChildren = 0; iChildren < nodes->GetnChildren_CV(iCoarsePoint); iChildren ++) {
        iFinePoint = nodes->GetChildren_CV(iCoarsePoint, iChildren);
        if (fine_grid->nodes->GetVertex(iFinePoint, iMarker)!=-1) {
          FineVertex = fine_grid->nodes->GetVertex(iFinePoint, iMarker);
          fine_grid->vertex[iMarker][FineVertex]->GetNormal(Normal);
          vertex[iMarker][iVertex]->AddNormal(Normal);
        }
      }
    }

  delete[] Normal;

  /*--- Check if there is a normal with null area ---*/
  for (iMarker = 0; iMarker < nMarker; iMarker ++)
    for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++) {
      NormalFace = vertex[iMarker][iVertex]->GetNormal();
      Area = 0.0; for (iDim = 0; iDim < nDim; iDim++) Area += NormalFace[iDim]*NormalFace[iDim];
      Area = sqrt(Area);
      if (Area == 0.0) for (iDim = 0; iDim < nDim; iDim++) NormalFace[iDim] = EPS*EPS;
    }

  } SU2_OMP_BARRIER
}

void CMultiGridGeometry::SetCoord(CGeometry *geometry) {

  SU2_OMP_FOR_STAT(roundUpDiv(nPoint, omp_get_max_threads()))
  for (auto Point_Coarse = 0ul; Point_Coarse < nPoint; Point_Coarse++) {
    auto Area_Parent = nodes->GetVolume(Point_Coarse);
    su2double Coordinates[3] = {0.0};
    for (auto iChildren = 0u; iChildren < nodes->GetnChildren_CV(Point_Coarse); iChildren++) {
      auto Point_Fine = nodes->GetChildren_CV(Point_Coarse, iChildren);
      auto Area_Children = geometry->nodes->GetVolume(Point_Fine);
      auto Coordinates_Fine = geometry->nodes->GetCoord(Point_Fine);
      for (auto iDim = 0u; iDim < nDim; iDim++)
        Coordinates[iDim] += Coordinates_Fine[iDim]*Area_Children/Area_Parent;
    }
    nodes->SetCoord(Point_Coarse, Coordinates);
  }
}

void CMultiGridGeometry::SetMultiGridWallHeatFlux(CGeometry *geometry, unsigned short val_marker){

  unsigned long Point_Fine, Point_Coarse, iVertex;
  unsigned short iChildren;
  long Vertex_Fine;
  su2double Area_Parent, Area_Children;
  su2double WallHeatFlux_Fine, WallHeatFlux_Coarse;
  bool isVertex;
  int numberVertexChildren;

  for(iVertex=0; iVertex < nVertex[val_marker]; iVertex++){
    Point_Coarse = vertex[val_marker][iVertex]->GetNode();
    if (nodes->GetDomain(Point_Coarse)){
      Area_Parent = 0.0;
      WallHeatFlux_Coarse = 0.0;
      numberVertexChildren = 0;
      /*--- Compute area parent by taking into account only volumes that are on the marker ---*/
      for(iChildren=0; iChildren < nodes->GetnChildren_CV(Point_Coarse); iChildren++){
        Point_Fine = nodes->GetChildren_CV(Point_Coarse, iChildren);
        isVertex = (nodes->GetDomain(Point_Fine) && geometry->nodes->GetVertex(Point_Fine, val_marker) != -1);
        if (isVertex){
          numberVertexChildren += 1;
          Area_Parent += geometry->nodes->GetVolume(Point_Fine);
        }
      }

      /*--- Loop again and propagate values to the coarser level ---*/
      for(iChildren=0; iChildren < nodes->GetnChildren_CV(Point_Coarse); iChildren++){
        Point_Fine = nodes->GetChildren_CV(Point_Coarse, iChildren);
        Vertex_Fine = geometry->nodes->GetVertex(Point_Fine, val_marker);
        isVertex = (nodes->GetDomain(Point_Fine) && Vertex_Fine != -1);
        if(isVertex){
          Area_Children = geometry->nodes->GetVolume(Point_Fine);
          //Get the customized BC values on fine level and compute the values at coarse level
          WallHeatFlux_Fine = geometry->GetCustomBoundaryHeatFlux(val_marker, Vertex_Fine);
          WallHeatFlux_Coarse += WallHeatFlux_Fine*Area_Children/Area_Parent;
        }

      }
      //Set the customized BC values at coarse level
      CustomBoundaryHeatFlux[val_marker][iVertex] = WallHeatFlux_Coarse;
    }
  }

}

void CMultiGridGeometry::SetMultiGridWallTemperature(CGeometry *geometry, unsigned short val_marker){

  unsigned long Point_Fine, Point_Coarse, iVertex;
  unsigned short iChildren;
  long Vertex_Fine;
  su2double Area_Parent, Area_Children;
  su2double WallTemperature_Fine, WallTemperature_Coarse;
  bool isVertex;
  int numberVertexChildren;

  for(iVertex=0; iVertex < nVertex[val_marker]; iVertex++){
    Point_Coarse = vertex[val_marker][iVertex]->GetNode();
    if (nodes->GetDomain(Point_Coarse)){
      Area_Parent = 0.0;
      WallTemperature_Coarse = 0.0;
      numberVertexChildren = 0;
      /*--- Compute area parent by taking into account only volumes that are on the marker ---*/
      for(iChildren=0; iChildren < nodes->GetnChildren_CV(Point_Coarse); iChildren++){
        Point_Fine = nodes->GetChildren_CV(Point_Coarse, iChildren);
        isVertex = (nodes->GetDomain(Point_Fine) && geometry->nodes->GetVertex(Point_Fine, val_marker) != -1);
        if (isVertex){
          numberVertexChildren += 1;
          Area_Parent += geometry->nodes->GetVolume(Point_Fine);
        }
      }

      /*--- Loop again and propagate values to the coarser level ---*/
      for(iChildren=0; iChildren < nodes->GetnChildren_CV(Point_Coarse); iChildren++){
        Point_Fine = nodes->GetChildren_CV(Point_Coarse, iChildren);
        Vertex_Fine = geometry->nodes->GetVertex(Point_Fine, val_marker);
        isVertex = (nodes->GetDomain(Point_Fine) && Vertex_Fine != -1);
        if(isVertex){
          Area_Children = geometry->nodes->GetVolume(Point_Fine);
          //Get the customized BC values on fine level and compute the values at coarse level
          WallTemperature_Fine = geometry->GetCustomBoundaryTemperature(val_marker, Vertex_Fine);
          WallTemperature_Coarse += WallTemperature_Fine*Area_Children/Area_Parent;
        }

      }
      //Set the customized BC values at coarse level
      CustomBoundaryTemperature[val_marker][iVertex] = WallTemperature_Coarse;
    }
  }

}

void CMultiGridGeometry::SetRestricted_GridVelocity(CGeometry *fine_mesh, CConfig *config) {

  /*--- Loop over all coarse mesh points. ---*/
  SU2_OMP_FOR_STAT(roundUpDiv(nPoint,omp_get_max_threads()))
  for (unsigned long Point_Coarse = 0; Point_Coarse < nPoint; Point_Coarse++) {
    su2double Area_Parent = nodes->GetVolume(Point_Coarse);

    /*--- Initialize coarse grid velocity to zero. ---*/
    su2double Grid_Vel[3] = {0.0, 0.0, 0.0};

    /*--- Loop over all of the children for this coarse CV and compute
     a grid velocity based on the values in the child CVs (fine mesh). ---*/
    for (unsigned short iChild = 0; iChild < nodes->GetnChildren_CV(Point_Coarse); iChild++) {
      unsigned long Point_Fine       = nodes->GetChildren_CV(Point_Coarse, iChild);
      su2double Area_Child           = fine_mesh->nodes->GetVolume(Point_Fine);
      const su2double* Grid_Vel_Fine = fine_mesh->nodes->GetGridVel(Point_Fine);
      for (unsigned short iDim = 0; iDim < nDim; iDim++)
        Grid_Vel[iDim] += Grid_Vel_Fine[iDim]*Area_Child/Area_Parent;
    }

    /*--- Set the grid velocity for this coarse node. ---*/
    for (unsigned short iDim = 0; iDim < nDim; iDim++)
      nodes->SetGridVel(Point_Coarse, iDim, Grid_Vel[iDim]);
  }
}


void CMultiGridGeometry::FindNormal_Neighbor(CConfig *config) {

  unsigned short iMarker, iDim;
  unsigned long iPoint, iVertex;

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {

    if (config->GetMarker_All_KindBC(iMarker) != SEND_RECEIVE &&
        config->GetMarker_All_KindBC(iMarker) != INTERFACE_BOUNDARY &&
        config->GetMarker_All_KindBC(iMarker) != NEARFIELD_BOUNDARY ) {

      for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++) {

        iPoint = vertex[iMarker][iVertex]->GetNode();

        /*--- If the node belong to the domain ---*/
        if (nodes->GetDomain(iPoint)) {

          /*--- Compute closest normal neighbor ---*/
          su2double cos_max, scalar_prod, norm_vect, norm_Normal, cos_alpha, diff_coord;
          unsigned long Point_Normal = 0, jPoint;
          unsigned short iNeigh;
          su2double *Normal = vertex[iMarker][iVertex]->GetNormal();
          cos_max = -1.0;
          for (iNeigh = 0; iNeigh < nodes->GetnPoint(iPoint); iNeigh++) {
            jPoint = nodes->GetPoint(iPoint, iNeigh);
            scalar_prod = 0.0; norm_vect = 0.0; norm_Normal = 0.0;
            for (iDim = 0; iDim < nDim; iDim++) {
              diff_coord = nodes->GetCoord(jPoint, iDim)-nodes->GetCoord(iPoint, iDim);
              scalar_prod += diff_coord*Normal[iDim];
              norm_vect += diff_coord*diff_coord;
              norm_Normal += Normal[iDim]*Normal[iDim];
            }
            norm_vect = sqrt(norm_vect);
            norm_Normal = sqrt(norm_Normal);
            cos_alpha = scalar_prod/(norm_vect*norm_Normal);

            /*--- Get maximum cosine (not minimum because normals are oriented inwards) ---*/
            if (cos_alpha >= cos_max) {
              Point_Normal = jPoint;
              cos_max = cos_alpha;
            }
          }
          vertex[iMarker][iVertex]->SetNormal_Neighbor(Point_Normal);
        }
      }
    }
  }
}

