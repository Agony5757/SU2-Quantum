/*!
 * \file CPrimalGridBoundFEM.hpp
 * \brief Headers of the main subroutines for storing the primal grid structure.
 *        The subroutines and functions are in the <i>CPrimalGridBoundFEM.cpp</i> file.
 * \author F. Palacios
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

#pragma once

#include "CPrimalGrid.hpp"

/*!
 * \class CPrimalGridBoundFEM
 * \brief Class to define primal grid boundary element for the FEM solver.
 * \version 7.0.6 "Blackbird"
 */
class CPrimalGridBoundFEM final: public CPrimalGrid {
private:
  unsigned short VTK_Type;     /*!< \brief Element type using the VTK convention. */
  unsigned short nPolyGrid;    /*!< \brief Polynomial degree for the geometry of the element. */
  unsigned short nDOFsGrid;    /*!< \brief Number of DOFs for the geometry of the element. */

  unsigned long boundElemIDGlobal;    /*!< \brief Global boundary element ID of this element. */
  bool JacobianConsideredConstant;    /*!< \brief Whether or not the Jacobian of the transformation to
                                                  is (almost) constant. */

  vector<unsigned long> donorElementsWallFunctions; /*!< \brief The global ID's of the donor elements
                                                                for the wall function treatment. */
public:

  /*!
   * \brief Constructor of the class.
   */
  CPrimalGridBoundFEM(void);

  /*!
   * \brief Constructor using data to initialize the boundary element.
   * \param[in] val_elemGlobalID    - Global boundary element ID of this element.
   * \param[in] val_domainElementID - Global ID of the corresponding domain element.
   * \param[in] val_VTK_Type        - VTK type to indicate the element type
   * \param[in] val_nPolyGrid       - Polynomial degree to describe the geometry of the element.
   * \param[in] val_nDOFsGrid       - Number of DOFs used to describe the geometry of the element.
   * \param[in] val_nodes           - Vector, which contains the global node IDs of the element.
   */
  CPrimalGridBoundFEM(unsigned long         val_elemGlobalID,
                      unsigned long         val_domainElementID,
                      unsigned short        val_VTK_Type,
                      unsigned short        val_nPolyGrid,
                      unsigned short        val_nDOFsGrid,
                      vector<unsigned long> &val_nodes);

  /*!
   * \brief Destructor of the class.
   */
  ~CPrimalGridBoundFEM(void) override;

  /*!
   * \brief Get the node shared by the element
   * \param[in] val_node - Local (to the element) index of the node.
   * \return Global index of the node.
   */
  inline unsigned long GetNode(unsigned short val_node) override { return Nodes[val_node]; }

  /*!
   * \brief Get the number of nodes that composes a face of an element.
   * \param[in] val_face - Local index of the face.
   * \return Number of nodes that composes a face of an element.
   */
  inline unsigned short GetnNodesFace(unsigned short val_face) override { return -1; }

  /*!
   * \brief Get the face index of an element.
   * \param[in] val_face - Local index of the face.
   * \param[in] val_index - Local (to the face) index of the nodes that compose the face.
   * \return Local (to the element) index of the nodes that compose the face.
   */
  inline unsigned short GetFaces(unsigned short val_face, unsigned short val_index) override { return -1; }

  /*!
   * \brief Get the local index of the neighbors to a node (given the local index).
   * \param[in] val_node - Local (to the element) index of a node.
   * \param[in] val_index - Local (to the neighbor nodes of val_node) index of the nodes that are neighbor to val_node.
   * \return Local (to the element) index of the nodes that are neighbor to val_node.
   */
  inline unsigned short GetNeighbor_Nodes(unsigned short val_node, unsigned short val_index) override { return -1; }

  /*!
   * \brief Get the number of nodes of an element.
   * \return Number of nodes that composes an element.
   */
  inline unsigned short GetnNodes(void) override { return nDOFsGrid; }

  /*!
   * \brief Get the number of faces of an element.
   * \return Number of faces of an element.
   */
  inline unsigned short GetnFaces(void) override { return -1; }

  /*!
   * \brief Get the number of neighbors nodes of a node.
   * \param[in] val_node - Local (to the element) index of a node.
   * \return Number if neighbors of a node val_node.
   */
  inline unsigned short GetnNeighbor_Nodes(unsigned short val_node) override { return -1; }

  /*!
   * \brief Change the orientation of an element.
   */
  inline void Change_Orientation(void) override {}

  /*!
   * \brief Get the number of element that are neighbor to this element.
   * \return Number of neighbor elements.
   */
  inline unsigned short GetnNeighbor_Elements(void) override { return -1; }

  /*!
   * \brief Get the Maximum number of nodes of a face of an element.
   * \return Maximum number of nodes of a face of an element.
   */
  inline unsigned short GetMaxNodesFace(void) override { return -1; }

  /*!
   * \brief Get the type of the element using VTK nomenclature.
   * \return Type of the element using VTK nomenclature.
   */
  inline unsigned short GetVTK_Type(void) override { return VTK_Type; }

  /*!
   * \brief Get the polynomial degree of the grid for this element.
   * \return The polynomial degree of the grid.
   */
  inline unsigned short GetNPolyGrid(void) override { return nPolyGrid; }

  /*!
   * \brief Function to make available the number of DOFs of the grid in the element.
   * \return The number of DOFs of the grid in the element.
   */
  inline unsigned short GetNDOFsGrid(void) override { return nDOFsGrid; }

  /*!
   * \brief Get the corner points of this boundary element.
   * \param[out] nFaces         - Number of faces of this element, i.e. 1.
   * \param[out] nPointsPerFace - Number of corner points for each of the faces.
   * \param[out] faceConn       - Global IDs of the corner points of the faces.
   */
  void GetCornerPointsAllFaces(unsigned short &nFaces,
                               unsigned short nPointsPerFace[],
                               unsigned long  faceConn[6][4]) override;

  /*!
   * \brief Static member function to get the local the corner points of all the face
           of this element.
   * \param[in]  elementType    - Type of the element using the VTK convention.
   * \param[in]  nPoly          - Polynomial degree of the element.
   * \param[in]  nDOFs          - Number of DOFs of the element.
   * \param[out] nPointsPerFace - Number of corner points of the face.
   * \param[out] faceConn       - Global IDs of the corner points of the face.
   */
  static void GetLocalCornerPointsFace(unsigned short elementType,
                                       unsigned short nPoly,
                                       unsigned short nDOFs,
                                       unsigned short &nPointsPerFace,
                                       unsigned long  faceConn[]);

  /*!
   * \brief Make available the global ID of this element.
   * \return The global ID of this element.
   */
  inline unsigned long GetGlobalElemID(void) override { return boundElemIDGlobal; }

  /*!
   * \brief Function to get whether or not the Jacobian is considered constant.
   * \return True if the Jacobian is (almost) constant and false otherwise.
   */
  inline bool GetJacobianConsideredConstant(void) override {return JacobianConsideredConstant;}

  /*!
   * \brief Function to set the value of JacobianConsideredConstant.
   * \param[in] val_JacobianConsideredConstant - The value to be set for JacobianConsideredConstant.
   */
  inline void SetJacobianConsideredConstant(bool val_JacobianConsideredConstant) override {JacobianConsideredConstant = val_JacobianConsideredConstant;}

  /*!
   * \brief Add the given donor ID to the donor elements for the wall function treatment.
   * \param[in] donorElement - Element to be added to donor elements.
   */
  inline void AddDonorWallFunctions(const unsigned long donorElement) override {donorElementsWallFunctions.push_back(donorElement);}

  /*!
   * \brief Make available the number of donor elements for the wall function treatment.
   * \return The number of donor elements.
   */
  inline unsigned short GetNDonorsWallFunctions(void) override {return donorElementsWallFunctions.size();}

  /*!
   * \brief Make available the pointer to the vector for the donor elements
           for the wall function treatment.
   * \return The pointer to the data of donorElementsWallFunctions.
   */
  inline unsigned long *GetDonorsWallFunctions(void) override {return donorElementsWallFunctions.data();}

  /*!
   * \brief Set the global ID's of the donor elements for the wall function treatment.
   * \param[in] donorElements - Vector, which contain the donor elements.
   */
  inline void SetDonorsWallFunctions(const vector<unsigned long> &donorElements) override {donorElementsWallFunctions = donorElements;}

  /*!
   * \brief Function to remove the multiple donors for the wall function treatment.
   */
  void RemoveMultipleDonorsWallFunctions(void) override;
};
