/*!
 * \file CDriverOutput.hpp
 * \brief Headers of the main subroutines for screen and history output in multizone problems.
 * \author R. Sanchez, T. Albring
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

#include "../../../Common/include/mpi_structure.hpp"

#ifdef HAVE_CGNS
  #include "cgnslib.h"
#endif
#ifdef HAVE_TECIO
  #include "../../../externals/tecio/teciosrc/TECIO.h"
#endif
#include <fstream>
#include <cmath>

#include "COutput.hpp"
#include "../../../Common/include/CConfig.hpp"

using namespace std;

/*!
 * \class CDriverOutput
 * \brief Class for writing the multizone output.
 * \author R. Sanchez, T. Albring.
 */
class CMultizoneOutput final: public COutput {

protected:
  unsigned short nZone; //!< Number of zones

  string bgs_res_name; //!< Block-Gauss seidel residual name
  bool write_zone;     //!< Boolean indicating whether the individual zones write to screen

public:

  /*!
   * \brief Constructor of the class.
   */
  CMultizoneOutput(CConfig *driver_config, CConfig** config, unsigned short nDim);

  /*!
   * \brief Destructor of the class.
   */
  ~CMultizoneOutput(void) override;

  /*!
   * \brief Load the multizone history output field values
   * \param[in] output - Container holding the output instances per zone.
   * \param[in] config - Definition of the particular problem.
   */
  void LoadMultizoneHistoryData(COutput **output, CConfig **config) override;

  /*!
   * \brief Set the available multizone history output fields
   * \param[in] output - Container holding the output instances per zone.
   * \param[in] config - Definition of the particular problem per zone.
   */
  void SetMultizoneHistoryOutputFields(COutput **output, CConfig **config) override;

  /*!
   * \brief Determines if the history file output.
   * \param[in] config - Definition of the particular problem.
   */
  bool WriteHistoryFile_Output(CConfig *config) override;

  /*!
   * \brief Determines if the screen header should be written.
   * \param[in] config - Definition of the particular problem.
   */
  bool WriteScreen_Header(CConfig *config) override;

  /*!
   * \brief Determines if the screen header should be written.
   * \param[in] config - Definition of the particular problem.
   */
  bool WriteScreen_Output(CConfig *config) override;
};
