﻿/*!
 * \file CMirror.hpp
 * \brief Mirror interpolation for the conservative (work-wise) approach in FSI problems.
 * \author P. Gomes
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

#include "CInterpolator.hpp"

/*!
 * \brief Mirror interpolation, transpose interpolation matrix of opposing mesh.
 * \note Requires that the opposing mesh has already run interpolation (jZone > iZone), otherwise throws.
 */
class CMirror final : public CInterpolator {
public:
  /*!
   * \brief Constructor of the class.
   * \note Data is set in geometry[targetZone].
   * \param[in] geometry_container
   * \param[in] config - config container
   * \param[in] iZone - First zone
   * \param[in] jZone - Second zone
   */
  CMirror(CGeometry ****geometry_container, const CConfig* const* config, unsigned int iZone, unsigned int jZone);

  /*!
   * \brief Set up transfer matrix defining relation between two meshes
   * \param[in] config - Definition of the particular problem.
   */
  void SetTransferCoeff(const CConfig* const* config) override;

};
