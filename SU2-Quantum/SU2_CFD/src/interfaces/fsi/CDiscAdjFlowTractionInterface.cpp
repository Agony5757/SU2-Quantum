/*!
 * \file CDiscAdjFlowTractionInterface.cpp
 * \brief Declaration and inlines of the class to transfer flow tractions
 *        from a fluid zone into a structural zone in a discrete adjoint simulation.
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

#include "../../../include/interfaces/fsi/CDiscAdjFlowTractionInterface.hpp"


CDiscAdjFlowTractionInterface::CDiscAdjFlowTractionInterface(unsigned short val_nVar, unsigned short val_nConst,
                                                             CConfig *config, bool integrate_tractions_) :
  CFlowTractionInterface(val_nVar, val_nConst, config, integrate_tractions_) {

}

void CDiscAdjFlowTractionInterface::GetPhysical_Constants(CSolver *flow_solution, CSolver *struct_solution,
                                                          CGeometry *flow_geometry, CGeometry *struct_geometry,
                                                          CConfig *flow_config, CConfig *struct_config){

  /*--- We have to clear the traction before applying it, because we are "adding" to node and not "setting" ---*/

  struct_solution->GetNodes()->Clear_FlowTraction();

  Preprocess(flow_config);

  /*--- No ramp applied ---*/
  Physical_Constants[1] = 1.0;
}
