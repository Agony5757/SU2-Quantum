/*!
 * \file output_heat.cpp
 * \brief Main subroutines for the heat solver output
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


#include "../../include/output/CHeatOutput.hpp"
#include "../../../Common/include/geometry/CGeometry.hpp"
#include "../../include/solvers/CSolver.hpp"

CHeatOutput::CHeatOutput(CConfig *config, unsigned short nDim) : COutput(config, nDim, false) {

  multiZone = config->GetMultizone_Problem();

  /*--- Set the default history fields if nothing is set in the config file ---*/

  if (nRequestedHistoryFields == 0){
    requestedHistoryFields.emplace_back("ITER");
    requestedHistoryFields.emplace_back("RMS_RES");
    nRequestedHistoryFields = requestedHistoryFields.size();
  }
  if (nRequestedScreenFields == 0){
    requestedScreenFields.emplace_back("OUTER_ITER");
    requestedScreenFields.emplace_back("INNER_ITER");
    requestedScreenFields.emplace_back("RMS_TEMPERATURE");
    nRequestedScreenFields = requestedScreenFields.size();
  }
  if (nRequestedVolumeFields == 0){
    requestedVolumeFields.emplace_back("COORDINATES");
    requestedVolumeFields.emplace_back("SOLUTION");
    nRequestedVolumeFields = requestedVolumeFields.size();
  }

  stringstream ss;
  ss << "Zone " << config->GetiZone() << " (Solid Heat)";
  multiZoneHeaderString = ss.str();

  /*--- Set the volume filename --- */

  volumeFilename = config->GetVolume_FileName();

  /*--- Set the surface filename --- */

  surfaceFilename = config->GetSurfCoeff_FileName();

  /*--- Set the restart filename --- */

  restartFilename = config->GetRestart_FileName();

  /*--- Set the default convergence field --- */

  if (convFields.empty() ) convFields.emplace_back("RMS_TEMPERATURE");


}

CHeatOutput::~CHeatOutput(void) {}

void CHeatOutput::LoadHistoryData(CConfig *config, CGeometry *geometry, CSolver **solver) {

  CSolver* heat_solver = solver[HEAT_SOL];

  SetHistoryOutputValue("TOTAL_HEATFLUX", heat_solver->GetTotal_HeatFlux());
  SetHistoryOutputValue("HEATFLUX_MAX", heat_solver->GetTotal_MaxHeatFlux());
  SetHistoryOutputValue("AVG_TEMPERATURE", heat_solver->GetTotal_AvgTemperature());
  SetHistoryOutputValue("RMS_TEMPERATURE", log10(heat_solver->GetRes_RMS(0)));
  SetHistoryOutputValue("MAX_TEMPERATURE", log10(heat_solver->GetRes_Max(0)));
  if (multiZone)
    SetHistoryOutputValue("BGS_TEMPERATURE", log10(heat_solver->GetRes_BGS(0)));

  SetHistoryOutputValue("LINSOL_ITER", heat_solver->GetIterLinSolver());
  SetHistoryOutputValue("LINSOL_RESIDUAL", log10(heat_solver->GetResLinSolver()));
  SetHistoryOutputValue("CFL_NUMBER", config->GetCFL(MESH_0));

}


void CHeatOutput::SetHistoryOutputFields(CConfig *config){

  AddHistoryOutput("LINSOL_ITER", "LinSolIter", ScreenOutputFormat::INTEGER, "LINSOL", "Number of iterations of the linear solver.");
  AddHistoryOutput("LINSOL_RESIDUAL", "LinSolRes", ScreenOutputFormat::FIXED, "LINSOL", "Residual of the linear solver.");

  AddHistoryOutput("RMS_TEMPERATURE", "rms[T]", ScreenOutputFormat::FIXED, "RMS_RES", "Root mean square residual of the temperature", HistoryFieldType::RESIDUAL);
  AddHistoryOutput("MAX_TEMPERATURE", "max[T]", ScreenOutputFormat::FIXED, "MAX_RES", "Maximum residual of the temperature", HistoryFieldType::RESIDUAL);
  AddHistoryOutput("BGS_TEMPERATURE", "bgs[T]", ScreenOutputFormat::FIXED, "BGS_RES", "Block-Gauss seidel residual of the temperature", HistoryFieldType::RESIDUAL);

  AddHistoryOutput("TOTAL_HEATFLUX", "HF", ScreenOutputFormat::SCIENTIFIC, "HEAT", "Total heatflux on all surfaces defined in MARKER_MONITORING", HistoryFieldType::COEFFICIENT);
  AddHistoryOutput("HEATFLUX_MAX", "MaxHF", ScreenOutputFormat::SCIENTIFIC, "HEAT", "Total maximal heatflux on all surfaces defined in MARKER_MONITORING", HistoryFieldType::COEFFICIENT);
  AddHistoryOutput("AVG_TEMPERATURE", "AvgTemp", ScreenOutputFormat::SCIENTIFIC, "HEAT", "Total average temperature on all surfaces defined in MARKER_MONITORING", HistoryFieldType::COEFFICIENT);
  AddHistoryOutput("CFL_NUMBER", "CFL number", ScreenOutputFormat::SCIENTIFIC, "CFL_NUMBER", "Current value of the CFL number");

}


void CHeatOutput::SetVolumeOutputFields(CConfig *config){

  // Grid coordinates
  AddVolumeOutput("COORD-X", "x", "COORDINATES", "x-component of the coordinate vector");
  AddVolumeOutput("COORD-Y", "y", "COORDINATES", "y-component of the coordinate vector");
  if (nDim == 3)
    AddVolumeOutput("COORD-Z", "z", "COORDINATES","z-component of the coordinate vector");

  // SOLUTION
  AddVolumeOutput("TEMPERATURE", "Temperature", "SOLUTION", "Temperature");

  // Primitives
  AddVolumeOutput("HEAT_FLUX", "Heat_Flux", "PRIMITIVE", "Heatflux");

  // Residuals
  AddVolumeOutput("RES_TEMPERATURE", "Residual_Temperature", "RESIDUAL", "Residual of the temperature");

}


void CHeatOutput::LoadVolumeData(CConfig *config, CGeometry *geometry, CSolver **solver, unsigned long iPoint){

  CVariable* Node_Heat = solver[HEAT_SOL]->GetNodes();
  CPoint*    Node_Geo  = geometry->nodes;

  // Grid coordinates
  SetVolumeOutputValue("COORD-X", iPoint,  Node_Geo->GetCoord(iPoint, 0));
  SetVolumeOutputValue("COORD-Y", iPoint,  Node_Geo->GetCoord(iPoint, 1));
  if (nDim == 3)
    SetVolumeOutputValue("COORD-Z", iPoint, Node_Geo->GetCoord(iPoint, 2));

  // SOLUTION
  SetVolumeOutputValue("TEMPERATURE", iPoint, Node_Heat->GetSolution(iPoint, 0));

  // Residuals
  SetVolumeOutputValue("RES_TEMPERATURE", iPoint, solver[HEAT_SOL]->LinSysRes(iPoint, 0));

}

void CHeatOutput::LoadSurfaceData(CConfig *config, CGeometry *geometry, CSolver **solver, unsigned long iPoint, unsigned short iMarker, unsigned long iVertex){

  /* Heat flux value at each surface grid node. */
  SetVolumeOutputValue("HEAT_FLUX", iPoint, solver[HEAT_SOL]->GetHeatFlux(iMarker, iVertex));

}

