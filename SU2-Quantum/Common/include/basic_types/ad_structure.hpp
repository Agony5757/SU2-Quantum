/*!
 * \file ad_structure.hpp
 * \brief Main routines for the algorithmic differentiation (AD) structure.
 * \author T. Albring
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

#include "datatype_structure.hpp"

/*!
 * \namespace AD
 * \brief Contains routines for the reverse mode of AD.
 * In case there is no reverse type configured, they have no effect at all,
 * and so the real versions of the routined are after #else.
 */
namespace AD{
#ifndef CODI_REVERSE_TYPE
  /*!
   * \brief Start the recording of the operations and involved variables.
   * If called, the computational graph of all operations occuring after the call will be stored,
   * starting with the variables registered with RegisterInput.
   */
  inline void StartRecording() {}

  /*!
   * \brief Stops the recording of the operations and variables.
   */
  inline void StopRecording() {}

  /*!
   * \brief Check if the tape is active
   * \param[out] Boolean which determines whether the tape is active.
   */
  inline bool TapeActive() {return false;}

  /*!
   * \brief Prints out tape statistics.
   */
  inline void PrintStatistics() {}

  /*!
   * \brief Registers the variable as an input and saves internal data (indices). I.e. as a leaf of the computational graph.
   * \param[in] data - The variable to be registered as input.
   * \param[in] push_index - boolean whether we also want to push the index.
   */
  inline void RegisterInput(su2double &data, bool push_index = true) {}

  /*!
   * \brief Registers the variable as an output. I.e. as the root of the computational graph.
   * \param[in] data - The variable to be registered as output.
   */
  inline void RegisterOutput(su2double &data) {}

  /*!
   * \brief Sets the adjoint value at index to val
   * \param[in] index - Position in the adjoint vector.
   * \param[in] val - adjoint value to be set.
   */
  inline void SetDerivative(int index, const double val) {}

  /*!
   * \brief Extracts the adjoint value at index
   * \param[in] index - position in the adjoint vector where the derivative will be extracted.
   * \return Derivative value.
   */
  inline double GetDerivative(int index) {return 0.0;}

  /*!
   * \brief Clears the currently stored adjoints but keeps the computational graph.
   */
  inline void ClearAdjoints() {}

  /*!
   * \brief Computes the adjoints, i.e. the derivatives of the output with respect to the input variables.
   */
  inline void ComputeAdjoint() {}

  /*!
   * \brief Computes the adjoints, i.e. the derivatives of the output with respect to the input variables.
   * \param[in] enter - Position where we start evaluating the tape.
   * \param[in] leave - Position where we stop evaluating the tape.
   */
  inline void ComputeAdjoint(unsigned short enter, unsigned short leave) {}

  /*!
   * \brief Reset the tape structure to be ready for a new recording.
   */
  inline void Reset() {}

  /*!
   * \brief Reset the variable (set index to zero).
   * \param[in] data - the variable to be unregistered from the tape.
   */
  inline void ResetInput(su2double &data) {}

  /*!
   * \brief Sets the scalar inputs of a preaccumulation section.
   * \param[in] data - the scalar input variables.
   */
  template<class... Ts>
  inline void SetPreaccIn(Ts&&... data) {}

  /*!
   * \brief Sets the input variables of a preaccumulation section using a 1D array.
   * \param[in] data - the input 1D array.
   * \param[in] size - size of the array.
   */
  template<class T>
  inline void SetPreaccIn(const T& data, const int size) {}

  /*!
   * \brief Sets the input variables of a preaccumulation section using a 2D array.
   * \param[in] data - the input 2D array.
   * \param[in] size_x - size of the array in x dimension.
   * \param[in] size_y - size of the array in y dimension.
   */
  template<class T>
  inline void SetPreaccIn(const T& data, const int size_x, const int size_y) {}

  /*!
   * \brief Starts a new preaccumulation section and sets the input variables.
   *
   * The idea of preaccumulation is to store only the Jacobi matrix of a code section during
   * the taping process instead of all operations. This decreases the tape size and reduces runtime.
   *
   * Input/Output of the section are set with several calls to SetPreaccIn()/SetPreaccOut().
   *
   * Note: the call of this routine must be followed by a call of EndPreacc() and the end of the code section.
   */
  inline void StartPreacc() {}

  /*!
   * \brief Sets the scalar outputs of a preaccumulation section.
   * \param[in] data - the scalar output variables.
   */
  template<class... Ts>
  inline void SetPreaccOut(Ts&&... data) {}

  /*!
   * \brief Sets the output variables of a preaccumulation section using a 1D array.
   * \param[in] data - the output 1D array.
   */
  template<class T>
  inline void SetPreaccOut(T&& data, const int size) {}

  /*!
   * \brief Sets the input variables of a preaccumulation section using a 2D array.
   * \param[in] data - the output 1D array.
   */
  template<class T>
  inline void SetPreaccOut(T&& data, const int size_x, const int size_y) {}

  /*!
   * \brief Ends a preaccumulation section and computes the local Jacobi matrix
   * of a code section using the variables set with SetLocalInput(), SetLocalOutput() and pushes a statement
   * for each output variable to the AD tape.
   */
  inline void EndPreacc() {}

  /*!
   * \brief Initializes an externally differentiated function. Input and output variables are set with SetExtFuncIn/SetExtFuncOut
   * \param[in] storePrimalInput - Specifies whether the primal input values are stored for the reverse call of the external function.
   * \param[in] storePrimalOutput - Specifies whether the primal output values are stored for the reverse call of the external function.
   */
  inline void StartExtFunc(bool storePrimalInput, bool storePrimalOutput) {}

  /*!
   * \brief Sets the scalar input of a externally differentiated function.
   * \param[in] data - the scalar input variable.
   */
  inline void SetExtFuncIn(su2double &data) {}

  /*!
   * \brief Sets the input variables of a externally differentiated function using a 1D array.
   * \param[in] data - the input 1D array.
   * \param[in] size - number of rows.
   */
  template<class T>
  inline void SetExtFuncIn(const T& data, const int size) {}

  /*!
  * \brief  Sets the input variables of a externally differentiated function using a 2D array.
  * \param[in] data - the input 2D array.
  * \param[in] size_x - number of rows.
  * \param[in] size_y - number of columns.
  */
  template<class T>
  inline void SetExtFuncIn(const T& data, const int size_x, const int size_y) {}

  /*!
   * \brief Sets the scalar output of a externally differentiated function.
   * \param[in] data - the scalar output variable.
   */
  inline void SetExtFuncOut(su2double &data) {}

  /*!
   * \brief Sets the output variables of a externally differentiated function using a 1D array.
   * \param[in] data - the output 1D array.
   * \param[in] size - number of rows.
   */
  template<class T>
  inline void SetExtFuncOut(T&& data, const int size) {}

  /*!
  * \brief  Sets the output variables of a externally differentiated function using a 2D array.
  * \param[in] data - the output 2D array.
  * \param[in] size_x - number of rows.
  * \param[in] size_y - number of columns.
  */
  template<class T>
  inline void SetExtFuncOut(T&& data, const int size_x, const int size_y) {}

  /*!
   * \brief Ends an external function section by deleting the structures.
   */
  inline void EndExtFunc() {}

  /*!
   * \brief Evaluates and saves gradient data from a variable.
   * \param[in] data - variable whose gradient information will be extracted.
   * \param[in] index - where obtained gradient information will be stored.
   */
  inline void SetIndex(int &index, const su2double &data) {}

  /*!
   * \brief Pushes back the current tape position to the tape position's vector.
   */
  inline void Push_TapePosition() {}

  /*!
   * \brief Start a passive region, i.e. stop recording.
   * \return True is tape was active.
   */
  inline bool BeginPassive() { return false; }

  /*!
   * \brief End a passive region, i.e. start recording if we were recording before.
   * \param[in] wasActive - Whether we were recording before entering the passive region.
   */
  inline void EndPassive(bool wasActive) {}

#else
  using CheckpointHandler = codi::DataStore;

  using Tape = su2double::TapeType;

  using ExtFuncHelper = codi::ExternalFunctionHelper<su2double>;

  extern ExtFuncHelper* FuncHelper;

  /*--- Stores the indices of the input variables (they might be overwritten) ---*/

  extern std::vector<su2double::GradientData> inputValues;

  /*--- Current position inside the adjoint vector ---*/

  extern int adjointVectorPosition;

  /*--- Reference to the tape ---*/

  extern su2double::TapeType& globalTape;

  extern bool Status;

  extern bool PreaccActive;

  extern bool PreaccEnabled;

  extern su2double::TapeType::Position StartPosition, EndPosition;

  extern std::vector<su2double::TapeType::Position> TapePositions;

  extern std::vector<su2double::GradientData> localInputValues;

  extern std::vector<su2double*> localOutputValues;

  extern codi::PreaccumulationHelper<su2double> PreaccHelper;

  FORCEINLINE void RegisterInput(su2double &data, bool push_index = true) {
    AD::globalTape.registerInput(data);
    if (push_index) {
      inputValues.push_back(data.getGradientData());
    }
  }

  FORCEINLINE void RegisterOutput(su2double& data) {AD::globalTape.registerOutput(data);}

  FORCEINLINE void ResetInput(su2double &data) {data.getGradientData() = su2double::GradientData();}

  FORCEINLINE void StartRecording() {AD::globalTape.setActive();}

  FORCEINLINE void StopRecording() {AD::globalTape.setPassive();}

  FORCEINLINE bool TapeActive() { return AD::globalTape.isActive(); }

  FORCEINLINE void PrintStatistics() {AD::globalTape.printStatistics();}

  FORCEINLINE void ClearAdjoints() {AD::globalTape.clearAdjoints(); }

  FORCEINLINE void ComputeAdjoint() {AD::globalTape.evaluate(); adjointVectorPosition = 0;}

  FORCEINLINE void ComputeAdjoint(unsigned short enter, unsigned short leave) {
    AD::globalTape.evaluate(TapePositions[enter], TapePositions[leave]);
    if (leave == 0)
      adjointVectorPosition = 0;
  }

  FORCEINLINE void Reset() {
    globalTape.reset();
    if (inputValues.size() != 0) {
      adjointVectorPosition = 0;
      inputValues.clear();
    }
    if (TapePositions.size() != 0) {
      TapePositions.clear();
    }
  }

  FORCEINLINE void SetIndex(int &index, const su2double &data) {
    index = data.getGradientData();
  }

  FORCEINLINE void SetDerivative(int index, const double val) {
    AD::globalTape.setGradient(index, val);
  }

  FORCEINLINE double GetDerivative(int index) {
    return AD::globalTape.getGradient(index);
  }

  /*--- Base case for parameter pack expansion. ---*/
  FORCEINLINE void SetPreaccIn() {}

  template<class T, class... Ts,
           typename std::enable_if<std::is_same<T,su2double>::value,bool>::type = 0>
  FORCEINLINE void SetPreaccIn(const T& data, Ts&&... moreData) {
    if (!PreaccActive) return;
    if (data.isActive())
      PreaccHelper.addInput(data);
    SetPreaccIn(moreData...);
  }

  template<class T>
  FORCEINLINE void SetPreaccIn(const T& data, const int size) {
    if (PreaccActive) {
      for (int i = 0; i < size; i++) {
        if (data[i].isActive()) {
          PreaccHelper.addInput(data[i]);
        }
      }
    }
  }

  template<class T>
  FORCEINLINE void SetPreaccIn(const T& data, const int size_x, const int size_y) {
    if (!PreaccActive) return;
    for (int i = 0; i < size_x; i++) {
      for (int j = 0; j < size_y; j++) {
        if (data[i][j].isActive()) {
          PreaccHelper.addInput(data[i][j]);
        }
      }
    }
  }

  FORCEINLINE void StartPreacc() {
    if (globalTape.isActive() && PreaccEnabled) {
      PreaccHelper.start();
      PreaccActive = true;
    }
  }

  /*--- Base case for parameter pack expansion. ---*/
  FORCEINLINE void SetPreaccOut() {}

  template<class T, class... Ts,
           typename std::enable_if<std::is_same<T,su2double>::value,bool>::type = 0>
  FORCEINLINE void SetPreaccOut(T& data, Ts&&... moreData) {
    if (!PreaccActive) return;
    if (data.isActive())
      PreaccHelper.addOutput(data);
    SetPreaccOut(moreData...);
  }

  template<class T>
  FORCEINLINE void SetPreaccOut(T&& data, const int size) {
    if (PreaccActive) {
      for (int i = 0; i < size; i++) {
        if (data[i].isActive()) {
          PreaccHelper.addOutput(data[i]);
        }
      }
    }
  }

  template<class T>
  FORCEINLINE void SetPreaccOut(T&& data, const int size_x, const int size_y) {
    if (!PreaccActive) return;
    for (int i = 0; i < size_x; i++) {
      for (int j = 0; j < size_y; j++) {
        if (data[i][j].isActive()) {
          PreaccHelper.addOutput(data[i][j]);
        }
      }
    }
  }

  FORCEINLINE void Push_TapePosition() {
    TapePositions.push_back(AD::globalTape.getPosition());
  }

  FORCEINLINE void EndPreacc(){
    if (PreaccActive) {
      PreaccHelper.finish(false);
    }
  }

  FORCEINLINE void StartExtFunc(bool storePrimalInput, bool storePrimalOutput){
    FuncHelper = new ExtFuncHelper(true);
    if (!storePrimalInput){
      FuncHelper->disableInputPrimalStore();
    }
    if (!storePrimalOutput){
      FuncHelper->disableOutputPrimalStore();
    }
  }

  FORCEINLINE void SetExtFuncIn(const su2double &data) {
    FuncHelper->addInput(data);
  }

  template<class T>
  FORCEINLINE void SetExtFuncIn(const T& data, const int size) {
    for (int i = 0; i < size; i++) {
      FuncHelper->addInput(data[i]);
    }
  }

  template<class T>
  FORCEINLINE void SetExtFuncIn(const T& data, const int size_x, const int size_y) {
    for (int i = 0; i < size_x; i++) {
      for (int j = 0; j < size_y; j++) {
        FuncHelper->addInput(data[i][j]);
      }
    }
  }

  FORCEINLINE void SetExtFuncOut(su2double& data) {
    if (globalTape.isActive()) {
      FuncHelper->addOutput(data);
    }
  }

  template<class T>
  FORCEINLINE void SetExtFuncOut(T&& data, const int size) {
    for (int i = 0; i < size; i++) {
      if (globalTape.isActive()) {
        FuncHelper->addOutput(data[i]);
      }
    }
  }

  template<class T>
  FORCEINLINE void SetExtFuncOut(T&& data, const int size_x, const int size_y) {
    for (int i = 0; i < size_x; i++) {
      for (int j = 0; j < size_y; j++) {
        if (globalTape.isActive()) {
          FuncHelper->addOutput(data[i][j]);
        }
      }
    }
  }

  FORCEINLINE void delete_handler(void *handler) {
    CheckpointHandler *checkpoint = static_cast<CheckpointHandler*>(handler);
    checkpoint->clear();
  }

  FORCEINLINE void EndExtFunc() { delete FuncHelper; }

  FORCEINLINE bool BeginPassive() {
    if(AD::globalTape.isActive()) {
      StopRecording();
      return true;
    }
    return false;
  }

  FORCEINLINE void EndPassive(bool wasActive) { if(wasActive) StartRecording(); }

#endif // CODI_REVERSE_TYPE

} // namespace AD


/*--- If we compile under OSX we have to overload some of the operators for
 *   complex numbers to avoid the use of the standard operators
 *  (they use a lot of functions that are only defined for doubles) ---*/

#ifdef __APPLE__

namespace std{

  template<>
  inline su2double abs(const complex<su2double>& x){

    return sqrt(x.real()*x.real() + x.imag()*x.imag());

  }

  template<>
  inline complex<su2double> operator/(const complex<su2double>& x,
                                      const complex<su2double>& y){

    su2double d    = (y.real()*y.real() + y.imag()*y.imag());
    su2double real = (x.real()*y.real() + x.imag()*y.imag())/d;
    su2double imag = (x.imag()*y.real() - x.real()*y.imag())/d;

    return complex<su2double>(real, imag);

  }

  template<>
  inline complex<su2double> operator*(const complex<su2double>& x,
                                      const complex<su2double>& y){

    su2double real = (x.real()*y.real() - x.imag()*y.imag());
    su2double imag = (x.imag()*y.real() + x.real()*y.imag());

    return complex<su2double>(real, imag);

  }
}
#endif
