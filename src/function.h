#ifndef OPENMC_FUNCTION_H
#define OPENMC_FUNCTION_H

#include <vector>

#include "hdf5.h"

namespace openmc {

//==============================================================================
//! Abstract one-dimensional function
//==============================================================================

class Function1D {
public:
  virtual double operator()(double x) const = 0;
};

//==============================================================================
//! One-dimensional function expressed as a polynomial
//==============================================================================

class Polynomial : public Function1D {
public:
  //! Construct polynomial from HDF5 data
  //! \param[in] dset Dataset containing coefficients
  explicit Polynomial(hid_t dset);

  //! Evaluate the polynomials
  //! \param[in] x independent variable
  //! \return Polynomial evaluated at x
  double operator()(double x) const;
private:
  std::vector<double> coef_; //!< Polynomial coefficients
};

using UPtrFunction = std::unique_ptr<Function1D>;

} //namespace openmc

#endif // OPENMC_FUNCTION_H
