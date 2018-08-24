//! \file gnds.h
//! Classes and funcitons related to the GNDS format

#ifndef OPENMC_GNDS_H
#define OPENMC_GNDS_H

#include <vector>

#include "hdf5.h"

#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/function.h"

namespace openmc {

//! Construct necessary Function1D object from HDF data
//! \param[in] obj_id HDF object containing Function1D data
//! \return Unique pointer to a Function1D object
UPtrFunction get_function1D(hid_t obj_id, const char* name);

//==============================================================================
//! One-dimensional interpolable function
//==============================================================================

class XYs1D : public Function1D {
public:
  XYs1D() = default;

  //! Construct function from HDF5 data
  //! \param[in] dset Dataset containing tabulated data
  explicit XYs1D(hid_t dset);
  
  //! Evaluate the tabulated function
  //! \param[in] x independent variable
  //! \return Function evaluated at x
  double operator()(double x) const;
private:
  Interpolation int_;
  std::size_t n_pairs_;
  std::vector<double> x_;
  std::vector<double> y_;
};

//==============================================================================
//! Vector of unique pointers to Function1D objects that span a domain
//==============================================================================

class Regions1D : public Function1D {
public:
  Regions1D() = default;

  //! Construct l from HDF5 data
  //! \param[in] group Group containing Function1D datasets
  explicit Regions1D(hid_t group);

  //! Evaluate a value using the contained functions
  //! \param[in] x independent variable
  //! \return Function evaluated at x
  double operator()(double x) const;
private:
  std::size_t n_regions_ {0};
  std::vector<UPtrFunction> regions_;
  std::vector<double> domainbreaks_;
};

} //namespace openmc

#endif // OPENMC_GNDS_H



