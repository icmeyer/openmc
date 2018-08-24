//! \file endf.h
//! Classes and functions related to the ENDF-6 format

#ifndef OPENMC_ENDF_H
#define OPENMC_ENDF_H

#include <vector>

#include "hdf5.h"

#include "openmc/constants.h"
#include "openmc/function.h"

namespace openmc {

//! Convert integer representing interpolation law to enum
//! \param[in] i Intereger (e.g. 1=histogram, 2=lin-lin)
//! \return Corresponding enum value
Interpolation int2interp(int i);

//! Determine whether MT number corresponds to a fission reaction
//! \param[in] MT ENDF MT value
//! \return Whether corresponding reaction is a fission reaction
bool is_fission(int MT);

//==============================================================================
//! One-dimensional interpolable function
//==============================================================================

class Tabulated1D : public Function1D {
public:
  Tabulated1D() = default;

  //! Construct function from HDF5 data
  //! \param[in] dset Dataset containing tabulated data
  explicit Tabulated1D(hid_t dset);

  //! Evaluate the tabulated function
  //! \param[in] x independent variable
  //! \return Function evaluated at x
  double operator()(double x) const;
private:
  std::size_t n_regions_ {0}; //!< number of interpolation regions
  std::vector<int> nbt_; //!< values separating interpolation regions
  std::vector<Interpolation> int_; //!< interpolation schemes
  std::size_t n_pairs_; //!< number of (x,y) pairs
  std::vector<double> x_; //!< values of abscissa
  std::vector<double> y_; //!< values of ordinate
};

//==============================================================================
//! Coherent elastic scattering data from a crystalline material
//==============================================================================

class CoherentElasticXS : public Function1D {
  explicit CoherentElasticXS(hid_t dset);
  double operator()(double E) const;
private:
  std::vector<double> bragg_edges_; //!< Bragg edges in [eV]
  std::vector<double> factors_;     //!< Partial sums of structure factors [eV-b]
};


} // namespace openmc

#endif // OPENMC_ENDF_H
