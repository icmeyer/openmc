#include "endf.h"

#include <algorithm> // for copy
#include <cmath>     // for log, exp
#include <iterator>  // for back_inserter

#include "constants.h"
#include "hdf5_interface.h"
#include "search.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

namespace openmc {

Interpolation int2interp(int i)
{
  switch (i) {
  case 1:
    return Interpolation::histogram;
  case 2:
    return Interpolation::lin_lin;
  case 3:
    return Interpolation::lin_log;
  case 4:
    return Interpolation::log_lin;
  case 5:
    return Interpolation::log_log;
  }
}

//==============================================================================
// Polynomial implementation
//==============================================================================

Polynomial::Polynomial(hid_t dset)
{
  // Read coefficients into a vector
  read_dataset(dset, coef_);
}

double Polynomial::operator()(double x) const
{
  // Use Horner's rule to evaluate polynomial. Note that coefficients are
  // ordered in increasing powers of x.
  double y = 0.0;
  for (auto c = coef_.crbegin(); c != coef_.crend(); ++c) {
    y = y*x + *c;
  }
  return y;
}

//==============================================================================
// Tabulated1D implementation
//==============================================================================

Tabulated1D::Tabulated1D(hid_t dset)
{
  read_attribute(dset, "breakpoints", nbt_);
  n_regions_ = nbt_.size();

  // Change 1-indexing to 0-indexing
  for (auto& b : nbt_) --b;

  std::vector<int> int_temp;
  read_attribute(dset, "interpolation", int_temp);

  // Convert vector of ints into Interpolation
  for (const auto i : int_temp)
    int_.push_back(int2interp(i));

  xt::xarray<double> arr;
  read_dataset(dset, arr);

  auto xs = xt::view(arr, 0);
  auto ys = xt::view(arr, 1);

  std::copy(xs.begin(), xs.end(), std::back_inserter(x_));
  std::copy(ys.begin(), ys.end(), std::back_inserter(y_));
  n_pairs_ = x_.size();
}

double Tabulated1D::operator()(double x) const
{
  // find which bin the abscissa is in -- if the abscissa is outside the
  // tabulated range, the first or last point is chosen, i.e. no interpolation
  // is done outside the energy range
  int i;
  if (x < x_[0]) {
    return y_[0];
  } else if (x > x_[n_pairs_ - 1]) {
    return y_[n_pairs_ - 1];
  } else {
    i = lower_bound_index(x_.begin(), x_.end(), x);
  }

  // determine interpolation scheme
  Interpolation interp;
  if (n_regions_ == 0) {
    interp = Interpolation::lin_lin;
  } else if (n_regions_ == 1) {
    interp = int_[0];
  } else if (n_regions_ > 1) {
    for (int j = 0; j < n_regions_; ++j) {
      if (i < nbt_[j]) {
        interp = int_[j];
        break;
      }
    }
  }

  // handle special case of histogram interpolation
  if (interp == Interpolation::histogram) return y_[i];

  // determine bounding values
  double x0 = x_[i];
  double x1 = x_[i + 1];
  double y0 = y_[i];
  double y1 = y_[i + 1];

  // determine interpolation factor and interpolated value
  double r;
  switch (interp) {
  case Interpolation::lin_lin:
    r = (x - x0)/(x1 - x0);
    return y0 + r*(y1 - y0);
  case Interpolation::lin_log:
    r = log(x/x0)/log(x1/x0);
    return y0 + r*(y1 - y0);
  case Interpolation::log_lin:
    r = (x - x0)/(x1 - x0);
    return y0*exp(r*log(y1/y0));
  case Interpolation::log_log:
    r = log(x/x0)/log(x1/x0);
    return y0*exp(r*log(y1/y0));
  }
}

} // namespace openmc