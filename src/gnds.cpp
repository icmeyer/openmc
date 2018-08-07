#include "gnds.h"

#include <algorithm> // for copy
#include <cmath>     // for log, exp
#include <iterator>  // for back_inserter

#include "constants.h"
#include "hdf5_interface.h"
#include "math_functions.h"
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

bool is_fission(int mt)
{
  return mt == 18 || mt == 19 || mt == 20 || mt == 21 || mt == 38;
}

//General function for returning a pointer to a Function1D-type from hdf5 group
std::uniqe_ptr<Function1D>
get1D(hid_t group, const char* name)
{
  H5O_info_t oinfo;
  H5Oget_info_by_name(group, name, &oinfo, H5P_DEFAULT);
  std::str temp;

  if (oinfo.type == H5O_TYPE_DATSET){
    hid_t dset = open_dataset(group, name);
    read_attribute(dset, "type", temp)
    if (temp == "XYs1D") {
      function = std::unique_ptr<Function1D{new XYs1D{dset}};
    } else if (temp == "Polynomial") {
      function = std::unique_ptr<Function1D{new Polynomial{dset}};
    } else if (temp == "Legendre") {
      function = std::unique_ptr<Function1D{new Legendre{dset}};
    }
    close_dataset(dset);
    return function;
  } else if (oinfo.type == H5O_TYPE_GROUP) {
    hid_t grp = open_group(group, name)
    function = std::unique_ptr<Function1D{new Regions1D{grp}};
    close_dataset(grp);
    return function;
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
// Legendre implementation
//==============================================================================

Legendre::Legendre(hid_t dset)

  // Read coefficients into a vecot
  read_dataset(dset, coef_);
}

double Legendre::operator()(double x) const
{
  int n = coef_.size();
  return evaluate_legendre_c(n, coef_, x);
}

//==============================================================================
// XYs1D implementation
//==============================================================================

XYs1D::XYs1D(hid_t dset)
{
  int int_temp;
  read_attribute(dset, "interpolation", int_temp);
  int_ = int2interp(int_temp);

  xt::xarray<double> arr;
  read_dataset(dset, arr);

  auto xs = xt::view(arr, 0);
  auto ys = xt::view(arr, 1);

  // FIXME What do these two lines do???
  // Why use xtensor when they end up as vectors anyway?
  std::copy(xs.begin(), xs.end(), std::back_inserter(x_));
  std::copy(ys.begin(), ys.end(), std::back_inserter(y_));
  n_pairs_ = x_.size();
}

double XYs1D::operator()(double x) const
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

  // handle special case of histogram interpolation
  if (int_ == Interpolation::histogram) return y_[i];

  // determine bounding values
  double x0 = x_[i];
  double x1 = x_[i + 1];
  double y0 = y_[i];
  double y1 = y_[i + 1];

  // determine interpolation factor and interpolated value
  double r;
  switch (int_) {
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

//==============================================================================
// Regions1D implementation
//==============================================================================

Regions1D::Regions1D(hid_t group)
{
  std::string temp
  read_attribute(group, "domainbreaks", domainbreaks_);
  n_regions_ = domainbreaks_.size() - 1;
  for (int i=0; i < n_regions_; i++){
    std::str region_str = "region_" + std::to_string(i);
    regions_.push_back(std::unique_ptr<Function1D>
                       get1D(group, region_str.c_str()));
  }
}

//Old implementation just in case
//Regions1D::Regions1D(hid_t group)
//{
//  std::string temp
//  read_attribute(dset, "domainbreaks", domainbreaks_);
//  n_regions_ = domainbreaks_.size() - 1;
//  for (int i=0; i < n_regions_; i++){
//    hid_t region = open_dataset(group, "region"+std::to_string(i));
//    read_attribute(region, "type", temp);
//    if (temp == "XYs1D") {
//      regions_.push_back(std::unique_ptr<Function1D{new XYs1D{region}});
//    } else if (temp == "Polynomial") {
//      regions_.push_back(std::unique_ptr<Function1D{new Polynomial{region}});
//    } else if (temp == "Legendre") {
//      regions_.push_back(std::unique_ptr<Function1D{new Legendre{region}});
//    }
//    close_dataset(region);
//  }
//}

double Regions1D::operator()(double x) const
{ // Write call function for Regions1D
  int i;
  if (x < domainbreaks_[0]) {
    return (*regions_)[0](x);
  } else if (x > domainbreaks_[n_regions_]) {
    return (*regions_)[n_regions_ - 1](x);
  } else {
    i = lower_bound_index(domainbreaks_.begin(), domainbreaks_.end(), domainbreaks);
    return (*regions_[i])(x);
  }
}

} //namespace openmc

