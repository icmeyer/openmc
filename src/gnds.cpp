#include "openmc/gnds.h"

#include <algorithm> // for copy
#include <cmath>     // for log, exp
#include <iterator>  // for back_inserter

#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/function.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/search.h"

namespace openmc {

//General function for returning a pointer to a Function1D-type from hdf5 group
UPtrFunction
get_function1D(hid_t group, const char* name)
{
  hid_t obj = open_object(group, name);
  std::string temp;
  read_attribute(obj, "type", temp);

  UPtrFunction function;
  if (temp == "XYs1D") {
    function = UPtrFunction{new XYs1D{obj}};
  } else if (temp == "Polynomial") {
    function = UPtrFunction{new Polynomial{obj}};
  } else if (temp == "Regions1D") {
    function = UPtrFunction{new Regions1D{obj}};
  }
  close_object(obj);
  return function;
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
  std::string temp;
  read_attribute(group, "domainbreaks", domainbreaks_);
  n_regions_ = domainbreaks_.size() - 1;
  for (int i=0; i < n_regions_; i++){
    std::string region_str = "region_" + std::to_string(i);
    regions_.push_back(get_function1D(group, region_str.c_str()));
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
  if (x <= domainbreaks_[0]) {
    return (*regions_[0])(x);
  } else if (x >= domainbreaks_[n_regions_]) {
    return (*regions_[n_regions_ - 1])(x);
  } else {
    i = lower_bound_index(domainbreaks_.begin(), domainbreaks_.end(), x);
    return (*regions_[i])(x);
  }
}

} //namespace openmc

