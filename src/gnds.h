#ifndef OPENMC_GNDS_H
#define OPENMC_GNDS_H

#include <vector>

#include "constants.h"
#include "function.h"
#include "hdf5.h"

namespace openmc {

Interpolation int2interp(int i);
bool is_fission(int MT);
UPtrFunction get_function1D(hid_t obj_id, const char* name);

class XYs1D : public Function1D {
public:
  XYs1D() = default;
  explicit XYs1D(hid_t dset);
  double operator()(double x) const;
private:
  Interpolation int_;
  std::size_t n_pairs_;
  std::vector<double> x_;
  std::vector<double> y_;
};

class Regions1D : public Function1D {
public:
  Regions1D() = default;
  explicit Regions1D(hid_t dset);
  double operator()(double x) const;
private:
  std::size_t n_regions_ {0};
  std::vector<UPtrFunction> regions_;
  std::vector<double> domainbreaks_;
};

} //namespace openmc

#endif // OPENMC_GNDS_H



