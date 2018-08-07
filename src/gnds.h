#ifndef OPENMC_GNDS_H
#define OPENMC_GNDS_H

#include <vector>

#include "constants.h"
#include "endf.h"
#include "hdf5.h"

namespace openmc {

Interpolation int2interp(int i);
bool is_fission(int MT);

class Function1D {
public:
  virtual double operator()(double x) const = 0;
};

class Polynomial : public Function1D {
public:
  explicit Polynomial(hid_t dset);
  double operator()(double x) const;
private:
  std::vector<double> coef_;
};

class Legendre : public Function1D {
public:
  explicit Legendre(hid_t dset);
  double operator()(double x) const;
private:
  std::vector<double> coef_;
};

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
  std::vector<Function1D*> regions_;
  std::vector<double> domainbreaks_;
};

} //namespace openmc

#endif // OPENMC_GNDS_H



