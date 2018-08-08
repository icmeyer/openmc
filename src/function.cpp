#include "function.h"

#include "hdf5_interface.h"

namespace openmc {

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

} // namespace openmc
