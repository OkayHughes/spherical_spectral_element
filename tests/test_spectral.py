from .context import spherical_spectral_element
from spherical_spectral_element.config import np
from spherical_spectral_element.spectral import deriv


def test_quadrature():
  assert(np.allclose(np.sum(deriv.gll_weights * deriv.gll_points**2), 2/3 ))

def test_generate_derivative():
  assert(np.allclose(np.dot(deriv.deriv, deriv.gll_points**2 - deriv.gll_points**3),  (2*deriv.gll_points - 3 * deriv.gll_points**2)))
