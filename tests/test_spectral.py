from .context import spherical_spectral_element
from spherical_spectral_element.config import np
from spherical_spectral_element.spectral import generate_derivative
from spherical_spectral_element.spectral import p_points, g_weights


def test_quadrature():
  assert(np.allclose(np.sum(g_weights * p_points**2), 2/3 ))

def test_generate_derivative():
  deriv = generate_derivative()
  assert(np.allclose(np.dot(deriv, p_points**2 - p_points**3),  (2*p_points - 3 * p_points**2)))
