from .config import np

# we assume this is fixed for the moment.
p_points = np.array([1.0, np.sqrt(1/5), -np.sqrt(1/5), -1.0])
g_weights = np.array([1/6, 5/6, 5/6, 1/6])
npt = len(p_points)


def generate_derivative():
  # uses the lagrange interpolating polynomials
  leg_eval = np.zeros(shape=(npt, npt))
  leg_der = np.zeros(shape=(npt, npt))

  for deg in range(npt):
    c = np.zeros(npt)
    c[deg] = 1.0
    leg_eval[:, deg] = np.polynomial.legendre.legval(p_points, c)
    der = np.polynomial.legendre.legder(c, 1)
    leg_der[:, deg] = np.polynomial.legendre.legval(p_points, der)

  coeffs = np.linalg.inv(leg_eval)
  deriv = np.dot(leg_der, coeffs)


  return deriv
