from .config import np

class Deriv:
  """ """
  def __init__(self, npt):
    self.npt = npt
    self.init_quadrature()
    self.init_deriv()

  def init_quadrature(self):
    if self.npt == 4:
      # we assume this is fixed for the moment.
      self.gll_points = np.array([1.0, np.sqrt(1/5), -np.sqrt(1/5), -1.0])
      self.gll_weights = np.array([1/6, 5/6, 5/6, 1/6])
    else:
      raise ValueError(f"GLL points not computed for npt={self.npt}")
  def init_deriv(self):
    # uses the lagrange interpolating polynomials
    leg_eval = np.zeros(shape=(self.npt, self.npt))
    leg_der = np.zeros(shape=(self.npt, self.npt))

    for deg in range(self.npt):
      c = np.zeros(self.npt)
      c[deg] = 1.0
      leg_eval[:, deg] = np.polynomial.legendre.legval(self.gll_points, c)
      der = np.polynomial.legendre.legder(c, 1)
      leg_der[:, deg] = np.polynomial.legendre.legval(self.gll_points, der)

    coeffs = np.linalg.inv(leg_eval)
    self.deriv = np.dot(leg_der, coeffs)





