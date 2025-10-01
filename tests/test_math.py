from .context import spherical_spectral_element
from spherical_spectral_element.config import np, npt
from spherical_spectral_element.math import bilinear, bilinear_jacobian
from spherical_spectral_element.cubed_sphere import gen_cube_topo
from spherical_spectral_element.spectral import Deriv

def test_bilinear():
  NFACES = 1000
  np.random.seed(0)
  face_position_2d = np.random.uniform(size=(NFACES, 4, 2))
  res = np.zeros(shape=(NFACES, 2))
  jac_test = np.zeros(shape=(NFACES, 2, 2))
  diff_minus = np.zeros(shape=(NFACES, 2))
  diff_plus = np.zeros(shape=(NFACES, 2))
  ncheck = 5
  nfrac = np.linspace(-1, 1, ncheck)
  for i in range(ncheck):
    for j in range(ncheck):
      alpha = nfrac[i]
      beta = nfrac[j]
      eps = 1e-6
      res = bilinear(face_position_2d[:, 0, :],
                     face_position_2d[:, 1, :],
                     face_position_2d[:, 2, :],
                     face_position_2d[:, 3, :], alpha, beta)
      diff_plus = bilinear(face_position_2d[:, 0, :],
                           face_position_2d[:, 1, :],
                           face_position_2d[:, 2, :],
                           face_position_2d[:, 3, :], alpha+eps, beta)
      diff_minus = bilinear(face_position_2d[:, 0, :],
                            face_position_2d[:, 1, :],
                            face_position_2d[:, 2, :],
                            face_position_2d[:, 3, :], alpha-eps, beta)
      dres_dalpha = (diff_plus - diff_minus)/(2 * eps)
      diff_plus = bilinear(face_position_2d[:, 0, :],
                           face_position_2d[:, 1, :],
                           face_position_2d[:, 2, :],
                           face_position_2d[:, 3, :], alpha, beta+eps)
      diff_minus = bilinear(face_position_2d[:, 0, :],
                            face_position_2d[:, 1, :],
                            face_position_2d[:, 2, :],
                            face_position_2d[:, 3, :], alpha, beta-eps)
      dphys_dalpha, dphys_dbeta = bilinear_jacobian(face_position_2d[:, 0, :],
                                                    face_position_2d[:, 1, :],
                                                    face_position_2d[:, 2, :],
                                                    face_position_2d[:, 3, :], alpha, beta)
      jac_test[:,:,0] = dphys_dalpha
      jac_test[:,:,1] = dphys_dbeta
      dres_dbeta = (diff_plus - diff_minus)/(2 * eps)
      assert(np.max(np.abs(dres_dalpha - jac_test[:, :, 0])) < 1e-7)
      assert(np.max(np.abs(dres_dbeta - jac_test[:, :, 1])) < 1e-7)

def test_bilinear_cs():
  nx = 15
  face_connectivity, face_position, face_position_2d = gen_cube_topo(nx)
  deriv = Deriv()
  NFACES = face_position.shape[0]
  res = np.zeros(shape=(NFACES, 2))
  jac_test = np.zeros(shape=(NFACES, 2, 2))
  diff_minus = np.zeros(shape=(NFACES, 2))
  diff_plus = np.zeros(shape=(NFACES, 2))
  for i in range(npt):
    for j in range(npt):
      alpha = deriv.gll_points[i]
      beta = deriv.gll_points[j]
      eps = 1e-4
      res = bilinear(face_position_2d[:, 0, :],
                     face_position_2d[:, 1, :],
                     face_position_2d[:, 2, :],
                     face_position_2d[:, 3, :], alpha, beta)
      diff_plus = bilinear(face_position_2d[:, 0, :],
                           face_position_2d[:, 1, :],
                           face_position_2d[:, 2, :],
                           face_position_2d[:, 3, :], alpha+eps, beta)
      diff_minus = bilinear(face_position_2d[:, 0, :],
                            face_position_2d[:, 1, :],
                            face_position_2d[:, 2, :],
                            face_position_2d[:, 3, :], alpha-eps, beta)
      dres_dalpha = (diff_plus - diff_minus)/(2 * eps)
      diff_plus = bilinear(face_position_2d[:, 0, :],
                           face_position_2d[:, 1, :],
                           face_position_2d[:, 2, :],
                           face_position_2d[:, 3, :], alpha, beta+eps)
      diff_minus = bilinear(face_position_2d[:, 0, :],
                            face_position_2d[:, 1, :],
                            face_position_2d[:, 2, :],
                            face_position_2d[:, 3, :], alpha, beta-eps)
      dphys_dalpha, dphys_dbeta = bilinear_jacobian(face_position_2d[:, 0, :],
                                                    face_position_2d[:, 1, :],
                                                    face_position_2d[:, 2, :],
                                                    face_position_2d[:, 3, :], alpha, beta)
      jac_test[:,:,0] = dphys_dalpha
      jac_test[:,:,1] = dphys_dbeta
      dres_dbeta = (diff_plus - diff_minus)/(2 * eps)
      assert(np.max(np.abs(dres_dalpha - jac_test[:, :, 0])) < 1e-7)
      assert(np.max(np.abs(dres_dbeta - jac_test[:, :, 1])) < 1e-7)