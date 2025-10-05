from .config import np, npt
from .spectral import deriv

def sphere_gradient(f, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet):
  df_dab = np.zeros((f.shape[0], npt, npt, 2))
  df_dab[:, :, :, 0] = np.einsum("fij,ki->fkj", f, deriv.deriv)
  df_dab[:, :, :, 1] = np.einsum("fij,kj->fik", f, deriv.deriv)
  return  np.einsum("fijg,fijgs->fijs", df_dab, sphere_to_gll_jacobian)

def sphere_divergence(u, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet):
  u_contra = metdet[:,:,:,np.newaxis] * sph_to_contra(u, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)
  div = np.zeros_like(u[:,:,:,0])
  div += np.einsum("fij,ki->fkj", u_contra[:,:,:,0], deriv.deriv)
  div += np.einsum("fij,kj->fik", u_contra[:,:,:,1], deriv.deriv)
  div *= rmetdet[:,:,:]
  return div

def sphere_vorticity(u, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet):
  u_cov =  sph_to_cov(u, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)
  vort = np.zeros_like(u[:,:,:,0])
  dv_da = np.einsum("fij,ki->fkj", u_cov[:,:,:,1], deriv.deriv)
  vort += dv_da
  du_db = np.einsum("fij,kj->fik", u_cov[:,:,:,0], deriv.deriv)
  vort -= du_db
  vort *= rmetdet[:,:,:]
  return vort

def sphere_laplacian(f, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet):
  grad = sphere_gradient(f, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)
  return sphere_divergence(grad, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)

def sph_to_contra(u, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet):
  return np.einsum("fijg,fijgs->fijs", u, sphere_to_gll_jacobian)

def sph_to_cov(u, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet):
  return np.einsum("fijg,fijgs->fijs", u, gll_to_sphere_jacobian)

def inner_prod(f, g, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet):
  return np.sum(f * g * metdet * deriv.gll_weights[np.newaxis, :, np.newaxis] * deriv.gll_weights[np.newaxis, np.newaxis, :])


