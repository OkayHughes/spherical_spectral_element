from .config import np, npt
from .spectral import deriv

def sphere_gradient(f, grid):
  df_dab = np.zeros((f.shape[0], npt, npt, 2))
  df_dab[:, :, :, 0] = np.einsum("fij,ki->fkj", f, deriv.deriv)
  df_dab[:, :, :, 1] = np.einsum("fij,kj->fik", f, deriv.deriv)
  return  np.einsum("fijg,fijgs->fijs", df_dab, grid.jacobian_inv)

def sphere_divergence(u, grid):
  u_contra = grid.met_det[:,:,:,np.newaxis] * sph_to_contra(u, grid)
  div = np.zeros_like(u[:,:,:,0])
  div += np.einsum("fij,ki->fkj", u_contra[:,:,:,0], deriv.deriv)
  div += np.einsum("fij,kj->fik", u_contra[:,:,:,1], deriv.deriv)
  div *= grid.recip_met_det[:,:,:]
  return div

def sphere_vorticity(u, grid):
  u_cov =  sph_to_cov(u, grid)
  vort = np.zeros_like(u[:,:,:,0])
  dv_da = np.einsum("fij,ki->fkj", u_cov[:,:,:,1], deriv.deriv)
  vort += dv_da
  du_db = np.einsum("fij,kj->fik", u_cov[:,:,:,0], deriv.deriv)
  vort -= du_db
  vort *= grid.recip_met_det[:,:,:]
  return vort

def sphere_laplacian(f, grid):
  grad = sphere_gradient(f, grid)
  return sphere_divergence(grad, grid)

def sph_to_contra(u, grid):
  return np.einsum("fijg,fijgs->fijs", u, grid.jacobian_inv)

def sph_to_cov(u, grid):
  return np.einsum("fijg,fijgs->fijs", u, grid.jacobian)

def inner_prod(f, g, grid):
  return np.sum(f * g * grid.met_det * deriv.gll_weights[np.newaxis, :, np.newaxis] * deriv.gll_weights[np.newaxis, np.newaxis, :])


