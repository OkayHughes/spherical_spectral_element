from .config import np, npt
from .spectral import deriv

def sphere_gradient(f, grid, a=1.0):
  df_dab = np.zeros((f.shape[0], npt, npt, 2))
  df_dab[:, :, :, 0] = np.einsum("fij,ki->fkj", f, deriv.deriv)
  df_dab[:, :, :, 1] = np.einsum("fij,kj->fik", f, deriv.deriv)
  return  1.0/a * np.flip(np.einsum("fijg,fijgs->fijs", df_dab, grid.jacobian_inv), axis=-1)

def sphere_divergence(u, grid, a=1.0):
  u_contra = 1.0/a * grid.met_det[:,:,:,np.newaxis] * sph_to_contra(u, grid)
  div = np.zeros_like(u[:,:,:,0])
  div += np.einsum("fij,ki->fkj", u_contra[:,:,:,0], deriv.deriv)
  div += np.einsum("fij,kj->fik", u_contra[:,:,:,1], deriv.deriv)
  div *= grid.recip_met_det[:,:,:]
  return div

def sphere_vorticity(u, grid, a=1.0):
  u_cov =  sph_to_cov(u, grid)
  vort = np.zeros_like(u[:,:,:,0])
  dv_da = np.einsum("fij,ki->fkj", u_cov[:,:,:,1], deriv.deriv)
  vort -= dv_da
  du_db = np.einsum("fij,kj->fik", u_cov[:,:,:,0], deriv.deriv)
  vort += du_db
  vort *= 1.0/a * grid.recip_met_det[:,:,:]
  return vort

def sphere_laplacian(f, grid, a=1.0):
  grad = sphere_gradient(f, grid, a=a)
  return sphere_divergence(grad, grid, a=a)
def sphere_vlaplacian(u, grid, a=1.0):
  #grad = sphere_gradient(f, grid, a=a)
  #return sphere_divergence(grad, grid, a=a)
  pass

def sphere_divergence_wk(u,grid, a=1.0):
  contra = sph_to_contra(u, grid)
  du_da_wk = - np.einsum("n,j,fjn,fjn,jm->fmn", deriv.gll_weights, deriv.gll_weights, grid.met_det, contra[:,:,:,0], deriv.deriv)
  du_db_wk = - np.einsum("m,j,fmj,fmj,jn->fmn", deriv.gll_weights, deriv.gll_weights, grid.met_det, contra[:,:,:,1], deriv.deriv)
  return 1.0/a * (du_da_wk + du_db_wk) 


def sph_to_contra(u, grid):
  return np.einsum("fijs,fijgs->fijg", np.flip(u, axis=-1), grid.jacobian_inv)

def sph_to_cov(u, grid):
  return np.einsum("fijs,fijsg->fijg", np.flip(u, axis=-1), grid.jacobian)

def inner_prod(f, g, grid):
  return np.sum(f * g * grid.met_det * deriv.gll_weights[np.newaxis, :, np.newaxis] * deriv.gll_weights[np.newaxis, np.newaxis, :])


