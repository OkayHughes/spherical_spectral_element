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
def sphere_laplacian_wk(f, grid, a=1.0):
  grad = sphere_gradient(f, grid, a=a)
  return sphere_divergence_wk(grad, grid, a=a)

def sphere_gradient_wk_cov(s, grid, a=1.0):
  ds_contra = np.zeros((*s.shape, 2))
  ds_contra[:,:,:,0] = - np.einsum("j,n,fmn,fmn,fjn,jm->fmn", deriv.gll_weights, deriv.gll_weights, grid.met_inv[:,:,:,0,0], grid.met_det, s, deriv.deriv)
  ds_contra[:,:,:,0] -=  np.einsum("m,j,fmn,fmn,fmj,jn->fmn", deriv.gll_weights, deriv.gll_weights, grid.met_inv[:,:,:,1,0], grid.met_det, s, deriv.deriv)
  ds_contra[:,:,:,1] = - np.einsum("j,n,fmn,fmn,fjn,jm->fmn", deriv.gll_weights, deriv.gll_weights, grid.met_inv[:,:,:,0,1], grid.met_det, s, deriv.deriv)
  ds_contra[:,:,:,1] -=  np.einsum("m,j,fmn,fmn,fmj,jn->fmn", deriv.gll_weights, deriv.gll_weights, grid.met_inv[:,:,:,1,1], grid.met_det, s, deriv.deriv)
  return 1.0/a * contra_to_sph(ds_contra, grid)

def sphere_curl_wk_cov(s, grid, a=1.0):
  ds_contra = np.zeros((*s.shape, 2))
  ds_contra[:,:,:,0] =  np.einsum("m,j,fmj,jn->fmn", deriv.gll_weights, deriv.gll_weights, s, deriv.deriv)
  ds_contra[:,:,:,1] = - np.einsum("j,n,fjn,jm->fmn", deriv.gll_weights, deriv.gll_weights, s, deriv.deriv)
  return 1.0/a * contra_to_sph(ds_contra, grid)

def sphere_vec_laplacian_wk(u, grid, a=1.0, nu_fact = 1.0, damp=False):
  div = sphere_divergence(u, grid, a=a) * nu_fact
  vor = sphere_vorticity(u, grid, a=a)
  laplacian = sphere_gradient_wk_cov(div, grid, a=a) - sphere_curl_wk_cov(vor, grid, a=a)
  if damp:
    laplacian[:,:,:,0] += 2 * (deriv.gll_weights[np.newaxis, :, np.newaxis] * deriv.gll_weights[np.newaxis, np.newaxis, :]
                                * grid.met_det * u[:,:,:,0] * (1/a)**2)
    laplacian[:,:,:,1] += 2 * (deriv.gll_weights[np.newaxis, :, np.newaxis] * deriv.gll_weights[np.newaxis, np.newaxis, :]
                                * grid.met_det * u[:,:,:,1] * (1/a)**2)
  return laplacian







def sphere_divergence_wk(u, grid, a=1.0):
  contra = sph_to_contra(u, grid)
  du_da_wk = - np.einsum("n,j,fjn,fjn,jm->fmn", deriv.gll_weights, deriv.gll_weights, grid.met_det, contra[:,:,:,0], deriv.deriv)
  du_db_wk = - np.einsum("m,j,fmj,fmj,jn->fmn", deriv.gll_weights, deriv.gll_weights, grid.met_det, contra[:,:,:,1], deriv.deriv)
  return 1.0/a * (du_da_wk + du_db_wk) 

def contra_to_sph(u, grid):
  return np.flip(np.einsum("fijg,fijsg->fijs", u, grid.jacobian), axis=-1)

def sph_to_contra(u, grid):
  return np.einsum("fijs,fijgs->fijg", np.flip(u, axis=-1), grid.jacobian_inv)

def sph_to_cov(u, grid):
  return np.einsum("fijs,fijsg->fijg", np.flip(u, axis=-1), grid.jacobian)

def inner_prod(f, g, grid):
  return np.sum(f * g * grid.met_det * deriv.gll_weights[np.newaxis, :, np.newaxis] * deriv.gll_weights[np.newaxis, np.newaxis, :])


