from .config import np
from scipy.sparse import coo_array


index_hack = np.zeros((NELEM, npt, npt), dtype=np.int64)
index_hack = np.arange(index_hack.size).reshape(index_hack.shape)

data = []
rows = []
cols = []

for face_idx in range(NELEM):
  for i_idx in range(npt):
    for j_idx in range(npt):
      data.append(metdet[face_idx, i_idx, j_idx] *  (g_weights[i_idx] * g_weights[j_idx]) * inv_mass_mat[face_idx, i_idx, j_idx])
      rows.append(index_hack[face_idx, i_idx, j_idx])
      cols.append(index_hack[face_idx, i_idx, j_idx])
for local_face_idx in vert_redundancy_gll.keys():
  for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
    for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
      data.append(metdet[local_face_idx, local_i, local_j] *  (g_weights[local_i] * g_weights[local_j]) * inv_mass_mat[local_face_idx, local_i, local_j])
      rows.append(index_hack[remote_face_id, remote_i, remote_j])
      cols.append(index_hack[local_face_idx, local_i, local_j])
# sparse matrix representation makes eventual autograd port significantly easier
dss_matrix = coo_array((data, (rows, cols)), shape=(NELEM * npt * npt, NELEM*npt * npt))
print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")

def dss_scalar_for(f):
  workspace = f.copy()
  workspace *= metdet * (g_weights[np.newaxis, :, np.newaxis] * g_weights[np.newaxis, np.newaxis, :])
  df_dab = np.zeros((gll_latlon.shape[0], npt, npt, 2))
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        workspace[remote_face_id, remote_i, remote_j] += metdet[local_face_idx, local_i, local_j] * f[local_face_idx, local_i, local_j] * (g_weights[local_i] * g_weights[local_j])
  workspace *= inv_mass_mat
  return workspace
def dss_scalar(f):
  return (dss_matrix@f.flatten()).reshape(f.shape)





def sphere_gradient(f):
  df_dab = np.zeros((f.shape[0], npt, npt, 2))
  df_dab[:, :, :, 0] = np.einsum("fij,ki->fkj", f, deriv)
  df_dab[:, :, :, 1] = np.einsum("fij,kj->fik", f, deriv)
  return  np.einsum("fijg,fijgs->fijs", df_dab, gll_to_sphere_jacobian_inv)

def sphere_divergence(u):
  u_contra = metdet[:,:,:,np.newaxis] * sph_to_contra(u)
  div = np.zeros_like(u[:,:,:,0])
  div += np.einsum("fij,ki->fkj", u_contra[:,:,:,0], deriv)
  div += np.einsum("fij,kj->fik", u_contra[:,:,:,1], deriv)
  div *= rmetdet[:,:,:]
  return div

def sphere_vorticity(u):
  u_cov =  sph_to_cov(u)
  vort = np.zeros_like(u[:,:,:,0])
  dv_da = np.einsum("fij,ki->fkj", u_cov[:,:,:,1], deriv)
  vort += dv_da
  du_db = np.einsum("fij,kj->fik", u_cov[:,:,:,0], deriv)
  vort -= du_db
  vort *= rmetdet[:,:,:]
  return vort

def sphere_laplacian(f):
  grad = sphere_gradient(f)
  return sphere_divergence(grad)

def sph_to_contra(u):
  return np.einsum("fijg,fijgs->fijs", u, gll_to_sphere_jacobian_inv)

def sph_to_cov(u):
  return np.einsum("fijg,fijgs->fijs", u, gll_to_sphere_jacobian)

def inner_prod(f, g):
  return np.sum(f * g * metdet * g_weights[np.newaxis, :, np.newaxis] * g_weights[np.newaxis, np.newaxis, :])

if TESTING:
  if STRINGENT:
    fn = np.zeros_like(gll_latlon[:, :, :, 0])
    for face_idx in tqdm(range(gll_latlon.shape[0])):
      for i_idx in range(npt):
        for j_idx in range(npt):
          fn[:] = 0.0
          fn[face_idx, i_idx, j_idx] = 1.0
          if face_idx in vert_redundancy_gll.keys():
            if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
              for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
                fn[remote_face_id, remote_i, remote_j] = 1.0
          assert(np.allclose(dss_scalar(fn), fn))
  fn = np.cos(gll_latlon[:, :, :, 1]) * np.cos(gll_latlon[:, :, :, 0])
  ones = np.ones_like(metdet)
  ones_out = dss_scalar(ones)
  assert(np.max(np.abs(ones_out - ones)) < 1e-7)
  ones_out_for = dss_scalar_for(ones)
  assert(np.max(np.abs(ones_out_for - ones)) < 1e-7)
  fn_rand = np.random.uniform(size=gll_latlon[:, :, :, 1].shape)
  assert(np.allclose(dss_scalar(fn_rand), dss_scalar_for(fn_rand)))
  print(inner_prod(ones, ones))
  grad = sphere_gradient(fn)
  vort = sphere_vorticity(grad)
  print(inner_prod(vort, vort))
  #print(vort)
  v = np.zeros_like(gll_latlon)
  v[:,:,:,0] = np.cos(gll_latlon[:, :, :, 0])
  v[:,:,:,1] = np.cos(gll_latlon[:, :, :, 0])
  u = np.zeros_like(gll_latlon)
  u[:,:,:,0] = np.cos(2*gll_latlon[:, :, :, 0])
  u[:,:,:,1] = np.cos(2*gll_latlon[:, :, :, 0])

  v_contra = sph_to_contra(v)
  grad = sphere_gradient(fn)
  print(inner_prod(v_contra[:,:,:,0], grad[:,:,:,0]) + inner_prod(v_contra[:,:,:,1], grad[:,:,:,1]) - inner_prod(fn, sphere_divergence(v)))

