from .config import np, npt
from scipy.sparse import coo_array
from .spectral import deriv


def init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll):
  NELEM = metdet.shape[0]
  index_hack = np.zeros((NELEM, npt, npt), dtype=np.int64)
  #hack: easier than figuring out indexing conventions
  index_hack = np.arange(index_hack.size).reshape(index_hack.shape)

  data = []
  rows = []
  cols = []

  for face_idx in range(NELEM):
    for i_idx in range(npt):
      for j_idx in range(npt):
        data.append(metdet[face_idx, i_idx, j_idx] *  (deriv.gll_weights[i_idx] * deriv.gll_weights[j_idx]) * inv_mass_mat[face_idx, i_idx, j_idx])
        rows.append(index_hack[face_idx, i_idx, j_idx])
        cols.append(index_hack[face_idx, i_idx, j_idx])
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        data.append(metdet[local_face_idx, local_i, local_j] *  (deriv.gll_weights[local_i] * deriv.gll_weights[local_j]) * inv_mass_mat[local_face_idx, local_i, local_j])
        rows.append(index_hack[remote_face_id, remote_i, remote_j])
        cols.append(index_hack[local_face_idx, local_i, local_j])
  # sparse matrix representation makes eventual autograd port significantly easier
  dss_matrix = coo_array((data, (rows, cols)), shape=(NELEM * npt * npt, NELEM*npt * npt))
  print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return dss_matrix
def dss_scalar_for(f, metdet, inv_mass_mat, vert_redundancy_gll):
  workspace = f.copy()
  workspace *= metdet * (deriv.gll_weights[np.newaxis, :, np.newaxis] * deriv.gll_weights[np.newaxis, np.newaxis, :])
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        workspace[remote_face_id, remote_i, remote_j] += metdet[local_face_idx, local_i, local_j] * f[local_face_idx, local_i, local_j] * (deriv.gll_weights[local_i] * deriv.gll_weights[local_j])
  workspace *= inv_mass_mat
  return workspace

def dss_scalar(f, dss_matrix):
  return (dss_matrix@f.flatten()).reshape(f.shape)

