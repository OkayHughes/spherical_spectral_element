from .config import np, npt
from scipy.sparse import coo_array
from .spectral import deriv



def dss_scalar_for(f, grid):
  metdet = grid.met_det
  inv_mass_mat = grid.mass_matrix_inv
  vert_redundancy_gll = grid.vert_redundancy
  workspace = f.copy()
  workspace *= metdet * (deriv.gll_weights[np.newaxis, :, np.newaxis] * deriv.gll_weights[np.newaxis, np.newaxis, :])
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        workspace[remote_face_id, remote_i, remote_j] += metdet[local_face_idx, local_i, local_j] * f[local_face_idx, local_i, local_j] * (deriv.gll_weights[local_i] * deriv.gll_weights[local_j])
  workspace *= inv_mass_mat
  return workspace

def dss_scalar(f, grid):
  return (grid.dss_matrix@f.flatten()).reshape(f.shape)

