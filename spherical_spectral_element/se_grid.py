from .config import np, npt
from .spectral import deriv
from scipy.sparse import coo_array
def init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll):

  NELEM = metdet.shape[0]
  index_hack = np.zeros((NELEM, npt, npt), dtype=np.int64)
  #hack: easier than figuring out indexing conventions
  index_hack = np.arange(index_hack.size).reshape(index_hack.shape)

  data = []
  rows = []
  cols = []
  data_un = []

  for face_idx in range(NELEM):
    for i_idx in range(npt):
      for j_idx in range(npt):
        data.append(metdet[face_idx, i_idx, j_idx] *  (deriv["gll_weights"][i_idx] * deriv["gll_weights"][j_idx]) * inv_mass_mat[face_idx, i_idx, j_idx])
        data_un.append(inv_mass_mat[face_idx, i_idx, j_idx])
        rows.append(index_hack[face_idx, i_idx, j_idx])
        cols.append(index_hack[face_idx, i_idx, j_idx])
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        data.append(metdet[local_face_idx, local_i, local_j] *  (deriv["gll_weights"][local_i] * deriv["gll_weights"][local_j]) * inv_mass_mat[local_face_idx, local_i, local_j])
        data_un.append(inv_mass_mat[local_face_idx, local_i, local_j])
        rows.append(index_hack[remote_face_id, remote_i, remote_j])
        cols.append(index_hack[local_face_idx, local_i, local_j])
  # sparse matrix representation makes eventual autograd port significantly easier
  dss_matrix = coo_array((data, (rows, cols)), shape=(NELEM * npt * npt, NELEM*npt * npt))
  dss_matrix_unscaled = coo_array((data_un, (rows, cols)), shape=(NELEM * npt * npt, NELEM*npt * npt))

  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return dss_matrix, dss_matrix_unscaled, (data, data_un, rows, cols)

def create_spectral_element_grid(latlon,
               gll_to_sphere_jacobian, 
               gll_to_sphere_jacobian_inv, 
               rmetdet, 
               metdet, 
               mass_mat, 
               inv_mass_mat,
               vert_redundancy):
  dss_matrix, dss_matrix_unscaled, dss_triple = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy)
  return {"physical_coords": latlon,
          "jacobian": gll_to_sphere_jacobian,
          "jacobian_inv": gll_to_sphere_jacobian_inv,
          "recip_met_det": rmetdet,
          "met_det": metdet,
          "mass_mat": mass_mat,
          "mass_matrix_inv": inv_mass_mat,
          "met_inv": np.einsum("fijgs, fijhs->fijgh", 
                               gll_to_sphere_jacobian_inv, 
                               gll_to_sphere_jacobian_inv),
          "vert_redundancy": vert_redundancy,
          "deriv": deriv["deriv"],
          "gll_weights": deriv["gll_weights"],
          "npt": npt,
          "dss_triple": dss_triple,
          "num_elem": metdet.shape[0],
          "dss_matrix": dss_matrix,
          "dss_matrix_unscaled": dss_matrix_unscaled
          }



