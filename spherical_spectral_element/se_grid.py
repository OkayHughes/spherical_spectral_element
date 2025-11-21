from .config import np, npt
from .spectral import deriv
from scipy.sparse import coo_array

class SpectralElementGrid():
  def __init__(self, latlon,
               gll_to_sphere_jacobian, 
               gll_to_sphere_jacobian_inv, 
               rmetdet, 
               metdet, 
               mass_mat, 
               inv_mass_mat,
               vert_redundancy):
    self.num_elem = latlon.shape[0]
    self.physical_coords = latlon
    self.jacobian = gll_to_sphere_jacobian
    self.jacobian_inv = gll_to_sphere_jacobian_inv
    self.recip_met_det = rmetdet
    self.met_det = metdet
    self.mass_matrix = mass_mat
    self.mass_matrix_inv = inv_mass_mat
    #note: this is for processor-local DSS
    self.vert_redundancy = vert_redundancy
    self.init_dss_matrix()
  def init_dss_matrix(self):
    metdet = self.met_det
    inv_mass_mat = self.mass_matrix_inv
    vert_redundancy_gll = self.vert_redundancy

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
    # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
    self.dss_matrix = dss_matrix