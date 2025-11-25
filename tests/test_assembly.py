from .context import spherical_spectral_element
from spherical_spectral_element.config import np, npt
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.spectral import deriv
from spherical_spectral_element.equiangular_metric import gen_metric_terms_equiangular, generate_metric_terms, gen_metric_from_topo
from spherical_spectral_element.assembly import dss_scalar_for, dss_scalar

def test_dss():
  nx = 3
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid  = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  vert_redundancy_gll = grid["vert_redundancy"]
  #gll_latlon, gll_to_sphere_jacobian, gll_to_sphere_jacobian_inv, rmetdet, metdet, mass_mat, inv_mass_mat,  = metrics
  fn = np.zeros_like(grid["physical_coords"][:, :, :, 0])
  for face_idx in range(grid["physical_coords"].shape[0]):
    for i_idx in range(npt):
      for j_idx in range(npt):
        fn[:] = 0.0
        fn[face_idx, i_idx, j_idx] = 1.0
        if face_idx in vert_redundancy_gll.keys():
          if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
            for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
              fn[remote_face_id, remote_i, remote_j] = 1.0
        assert(np.allclose(dss_scalar(fn, grid), fn))
        assert(np.allclose(dss_scalar_for(fn, grid), fn))

def test_dss_equiv():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid  = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  #gll_latlon, gll_to_sphere_jacobian, gll_to_sphere_jacobian_inv, rmetdet, metdet, mass_mat, inv_mass_mat,  = metrics
  fn = np.cos(grid["physical_coords"][:, :, :, 1]) * np.cos(grid["physical_coords"][:, :, :, 0])
  assert(np.allclose(dss_scalar(fn, grid), fn))
  ones = np.ones_like(grid["met_det"])
  ones_out = dss_scalar(ones, grid)
  assert(np.allclose(ones_out, ones))
  ones_out_for = dss_scalar_for(ones, grid)
  assert(np.allclose(ones_out_for, ones))


def test_dss_equiv_rand():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  for _ in range(20):
    fn_rand = np.random.uniform(size=grid["physical_coords"][:, :, :, 1].shape)
    assert(np.allclose(dss_scalar(fn_rand, grid), dss_scalar_for(fn_rand, grid)))


