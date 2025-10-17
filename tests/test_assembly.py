from .context import spherical_spectral_element
from spherical_spectral_element.config import np, npt
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.spectral import deriv
from spherical_spectral_element.equiangular_metric import gen_metric_terms_equiangular, generate_metric_terms, gen_metric_from_topo
from spherical_spectral_element.assembly import init_dss_matrix, dss_scalar_for, dss_scalar



def test_dss():
  nx = 3
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  metrics = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  gll_latlon, gll_to_sphere_jacobian, gll_to_sphere_jacobian_inv, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)
  fn = np.zeros_like(gll_latlon[:, :, :, 0])
  for face_idx in range(gll_latlon.shape[0]):
    for i_idx in range(npt):
      for j_idx in range(npt):
        fn[:] = 0.0
        fn[face_idx, i_idx, j_idx] = 1.0
        if face_idx in vert_redundancy_gll.keys():
          if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
            for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
              fn[remote_face_id, remote_i, remote_j] = 1.0
        assert(np.allclose(dss_scalar(fn, dss_matrix), fn))
        assert(np.allclose(dss_scalar_for(fn, metdet, inv_mass_mat, vert_redundancy_gll), fn))


def test_dss_equiv():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  metrics = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  gll_latlon, gll_to_sphere_jacobian, gll_to_sphere_jacobian_inv, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)
  fn = np.cos(gll_latlon[:, :, :, 1]) * np.cos(gll_latlon[:, :, :, 0])
  assert(np.allclose(dss_scalar(fn, dss_matrix), fn))
  ones = np.ones_like(metdet)
  ones_out = dss_scalar(ones, dss_matrix)
  assert(np.allclose(ones_out, ones))
  ones_out_for = dss_scalar_for(ones, metdet, inv_mass_mat, vert_redundancy_gll)
  assert(np.allclose(ones_out_for, ones))


def test_dss_equiv_rand():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  metrics = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  gll_latlon, gll_to_sphere_jacobian, gll_to_sphere_jacobian_inv, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)
  for _ in range(20):
    fn_rand = np.random.uniform(size=gll_latlon[:, :, :, 1].shape)
    assert(np.allclose(dss_scalar(fn_rand, dss_matrix), dss_scalar_for(fn_rand, metdet, inv_mass_mat, vert_redundancy_gll)))


