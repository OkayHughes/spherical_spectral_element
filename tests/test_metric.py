from .context import spherical_spectral_element
from spherical_spectral_element.config import np, npt
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.spectral import deriv
from spherical_spectral_element.equiangular_metric import gen_metric_terms_equiangular, generate_metric_terms, gen_metric_from_topo
from spherical_spectral_element.mesh import gen_bilinear_grid


def test_gen_metric():
  nx = 7
  face_connectivity, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  gll_pos, gll_pos_2d, gll_jacobian_2d, gll_jacobian_2d_inv, cube_redundancy = gen_bilinear_grid(face_connectivity, face_position, face_position_2d, vert_redundancy)
  gll_latlon, cube_to_sphere_jacobian, cube_to_sphere_jacobian_inv = gen_metric_terms_equiangular(gll_pos, gll_pos_2d, cube_redundancy)

  for elem_idx in cube_redundancy.keys():
    for (i_idx, j_idx) in cube_redundancy[elem_idx].keys():
      for elem_idx_pair, i_idx_pair, j_idx_pair in cube_redundancy[elem_idx][(i_idx, j_idx)]:
        try:
          assert(np.max(np.abs(gll_latlon[elem_idx, i_idx, j_idx, :] - gll_latlon[elem_idx_pair, i_idx_pair, j_idx_pair, :])) < 1e-10)
        except:
          print(f"Position failure: local: {(inv_elem_id_fn(elem_idx), i_idx, j_idx)} {180/np.pi * gll_latlon[elem_idx][(i_idx, j_idx)]} pair: {(inv_elem_id_fn(elem_idx_pair), i_idx_pair, j_idx_pair)} {180/ np.pi * gll_latlon[elem_idx_pair][(i_idx_pair, j_idx_pair)]}")


def test_gen_mass_mat():
  nx = 15
  face_connectivity, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  metrics = gen_metric_from_topo(face_connectivity, face_position, face_position_2d, vert_redundancy)
  gll_latlon, gll_to_sphere_jacobian, gll_to_sphere_jacobian_inv, rmetdet, metdet, mass_mat, inv_mass_mat, cube_redundancy = metrics
  assert(np.allclose(np.sum(metdet * (deriv.gll_weights[np.newaxis, :, np.newaxis] * deriv.gll_weights[np.newaxis, np.newaxis, :])), 4 * np.pi))