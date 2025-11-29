
from spherical_spectral_element.config import np, jnp, eps
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.equiangular_metric import gen_metric_from_topo
from spherical_spectral_element.assembly import dss_scalar
from spherical_spectral_element.operators import sphere_gradient, sphere_divergence, sphere_vorticity, inner_prod
from spherical_spectral_element.operators import sphere_divergence_wk, sphere_gradient_wk_cov, sphere_vec_laplacian_wk
from spherical_spectral_element.theta_l.operators_3d import sphere_divergence_3d, sphere_gradient_3d, sphere_vorticity_3d, sphere_vec_laplacian_wk_3d


def threedify(field, nlev, axis=-1):
  if axis < 0:
    shape_out = list(field.shape)
    shape_out.reverse()
    shape_out.insert(-axis - 1, 1)
    shape_out.reverse()
  else:
    shape_out = field.shape
    shape_out.insert(axis, 1)
  ones_shape = [1 for _ in range(len(field.shape)+1)]
  ones_shape[axis] = nlev
  return field.reshape(shape_out) * jnp.ones(ones_shape)

def test_vector_identites():
  nx = 31
  nlev = 3
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  config = {"radius_earth": 1.0}
  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  fn_3d = threedify(fn, nlev)
  v = np.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
  v_3d = threedify(v, nlev, axis=-2)
  
  grad = sphere_gradient_3d(fn_3d, grid, config)
  vort = sphere_vorticity_3d(grad, grid, config)
  div = sphere_divergence_3d(v_3d, grid, config)
  for k_idx in range(nlev):
    iprod_vort = inner_prod(vort[:, :, :, k_idx], vort[:, :, :, k_idx], grid)
    assert (jnp.allclose(iprod_vort, 0, atol=eps))


    discrete_divergence_thm = (inner_prod(v_3d[:, :, :, k_idx, 0], grad[:, :, :, k_idx, 0], grid) +
                              inner_prod(v_3d[:, :, :, k_idx, 1], grad[:, :, :, k_idx, 1], grid) -
                              inner_prod(fn_3d[:, :, :, k_idx], div[:, :, :, k_idx], grid))
    assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))


def test_divergence():
  nx = 31
  nlev=3
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  config = {"radius_earth": 1.0}
  vec = np.zeros_like(grid["physical_coords"])
  lat = grid["physical_coords"][:, :, :, 0]
  lon = grid["physical_coords"][:, :, :, 1]
  vec[:, :, :, 0] = np.cos(lat)**2 * np.cos(lon)**3
  vec[:, :, :, 1] = np.cos(lat)**2 * np.cos(lon)**3
  
  vec_3d = threedify(vec, nlev, axis=-2)

  vort_analytic = (-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) +
                   3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3)

  div_analytic = (-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) -
                  3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3)
  div = dss_scalar(sphere_divergence(vec, grid), grid, dims)
  div_wk = dss_scalar(sphere_divergence_wk(vec, grid), grid, dims, scaled=False)
  vort = dss_scalar(sphere_vorticity(vec, grid), grid, dims)
  assert (inner_prod(div_wk - div, div_wk - div, grid) < 1e-5)
  assert (inner_prod(div_analytic - div, div_analytic - div, grid) < 1e-5)
  assert (inner_prod(vort_analytic - vort, vort_analytic - vort, grid) < 1e-5)


def test_analytic_soln():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)

  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  grad_f_numerical = sphere_gradient(fn, grid)
  sph_grad_wk = sphere_gradient_wk_cov(fn, grid)
  sph_grad_wk = jnp.stack((dss_scalar(sph_grad_wk[:, :, :, 0], grid, dims, scaled=False),
                           dss_scalar(sph_grad_wk[:, :, :, 1], grid, dims, scaled=False)), axis=-1)
  grad_diff = sph_grad_wk - grad_f_numerical

  sph_grad_lat = -jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.sin(grid["physical_coords"][:, :, :, 0])
  sph_grad_lon = -jnp.sin(grid["physical_coords"][:, :, :, 1])
  assert ((inner_prod(grad_diff[:, :, :, 0], grad_diff[:, :, :, 0], grid) +
           inner_prod(grad_diff[:, :, :, 1], grad_diff[:, :, :, 1], grid)
           ) < 1e-5)
  assert (np.max(np.abs(sph_grad_lat - grad_f_numerical[:, :, :, 1])) < 1e-4)
  assert (np.max(np.abs(sph_grad_lon - grad_f_numerical[:, :, :, 0])) < 1e-4)


def test_vector_laplacian():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                 jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
  laplace_v_wk = sphere_vec_laplacian_wk(v, grid)
  laplace_v_wk = jnp.stack((dss_scalar(laplace_v_wk[:, :, :, 0], grid, dims, scaled=False),
                            dss_scalar(laplace_v_wk[:, :, :, 1], grid, dims, scaled=False)), axis=-1)

  lap_diff = laplace_v_wk + 2 * v
  assert ((inner_prod(lap_diff[:, :, :, 0], lap_diff[:, :, :, 0], grid) +
           inner_prod(lap_diff[:, :, :, 1], lap_diff[:, :, :, 1], grid)) < 1e-5)
  v = jnp.stack((np.cos(grid["physical_coords"][:, :, :, 0])**2,
                 np.cos(grid["physical_coords"][:, :, :, 0])**2), axis=-1)
  laplace_v_wk = sphere_vec_laplacian_wk(v, grid)
  laplace_v_wk = jnp.stack((dss_scalar(laplace_v_wk[:, :, :, 0], grid, dims, scaled=False),
                            dss_scalar(laplace_v_wk[:, :, :, 1], grid, dims, scaled=False)), axis=-1)
  lap_diff = laplace_v_wk + 3 * (np.cos(2 * grid["physical_coords"][:, :, :, 0]))[:, :, :, np.newaxis]
  lap_diff *= np.cos(grid["physical_coords"][:, :, :, 0])[:, :, :, np.newaxis]**2  # hack to negate pole point

  assert ((inner_prod(lap_diff[:, :, :, 0], lap_diff[:, :, :, 0], grid) +
           inner_prod(lap_diff[:, :, :, 1], lap_diff[:, :, :, 1], grid)) < 1e-5)
