from .context import spherical_spectral_element
from spherical_spectral_element.config import np, npt, jax_wrapper, jax_unwrapper, jnp, eps
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.spectral import deriv
from spherical_spectral_element.equiangular_metric import gen_metric_terms_equiangular, generate_metric_terms, gen_metric_from_topo
from spherical_spectral_element.assembly import dss_scalar_for, dss_scalar
from spherical_spectral_element.operators import sphere_gradient, sphere_divergence, sphere_vorticity, inner_prod, sph_to_contra, sphere_gradient, sphere_divergence_wk, sphere_gradient_wk_cov, sphere_vec_laplacian_wk


def test_vector_identites():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)

  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  grad = sphere_gradient(fn, grid)
  vort = sphere_vorticity(grad, grid)

  iprod_vort = inner_prod(vort, vort, grid) 
  assert(jnp.allclose(iprod_vort, 0, atol=eps))
  v = np.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
  u = np.stack((jnp.cos(2*grid["physical_coords"][:, :, :, 0]),
                jnp.cos(2*grid["physical_coords"][:, :, :, 0])), axis=-1)

  grad = sphere_gradient(fn, grid)
  discrete_divergence_thm = (inner_prod(v[:,:,:,0], grad[:,:,:,0], grid) + 
                             inner_prod(v[:,:,:,1], grad[:,:,:,1], grid) - 
                             inner_prod(fn, sphere_divergence(v, grid), grid))
  assert(jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))

def test_divergence():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  vec = np.zeros_like(grid["physical_coords"])
  lat = grid["physical_coords"][:,:,:,0]
  lon = grid["physical_coords"][:,:,:,1]
  vec[:,:,:,0] = np.cos(lat)**2 * np.cos(lon)**3
  vec[:,:,:,1] = np.cos(lat)**2 * np.cos(lon)**3

  vort_analytic = (-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) +
                   3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3)

  div_analytic = (-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) -
                  3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3)
  #vec = np.zeros_like(grid.physical_coords)
  #lat = grid.physical_coords[:,:,:,0]
  #lon = grid.physical_coords[:,:,:,1]
  ##vec[:,:,:,0] = np.cos(lat)
  #vec[:,:,:,1] = 0.0
  div = dss_scalar(sphere_divergence(vec, grid), grid, dims)
  div_wk = dss_scalar(sphere_divergence_wk(vec, grid),grid, dims, scaled=False) 
  vort = dss_scalar(sphere_vorticity(vec, grid), grid, dims)
  assert(inner_prod(div_wk - div, div_wk - div, grid) < 1e-5)
  assert(inner_prod(div_analytic - div, div_analytic - div, grid) < 1e-5)
  assert(inner_prod(vort_analytic - vort, vort_analytic - vort, grid) < 1e-5)

def test_analytic_soln():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  #gll_latlon, gll_to_sphere_jacobian, sphere_to_gll_jacobian, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  #dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)

  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  grad_f_numerical = sphere_gradient(fn, grid)
  lon = grid["physical_coords"][:, :, :, 1]
  lat = grid["physical_coords"][:, :, :, 0]
  sph_grad_wk = sphere_gradient_wk_cov(fn, grid)
  sph_grad_wk = jnp.stack((dss_scalar(sph_grad_wk[:,:,:,0], grid, dims, scaled=False),
                           dss_scalar(sph_grad_wk[:,:,:,1], grid, dims, scaled=False)), axis=-1)
  grad_diff = sph_grad_wk - grad_f_numerical
  #import matplotlib.pyplot as plt
  #plt.figure()
  #plt.scatter(lon.flatten(), lat.flatten(), c=sph_grad_wk[:,:,:,0].flatten(), s=0.01)
  #i = 1
  #j = 1
  #plt.scatter(lon[:,i,j].flatten(), lat[:,i,j].flatten(), c=sph_grad_wk[:,i,j,0].flatten(), s=0.01)

  #plt.colorbar()
  #plt.savefig("_figures/grad_wk_test.pdf")
  sph_grad_lat = -jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.sin(grid["physical_coords"][:, :, :, 0])
  sph_grad_lon = -jnp.sin(grid["physical_coords"][:, :, :, 1])
  assert((inner_prod(grad_diff[:,:,:,0], grad_diff[:,:,:,0], grid)+
          inner_prod(grad_diff[:,:,:,1], grad_diff[:,:,:,1], grid) ) < 1e-5)
  assert(np.max(np.abs(sph_grad_lat- grad_f_numerical[:,:,:,1])) < 1e-4)
  assert(np.max(np.abs(sph_grad_lon- grad_f_numerical[:,:,:,0])) < 1e-4)

def test_vector_laplacian():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  #gll_latlon, gll_to_sphere_jacobian, sphere_to_gll_jacobian, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  #dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)
  v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                          jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
  lon = grid["physical_coords"][:, :, :, 1]
  lat = grid["physical_coords"][:, :, :, 0]
  laplace_v_wk = sphere_vec_laplacian_wk(v, grid)
  laplace_v_wk = jnp.stack((dss_scalar(laplace_v_wk[:,:,:,0], grid, dims, scaled=False),
                 dss_scalar(laplace_v_wk[:,:,:,1], grid, dims, scaled=False)), axis=-1)
  import matplotlib.pyplot as plt
  #plt.figure()
  #plt.scatter(lon.flatten(), lat.flatten(), c=laplace_v_wk[:,:,:,0].flatten(), s=0.01)
  #i = 1
  #j = 1
  #plt.scatter(lon[:,i,j].flatten(), lat[:,i,j].flatten(), c=laplace_v_wk[:,i,j,0].flatten(), s=0.01)
  #plt.colorbar()
  #plt.savefig("_figures/lap_wk_test.pdf")
  lap_diff = laplace_v_wk + 2 * v
  assert((inner_prod(lap_diff[:,:,:,0], lap_diff[:,:,:,0], grid)+
          inner_prod(lap_diff[:,:,:,1], lap_diff[:,:,:,1], grid) ) < 1e-5)
  v = jnp.stack((np.cos(grid["physical_coords"][:, :, :, 0])**2,
                 np.cos(grid["physical_coords"][:, :, :, 0])**2), axis=-1)
  laplace_v_wk = sphere_vec_laplacian_wk(v, grid)
  laplace_v_wk = jnp.stack((dss_scalar(laplace_v_wk[:,:,:,0], grid, dims, scaled=False),
                            dss_scalar(laplace_v_wk[:,:,:,1], grid, dims, scaled=False)), axis=-1)
  lap_diff = laplace_v_wk  + 3 * (np.cos(2 * grid["physical_coords"][:, :, :, 0]))[:,:,:,np.newaxis]
  lap_diff *= np.cos(grid["physical_coords"][:, :, :, 0])[:,:,:,np.newaxis]**2 #hack to negate pole point
  # plt.figure()
  # plt.scatter(lon.flatten(), lat.flatten(), c=lap_diff[:,:,:,0].flatten(), s=0.01)
  #i = 1
  #j = 1
  #plt.scatter(lon[:,i,j].flatten(), lat[:,i,j].flatten(), c=laplace_v_wk[:,i,j,0].flatten(), s=0.01)
  # plt.colorbar()
  # plt.savefig("_figures/lap_wk_test.pdf")
  assert((inner_prod(lap_diff[:,:,:,0], lap_diff[:,:,:,0], grid)+
          inner_prod(lap_diff[:,:,:,1], lap_diff[:,:,:,1], grid) ) < 1e-5)



