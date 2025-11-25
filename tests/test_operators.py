from .context import spherical_spectral_element
from spherical_spectral_element.config import np, npt
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.spectral import deriv
from spherical_spectral_element.equiangular_metric import gen_metric_terms_equiangular, generate_metric_terms, gen_metric_from_topo
from spherical_spectral_element.assembly import dss_scalar_for, dss_scalar
from spherical_spectral_element.operators import sphere_gradient, sphere_divergence, sphere_vorticity, inner_prod, sph_to_contra, sphere_gradient, sphere_divergence_wk, sphere_gradient_wk_cov, sphere_vec_laplacian_wk


def test_vector_identites():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  #gll_latlon, gll_to_sphere_jacobian, sphere_to_gll_jacobian, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  #dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)

  fn = np.cos(grid["physical_coords"][:, :, :, 1]) * np.cos(grid["physical_coords"][:, :, :, 0])
  grad = sphere_gradient(fn, grid)
  vort = sphere_vorticity(grad, grid)
  import matplotlib.pyplot as plt
  plt.figure()
  plt.tricontourf(grid["physical_coords"][:, :, :, 1].flatten(), grid["physical_coords"][:, :, :, 0].flatten(), vort.flatten())
  plt.colorbar()
  plt.savefig("_figures/vort.pdf")

  iprod_vort = inner_prod(vort, vort, grid) 
  assert(np.allclose(iprod_vort, 0))
  v = np.zeros_like(grid["physical_coords"])
  v[:,:,:,0] = np.cos(grid["physical_coords"][:, :, :, 0])
  v[:,:,:,1] = np.cos(grid["physical_coords"][:, :, :, 0])
  u = np.zeros_like(grid["physical_coords"])
  u[:,:,:,0] = np.cos(2*grid["physical_coords"][:, :, :, 0])
  u[:,:,:,1] = np.cos(2*grid["physical_coords"][:, :, :, 0])

  #v_cov = sph_to_cov(v, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)
  grad = sphere_gradient(fn, grid)
  discrete_divergence_thm = (inner_prod(v[:,:,:,0], grad[:,:,:,0], grid) + 
                             inner_prod(v[:,:,:,1], grad[:,:,:,1], grid) - 
                             inner_prod(fn, sphere_divergence(v, grid), grid))
  assert(np.allclose(discrete_divergence_thm, np.zeros_like(discrete_divergence_thm)))

def test_divergence():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
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
  div = dss_scalar(sphere_divergence(vec, grid), grid)
  div_wk = dss_scalar(sphere_divergence_wk(vec, grid),grid, scaled=False) 
  vort = dss_scalar(sphere_vorticity(vec, grid), grid)
  assert(inner_prod(div_wk - div, div_wk - div, grid) < 1e-5)
  assert(inner_prod(div_analytic - div, div_analytic - div, grid) < 1e-5)
  assert(inner_prod(vort_analytic - vort, vort_analytic - vort, grid) < 1e-5)

def test_analytic_soln():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  #gll_latlon, gll_to_sphere_jacobian, sphere_to_gll_jacobian, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  #dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)

  fn = np.cos(grid["physical_coords"][:, :, :, 1]) * np.cos(grid["physical_coords"][:, :, :, 0])
  grad_f_numerical = sphere_gradient(fn, grid)
  lon = grid["physical_coords"][:, :, :, 1]
  lat = grid["physical_coords"][:, :, :, 0]
  sph_grad_wk = sphere_gradient_wk_cov(fn, grid)
  sph_grad_wk[:,:,:,0] = dss_scalar(sph_grad_wk[:,:,:,0], grid, scaled=False)
  sph_grad_wk[:,:,:,1] = dss_scalar(sph_grad_wk[:,:,:,1], grid, scaled=False)
  grad_diff = sph_grad_wk - grad_f_numerical
  #import matplotlib.pyplot as plt
  #plt.figure()
  #plt.scatter(lon.flatten(), lat.flatten(), c=sph_grad_wk[:,:,:,0].flatten(), s=0.01)
  #i = 1
  #j = 1
  #plt.scatter(lon[:,i,j].flatten(), lat[:,i,j].flatten(), c=sph_grad_wk[:,i,j,0].flatten(), s=0.01)

  #plt.colorbar()
  #plt.savefig("_figures/grad_wk_test.pdf")
  sph_grad_lat = -np.cos(grid["physical_coords"][:, :, :, 1]) * np.sin(grid["physical_coords"][:, :, :, 0])
  sph_grad_lon = -np.sin(grid["physical_coords"][:, :, :, 1])
  assert((inner_prod(grad_diff[:,:,:,0], grad_diff[:,:,:,0], grid)+
          inner_prod(grad_diff[:,:,:,1], grad_diff[:,:,:,1], grid) ) < 1e-5)
  assert(np.max(np.abs(sph_grad_lat- grad_f_numerical[:,:,:,1])) < 1e-4)
  assert(np.max(np.abs(sph_grad_lon- grad_f_numerical[:,:,:,0])) < 1e-4)

def test_vector_laplacian():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  #gll_latlon, gll_to_sphere_jacobian, sphere_to_gll_jacobian, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  #dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)
  v = np.zeros_like(grid["physical_coords"])
  v[:,:,:,0] = np.cos(grid["physical_coords"][:, :, :, 0])
  v[:,:,:,1] = np.cos(grid["physical_coords"][:, :, :, 0])
  lon = grid["physical_coords"][:, :, :, 1]
  lat = grid["physical_coords"][:, :, :, 0]
  laplace_v_wk = sphere_vec_laplacian_wk(v, grid)
  laplace_v_wk[:,:,:,0] = dss_scalar(laplace_v_wk[:,:,:,0], grid, scaled=False)
  laplace_v_wk[:,:,:,1] = dss_scalar(laplace_v_wk[:,:,:,1], grid, scaled=False)
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
  v[:,:,:,0] = np.cos(grid["physical_coords"][:, :, :, 0])**2
  v[:,:,:,1] = np.cos(grid["physical_coords"][:, :, :, 0])**2
  laplace_v_wk = sphere_vec_laplacian_wk(v, grid)
  laplace_v_wk[:,:,:,0] = dss_scalar(laplace_v_wk[:,:,:,0], grid, scaled=False)
  laplace_v_wk[:,:,:,1] = dss_scalar(laplace_v_wk[:,:,:,1], grid, scaled=False)
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



