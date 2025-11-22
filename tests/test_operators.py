from .context import spherical_spectral_element
from spherical_spectral_element.config import np, npt
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.spectral import deriv
from spherical_spectral_element.equiangular_metric import gen_metric_terms_equiangular, generate_metric_terms, gen_metric_from_topo
from spherical_spectral_element.assembly import dss_scalar_for, dss_scalar
from spherical_spectral_element.operators import sphere_gradient, sphere_divergence, sphere_vorticity, inner_prod, sph_to_contra, sphere_gradient, sphere_divergence_wk


def test_vector_identites():
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
  #gll_latlon, gll_to_sphere_jacobian, sphere_to_gll_jacobian, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  #dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)

  fn = np.cos(grid.physical_coords[:, :, :, 1]) * np.cos(grid.physical_coords[:, :, :, 0])
  grad = sphere_gradient(fn, grid)
  vort = sphere_vorticity(grad, grid)
  import matplotlib.pyplot as plt
  plt.figure()
  plt.tricontourf(grid.physical_coords[:, :, :, 1].flatten(), grid.physical_coords[:, :, :, 0].flatten(), vort.flatten())
  plt.colorbar()
  plt.savefig("_figures/vort.pdf")

  iprod_vort = inner_prod(vort, vort, grid) 
  assert(np.allclose(iprod_vort, 0))
  v = np.zeros_like(grid.physical_coords)
  v[:,:,:,0] = np.cos(grid.physical_coords[:, :, :, 0])
  v[:,:,:,1] = np.cos(grid.physical_coords[:, :, :, 0])
  u = np.zeros_like(grid.physical_coords)
  u[:,:,:,0] = np.cos(2*grid.physical_coords[:, :, :, 0])
  u[:,:,:,1] = np.cos(2*grid.physical_coords[:, :, :, 0])

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
  vec = np.zeros_like(grid.physical_coords)
  lat = grid.physical_coords[:,:,:,0]
  lon = grid.physical_coords[:,:,:,1]
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
  #div_wk = sphere_divergence_wk(vec, grid)
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

  fn = np.cos(grid.physical_coords[:, :, :, 1]) * np.cos(grid.physical_coords[:, :, :, 0])
  grad_f_numerical = sphere_gradient(fn, grid)
  lon = grid.physical_coords[:, :, :, 1]
  lat = grid.physical_coords[:, :, :, 0]
  sph_grad_lat = -np.cos(grid.physical_coords[:, :, :, 1]) * np.sin(grid.physical_coords[:, :, :, 0])
  sph_grad_lon = -np.sin(grid.physical_coords[:, :, :, 1])
  print(np.max(np.abs(sph_grad_lat- grad_f_numerical[:,:,:,1])))
  print(np.max(np.abs(sph_grad_lon- grad_f_numerical[:,:,:,0])))
  assert(np.max(np.abs(sph_grad_lat- grad_f_numerical[:,:,:,1])) < 1e-4)
  assert(np.max(np.abs(sph_grad_lon- grad_f_numerical[:,:,:,0])) < 1e-4)

# import matplotlib.pyplot as plt

# if TESTING:
#   fn = np.cos(gll_latlon[:, :, :, 1]) * np.cos(gll_latlon[:, :, :, 0])
#   df_dab = np.zeros((gll_latlon.shape[0], npt, npt, 2))
#   df_dab[:, :, :, 0] = np.einsum("fij,ki->fkj", fn, deriv)
#   df_dab[:, :, :, 1] = np.einsum("fij,kj->fik", fn, deriv)
#   dlatlon_dab = np.zeros((gll_latlon.shape[0], npt, npt, 2, 2))
#   dlatlon_dab[:, :, :, 0] = np.einsum("fijs,ki->fjks", gll_latlon, deriv)
#   dlatlon_dab[:, :, :, 1] = np.einsum("fijs,kj->fiks", gll_latlon, deriv)


#   df_dcart = np.einsum("fijg,fijgp->fijp",df_dab, gll_to_cube_jacobian_inv)
#   df_dlatlon = np.einsum("fijg,fijgs->fijs", df_dab, gll_to_sphere_jacobian_inv)#*(g_weights[np.newaxis, :, np.newaxis, np.newaxis])
#   df_dlatlon[:,:,:,0] = dss_scalar(df_dlatlon[:,:,:,0])
#   df_dlatlon[:,:,:,1] = dss_scalar(df_dlatlon[:,:,:,1])
#   dlatlon_dcart = np.einsum("fijgs,fijgp->fijps", dlatlon_dab, gll_to_cube_jacobian_inv)
#   #print(np.max(np.abs(dlatlon_dcart - cube_to_sphere_jacobian)))
#   print(f"computational derivative vs analytical jacobian: {np.max(dlatlon_dcart - cube_to_sphere_jacobian)}")

#   df_dlat = -np.cos(gll_latlon[:, :, :, 1]) * np.sin(gll_latlon[:, :, :, 0])
#   df_dlon = -np.sin(gll_latlon[:, :, :, 1])

#   #print(f"max df_dx_error: {np.max(np.abs(df_dcart[:, :, :, 0] - df_dx))} max df_dy_error: {np.max(np.abs(df_dcart[:, :, :, 1] - df_dy))} max df_dz_error: {np.max(np.abs(df_dcart[:, :, :, 2] - df_dz))}")
#   print(f"max df_dlat error: {np.max(np.abs(df_dlatlon[:, :, :, 0] - df_dlat))} max df_dlon_error: {np.max(np.abs(df_dlatlon[:, :, :, 1] - df_dlon))}")
#   print(f"min df_dlat error: {np.min(np.abs(df_dlatlon[:, :, :, 0] - df_dlat))} min df_dlon_error: {np.min(np.abs(df_dlatlon[:, :, :, 1] - df_dlon))}")
#   print(df_dcart.shape)
#   print(df_dlatlon.shape)

# i_plot = np.arange(0, npt).reshape((1, -1, 1)) * np.ones_like(gll_latlon[:, :, :, 0])
# j_plot = np.arange(0, npt).reshape((1, 1, -1)) * np.ones_like(gll_latlon[:, :, :, 0])
# st = 0
# nd = 4
# cmap="jet"
# if TESTING:
#   start_i = st
#   end_i = nd
#   start_j = st
#   end_j = nd
#   plt.figure()
#   plt.title("f")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(fn[:, start_i:end_i, start_j:end_j]).flatten(), cmap="jet")
#   plt.colorbar()
#   plt.show()
#   plt.figure()
#   plt.title("df_da")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(df_dab[:, start_i:end_i, start_j:end_j, 0]).flatten(), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()
#   plt.figure()
#   plt.title("df_db")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(df_dab[:, start_i:end_i, start_j:end_j, 0]).flatten(), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()
#   plt.figure()
#   plt.title("df_dlat")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(df_dlatlon[:, start_i:end_i, start_j:end_j, 0]).flatten(), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()
#   plt.figure()
#   plt.title("df_dlat_analytic")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(df_dlat[:, start_i:end_i, start_j:end_j]).flatten(), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()
#   plt.figure()
#   plt.title("df_dlon")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(df_dlatlon[:, start_i:end_i, start_j:end_j, 1]).flatten(), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()
#   plt.figure()
#   plt.title("df_dlon_analytic")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(df_dlon[:, start_i:end_i, start_j:end_j]).flatten(), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()
#   plt.figure()
#   plt.title("vort grad f")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(vort[:, start_i:end_i, start_j:end_j]).flatten(), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()

#   plt.title("log lat err")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=np.log(np.abs((df_dlatlon[:, start_i:end_i, start_j:end_j, 0]-df_dlat[:, start_i:end_i, start_j:end_j]).flatten())), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()
#   plt.figure()
#   plt.title("log lon err")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=np.log(np.abs((df_dlatlon[:, start_i:end_i, start_j:end_j, 1]-df_dlon[:, start_i:end_i, start_j:end_j]).flatten())), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()
#   plt.title("lat err")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(df_dlatlon[:, start_i:end_i, start_j:end_j, 0]-df_dlat[:, start_i:end_i, start_j:end_j]).flatten(), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()
#   plt.show()
#   plt.figure()
#   plt.title("lon err")
#   plt.scatter(gll_latlon[:, start_i:end_i, start_j:end_j, 1].flatten(), gll_latlon[:, start_i:end_i, start_j:end_j, 0].flatten(), c=(df_dlatlon[:, start_i:end_i, start_j:end_j, 1]-df_dlon[:, start_i:end_i, start_j:end_j]).flatten(), alpha=0.5, s=np.random.uniform(size=gll_latlon[:, start_i:end_i, start_j:end_j, 1].size, high=50, low=10), cmap=cmap)
#   plt.colorbar()

