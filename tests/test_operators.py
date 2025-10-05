from .context import spherical_spectral_element
from spherical_spectral_element.config import np, npt
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.spectral import deriv
from spherical_spectral_element.equiangular_metric import gen_metric_terms_equiangular, generate_metric_terms, gen_metric_from_topo
from spherical_spectral_element.mesh import gen_bilinear_grid
from spherical_spectral_element.assembly import init_dss_matrix, dss_scalar_for, dss_scalar
from spherical_spectral_element.operators import sphere_gradient, sphere_divergence, sphere_vorticity, inner_prod, sph_to_contra, sphere_gradient


def test_vector_identites():
  nx = 15
  face_connectivity, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  metrics = gen_metric_from_topo(face_connectivity, face_position, face_position_2d, vert_redundancy)
  gll_latlon, gll_to_sphere_jacobian, sphere_to_gll_jacobian, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)


  fn = np.cos(gll_latlon[:, :, :, 1]) * np.cos(gll_latlon[:, :, :, 0])
  grad = sphere_gradient(fn, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)
  vort = sphere_vorticity(grad, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)
  iprod_vort = inner_prod(vort, vort, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet) 
  assert(np.allclose(iprod_vort, np.zeros_like(vort)))
  v = np.zeros_like(gll_latlon)
  v[:,:,:,0] = np.cos(gll_latlon[:, :, :, 0])
  v[:,:,:,1] = np.cos(gll_latlon[:, :, :, 0])
  u = np.zeros_like(gll_latlon)
  u[:,:,:,0] = np.cos(2*gll_latlon[:, :, :, 0])
  u[:,:,:,1] = np.cos(2*gll_latlon[:, :, :, 0])

  #v_cov = sph_to_cov(v, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)
  grad = sphere_gradient(fn, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)
  discrete_divergence_thm = (inner_prod(v[:,:,:,0], grad[:,:,:,0], sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet) + 
                             inner_prod(v[:,:,:,1], grad[:,:,:,1], sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet) - 
                             inner_prod(fn, sphere_divergence(v, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet), sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet))
  assert(np.allclose(discrete_divergence_thm, np.zeros_like(discrete_divergence_thm)))

def test_analytic_soln():
  nx = 61
  face_connectivity, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  metrics = gen_metric_from_topo(face_connectivity, face_position, face_position_2d, vert_redundancy)
  gll_latlon, gll_to_sphere_jacobian, sphere_to_gll_jacobian, rmetdet, metdet, mass_mat, inv_mass_mat, vert_redundancy_gll = metrics
  dss_matrix = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll)

  fn = np.cos(gll_latlon[:, :, :, 1]) * np.cos(gll_latlon[:, :, :, 0])
  grad_f_numerical = sphere_gradient(fn, sphere_to_gll_jacobian, gll_to_sphere_jacobian, metdet, rmetdet)

  sph_grad_lat = -np.cos(gll_latlon[:, :, :, 1]) * np.sin(gll_latlon[:, :, :, 0])
  sph_grad_lon = -np.sin(gll_latlon[:, :, :, 1])
  print(np.max(np.abs(sph_grad_lat- grad_f_numerical[:,:,:,0])))
  print(np.max(np.abs(sph_grad_lon- grad_f_numerical[:,:,:,1])))
  assert(np.max(np.abs(sph_grad_lat- grad_f_numerical[:,:,:,0])) < 1e-5)
  assert(np.max(np.abs(np.allclose(sph_grad_lon, grad_f_numerical[:,:,:,1]))) < 1e-5)

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

