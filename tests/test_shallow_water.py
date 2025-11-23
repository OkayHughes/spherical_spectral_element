from .context import spherical_spectral_element
from spherical_spectral_element.config import npt, np, DEBUG
from spherical_spectral_element.constants import gravit, rearth
from spherical_spectral_element.shallow_water.model import shallow_water_model
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.equiangular_metric import gen_metric_from_topo
from spherical_spectral_element.operators import inner_prod, sphere_vorticity
from spherical_spectral_element.assembly import dss_scalar
from os import makedirs
from os.path import join
if DEBUG:
  import matplotlib.pyplot as plt

def test_sw_model():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)

  model = shallow_water_model(nx, grid, alpha=np.pi/4, diffusion=False)
  u0 = 2.0 * np.pi * model.radius_earth / (12.0 * 24.0 * 60.0 * 60.0)
  h0 = 2.94e4/model.gravity
  def williamson_tc2_u(lat, lon):
    wind = np.zeros((*lat.shape, 2))
    wind[:,:,:,0] =  u0 * (np.cos(lat) * np.cos(model.alpha) + np.cos(lon) * np.sin(lat) * np.sin(model.alpha))
    wind[:,:,:,1] =  -u0 * (np.sin(lon) * np.sin(model.alpha))
    return wind
  def williamson_tc2_h(lat, lon):
    h = np.zeros_like(lat)
    h += h0
    h -= (model.radius_earth * model.earth_period * u0 + u0**2/2.0)/model.gravity * (
                    -np.cos(lon)*np.cos(lat)*np.sin(model.alpha) + np.sin(lat) * np.cos(model.alpha))**2
    return h
  def williamson_tc2_hs(lat, lon):
    return np.zeros_like(lat)
  T = 4000.0
  t, final_state = model.simulate(T, williamson_tc2_u, williamson_tc2_h, williamson_tc2_hs)
  print(final_state["u"].dtype)
  u_init = williamson_tc2_u(grid.physical_coords[:,:,:,0], grid.physical_coords[:,:,:,1])
  h_init = williamson_tc2_h(grid.physical_coords[:,:,:,0], grid.physical_coords[:,:,:,1])
  diff_u = u_init - final_state["u"]
  diff_h = h_init - final_state["h"]
  assert(inner_prod(diff_u[:,:,:,0], diff_u[:,:,:,0], grid) < 1e-5)
  assert(inner_prod(diff_u[:,:,:,1], diff_u[:,:,:,1], grid) < 1e-5)
  assert(inner_prod(diff_h, diff_h, grid)/np.max(h_init) < 1e-5)
  if DEBUG:
    fig_dir = "_figures"
    makedirs(fig_dir, exist_ok=True)
    plt.figure()
    plt.title("U at time {t}")
    lon = grid.physical_coords[:,:,:,1]
    lat = grid.physical_coords[:,:,:,0]
    plt.tricontourf(lon.flatten(), lat.flatten(), final_state["u"][:,:,:,0].flatten())
    plt.colorbar()
    plt.savefig(join(fig_dir, "U_final.pdf"))
    plt.figure()
    plt.title("V at time {t}")
    plt.tricontourf(lon.flatten(), lat.flatten(), final_state["u"][:,:,:,1].flatten())
    plt.colorbar()
    plt.savefig(join(fig_dir, "V_final.pdf"))
    plt.figure()
    plt.title("h at time {t}")
    plt.tricontourf(lon.flatten(), lat.flatten(), final_state["h"].flatten())
    plt.colorbar()
    plt.savefig(join(fig_dir, "h_final.pdf"))


def test_galewsky():
  nx = 61
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)

  model = shallow_water_model(nx, grid, diffusion=True)

  deg = 100
  pts, weights = np.polynomial.legendre.leggauss(deg)
  pts = (pts+1.0)/2.0
  weights /= 2.0
  u_max = 80
  gravit = model.gravity
  phi0 = np.pi/7
  phi1 = np.pi/2 - phi0
  e_norm = np.exp(-4 /(phi1 - phi0)**2)
  a = model.radius_earth
  Omega = model.earth_period
  h0 = 1e4
  hat_h = 120.0
  alpha = 1.0/3.0
  beta = 1.0/15.0
  pert_center = np.pi/4

  def galewsky_u(lat):
    u = np.zeros_like(lat)
    mask = np.logical_and(lat > phi0, lat < phi1)
    u[mask] = u_max/e_norm * np.exp(1/((lat[mask]-phi0)*(lat[mask]-phi1)))
    return u
  def galewsky_wind(lat, lon):
    u = np.zeros([*lat.shape, 2])
    u[:, :, :, 0] = galewsky_u(lat)
    return u
  def galewsky_h(lat, lon):
    quad_amount = lat+np.pi/2.0
    weights_quad = quad_amount.reshape([*lat.shape, 1]) * weights.reshape((*[1 for _ in lat.shape], deg))
    phi_quad = quad_amount.reshape([*lat.shape, 1]) * pts.reshape((*[1 for _ in lat.shape], deg)) - np.pi/2
    u_quad = galewsky_u(phi_quad)
    f = 2.0 * Omega * np.sin(phi_quad)
    integrand = a * u_quad * (f + np.tan(phi_quad)/a * u_quad)
    h = h0 - 1.0/gravit * np.sum(integrand * weights_quad, axis=-1)
    h_prime = hat_h * np.cos(lat)* np.exp(-(lon/alpha)**2)*np.exp(-((pert_center-lat)/beta)**2)

    return h +  h_prime
  def galewsky_hs(lat, lon):
    return np.zeros_like(lat)
  #T = 18000.0
  T = (144 * 3600)
  t, final_state = model.simulate(T, galewsky_wind, galewsky_h, galewsky_hs)
  assert(not np.any(np.isnan(final_state["u"])))
  if DEBUG:
    fig_dir = "_figures"
    makedirs(fig_dir, exist_ok=True)
    lon = grid.physical_coords[:,:,:,1]
    lat = grid.physical_coords[:,:,:,0]
    levels = np.arange(-10+1e-4, 101, 10)
    vort = dss_scalar(sphere_vorticity(final_state["u"], grid, a=model.radius_earth), grid)  
    plt.figure()
    plt.title(f"U at time {t}s")
    #levels = None
    plt.tricontourf(lon.flatten(), lat.flatten(), final_state["u"][:,:,:,0].flatten(), levels = levels)
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_U_final.pdf"))
    plt.figure()
    plt.title(f"V at time {t}s")
    plt.tricontourf(lon.flatten(), lat.flatten(), final_state["u"][:,:,:,1].flatten())
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_V_final.pdf"))
    plt.figure()
    plt.title(f"h at time {t}s")
    plt.tricontourf(lon.flatten(), lat.flatten(), final_state["h"].flatten())
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_h_final.pdf"))
    plt.figure()
    plt.title(f"vorticity at time {t}s")
    plt.tricontourf(lon.flatten(), lat.flatten(), vort.flatten(), vmin=-0.0002, vmax=0.0002)
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_vort_final.pdf"))
