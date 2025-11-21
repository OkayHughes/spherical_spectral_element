from .context import spherical_spectral_element
from spherical_spectral_element.config import npt, np, DEBUG
from spherical_spectral_element.constants import gravit, rearth
from spherical_spectral_element.shallow_water.model import shallow_water_model
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.equiangular_metric import gen_metric_from_topo
from spherical_spectral_element.operators import inner_prod
from os import makedirs
from os.path import join
if DEBUG:
  import matplotlib.pyplot as plt

def test_sw_model():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)

  model = shallow_water_model(nx, grid, alpha=np.pi/2)
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

