from spherical_spectral_element.equiangular_metric import create_quasi_uniform_grid
from .vertical_grids import cam30
from spherical_spectral_element.theta_l.vertical_coordinate import (create_vertical_grid,
                                                                    mass_from_coordinate_midlev,
                                                                    mass_from_coordinate_interface)
from spherical_spectral_element.theta_l.init_model import z_from_p_monotonic, init_model_p_hydro
from spherical_spectral_element.theta_l.constants import init_config
from spherical_spectral_element.config import jnp
def test_z_p_func():
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                 cam30["hybrid_b_i"],
                                 cam30["p0"])
  config = init_config()
  pressures = jnp.linspace(config["p0"], 100, 10)
  T0 = 300
  def p_given_z(z):
    return config["p0"] * jnp.exp(-config["gravity"] * z /
                                  (config["Rgas"] * T0))
  def z_given_p(p):
    return -T0 * config["Rgas"] / config["gravity"] * jnp.log(p / config["p0"])
  heights = z_from_p_monotonic(pressures, p_given_z, eps=1e-10, z_top=80e3)
  assert(jnp.max(jnp.abs(heights - z_given_p(pressures))) < 1e-5)


def test_init():
  nx = 15
  nlev = 5
  grid, dims = create_quasi_uniform_grid(nx)
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                 cam30["hybrid_b_i"],
                                 cam30["p0"])
  def z_pi_surf_func():
    pass
  def Q(lat, lon, z):
    return 0.0
  #model_state, tracer_state = init_model_p_hydro(z_pi_surf_func, p_func, Tv_func, u_func, v_func, Q_funcs, h_grid, v_grid, config, dims):
