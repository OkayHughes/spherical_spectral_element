from ..config import jnp
from .model_state import init_model_struct
from .infra import get_delta


def z_from_p_monotonic(pressures, p_given_z, eps=1e-5, z_top=80e3):
  z_guesses = p_given_z(z_top * jnp.ones_like(pressures))
  not_converged = jnp.logical_not(jnp.abs((p_given_z(z_guesses) - pressures)) / pressures < eps)
  frac = 0.5
  while jnp.any(not_converged):
    p_guess = p_given_z(z_guesses)
    too_high = p_guess < pressures
    z_guesses = jnp.where(not_converged,
                          jnp.where(jnp.logical_and(not_converged, too_high),
                                    z_guesses - frac * z_top,
                                    z_guesses + frac * z_top), z_guesses)
    not_converged = jnp.logical_not(jnp.abs((p_given_z(z_guesses) - pressures)) / pressures < eps)
    frac *= 0.5
  return z_guesses


def init_model_p_hydro(z_pi_surf_func, p_func, Tv_func, u_func, v_func, Q_funcs, h_grid, v_grid, config, dims):
  lat = h_grid["physical_coords"][:, :, :, 0]
  lon = h_grid["physical_coords"][:, :, :, 0]
  z_surf, pi_surf = z_pi_surf_func(lat, lon)
  pressures_mid = mass_from_coordinate_midlev(pi_surf, v_grid)
  pressures_int = mass_from_coordinate_interface(pi_surf, v_grid)
  z_mid = p_from_z_monotonic(pressures_mid, p_func)
  dpi = get_delta(pressures_int)
  w_i = 0.0
  phi_i = 0.0
  phi_surf = z_surf * config["gravity"]
  vtheta = Tv_func(lat, lon, z_mid) * (p0/p_mid)**(config["Rgas"]/config["cp"])
  vtheta_dpi = vtheta * dpi
  initial_state = init_model_struct(jnp.stack((u, v), axis=-1),
                                    vtheta_dpi,
                                    dpi,
                                    phi_surf,
                                    phi_i,
                                    w_i,
                                    h_grid,
                                    dims,
                                    config)
  tracer_struct = init_tracer_struct(Q_funcs(lat, lon, z_mid))
  return initial_state, tracer_struct
