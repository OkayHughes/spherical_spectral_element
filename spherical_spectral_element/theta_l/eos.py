from ..config import jnp
from infra import z_from_phi, model_to_interface, get_delta
from constants import Rgas, p0

def r_hat_sq_avg(r_hat_i):
  r_hat_sq_avg = (r_hat_i[:, :, :, :-1] * r_hat_i[:, :, :, 1:]+
              r_hat_i[:, :, :, :-1] * r_hat_i[:, :, :, :-1]+
              r_hat_i[:, :, :, 1:] * r_hat_i[:, :, :, 1:]) / 3.0
  return r_hat_sq_avg

def p_exner_nonhydrostatic(vtheta_dpi, dphi, r_hat_sq_avg, config):
  pnh_over_exner = - config["Rgas"]*state["vtheta_dpi"] / dphi
  pnh_over_exner /= r_hat_sq_avg
  pnh = p0 * (p_over_exner / config["p0"])**(1.0/(1.0-config["Rgas"]/config["cp"]))
  return pnh, pnh/p_over_exner


def get_mu(state, phi_i, v_grid, config, deep=False, hydrostatic=True):
  # note: assumes that phi_i is in hydrostatic balance.
  vtheta_dpi = state["vtheta_dpi"]
  dphi = get_delta(phi_i)
  if deep:
    r_hat_i = r_hat_from_phi(phi_i, config, deep=deep)
    r_hat_sq_avg = r_hat_sq_avg(r_hat_i)
  else:
    r_hat_i = 1.0
    r_hat_sq_avg = 1.0
  p_model, exner = p_exner_nonhydrostatic(vtheta_dpi, dphi, config)
  if hydrostatic:
    dpnh_dpi = 1.0
  else:
    p_top = v_grid["hybrid_a_i"][0] * v_grid["reference_pressure"]
    if deep:
      p_top /= r_hat_i[:, :, :, 0]**2
    dpi_i = model_to_interface(state["dpi"])
    dpnh_dpi_top = 2 * (p_model[:, :, :, 0] - p_top)/dp3d_i[:, :, :, 0]
    dpnh_dpi_bottom = jnp.ones_like(p_model[:, :, :, 0])
    dpnh_dpi_int = (p_model[:, :, :, :-1] - p_model[:, :, :, 1:])/dp3d_i[:, :, :, 1:-1]
    dpnh_dpi = np.stack((dpnh_dpi_top,
                        dpnh_dpi_int,
                        dpnh_dpi_bottom), axis=-1)
    if deep:
      dpnh_dpi *= r_hat_i**2
  return p_model, exner, r_hat_i, dpnh_dpi

def get_p_mid(state, v_grid, config):
  p = jnp.cumsum(state["dp3d"], axis=-1) + v_grid["hybrid_a_i"][0] * v_grid["reference_pressure"]
  p -= 0.5 * state["dp3d"]
  return p


def get_balanced_phi(state, v_grid, config):
  p = get_p_mid(state, v_grid)
  dphi = -config["Rgas"] * (state["vtheta_dpi"] *
                            (p / config["p0"])**(config["Rgas"] / config["cp"]-1.0) / config["p0"])
  dphi_augment = np.stack((dphi[:, :, :, 0] + state["phi_surf"], dphi[:, :, :, :-1:-1]))
  phi_i_above_surf = np.cumsum(dphi_augment, axis=-1)
  return jnp.flip(np.stack((state["phi_surf"], phi_i_above_surf), axis=-1), axis=-1)


