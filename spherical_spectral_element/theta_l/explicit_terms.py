from ..config import jnp
from .infra import vel_model_to_interface, model_to_interface, interface_to_model, interface_to_model_vec
from .infra import z_from_phi, g_from_z
from .eos import get_mu, get_balanced_phi
from .operators_3d import sphere_gradient_3d, sphere_vorticity_3d, sphere_divergence_3d
from .model_state import wrap_model_struct


def calc_shared_quantities(state, h_grid, v_grid, config, hydrostatic=True, deep=False):
  if hydrostatic:
    phi_i = get_balanced_phi(state, v_grid, config)
  else:
    phi_i = state["phi_i"]
  w_i = state["w_i"]
  dpi = state["dpi"]
  u = state["u"]
  radius_earth = config["radius_earth"]
  period_earth = config["period_earth"]
  lat = h_grid["physical_coordinates"][:, :, :, 0]

  dpi_i = model_to_interface(dpi)
  phi = interface_to_model(phi_i)
  pnh, exner, r_hat_i, mu = get_mu(state, phi_i, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  if deep:
    r_hat_m = interface_to_model(r_hat_i)
    z = z_from_phi(phi_i, config, deep=deep)
    r_m = interface_to_model(z + radius_earth)
    g = g_from_z(z, config, deep=deep)
  else:
    r_hat_m = 1.0
    r_m = radius_earth
    g = config["gravity"]
  fcor = 2.0 * period_earth * jnp.sin(lat)
  fcorcos = 2.0 * period_earth * jnp.cos(lat)
  w_m = interface_to_model(w_i)
  grad_w_i = sphere_gradient_3d(w_i, h_grid, config)
  grad_exner = sphere_gradient_3d(exner, h_grid, config) / r_hat_m
  vtheta = state["vtheta_dpi"] / dpi
  grad_phi_i = sphere_gradient_3d(phi_i, h_grid, config)
  v_over_r_hat_i = model_to_interface(u / r_hat_m[:, :, :, jnp.newaxis])
  div_dp = sphere_divergence_3d(dpi[:, :, :, :, jnp.newaxis] * u /
                                r_hat_m[:, :, :, :, jnp.newaxis], h_grid, config)
  u_i = vel_model_to_interface(u, dpi, dpi_i)
  return (phi_i, phi, dpi_i, pnh, exner,
          r_hat_i, mu, r_hat_m, z, r_m, g,
          fcor, fcorcos, w_m,
          grad_w_i, grad_exner, vtheta,
          grad_phi_i, v_over_r_hat_i,
          div_dp, u_i)


def vorticity_term(u, fcor, r_hat_m, h_grid, config):
  vort = sphere_vorticity_3d(u, h_grid, config)
  vort /= r_hat_m
  vort_term = jnp.stack((u[:, :, :, :, 1] * (fcor[:, :, :, jnp.newaxis, jnp.newaxis] + vort),
                         -u[:, :, :, :, 0] * (fcor[:, :, :, jnp.newaxis, jnp.newaxis] + vort)), axis=-1)
  return vort_term


def grad_kinetic_energy_h_term(u, r_hat_m, h_grid, config):
  grad_kinetic_energy = sphere_gradient_3d((u[:, :, :, :, 0]**2 +
                                            u[:, :, :, :, 1]**2) / 2.0, h_grid, config)
  return -grad_kinetic_energy / r_hat_m


def grad_kinetic_energy_v_term(w_i, r_hat_m, h_grid, config):
  w_sq_m = interface_to_model(w_i * w_i) / 2.0
  w2_grad_sph = sphere_gradient_3d(w_sq_m, h_grid, config) / r_hat_m
  return -w2_grad_sph


def w_vorticity_correction_term(w_i, grad_w_i, r_hat_m):
  w_grad_w_m = interface_to_model_vec(w_i[:, :, :, :, jnp.newaxis] * grad_w_i)
  w_grad_w_m /= r_hat_m[:, :, :, :, jnp.newaxis]
  return w_grad_w_m


def u_metric_term(u, w_m, r_m):
  return -w_m[:, :, :, :, jnp.newaxis] * u / r_m[:, :, :, jnp.newaxis]


def u_nct_term(w_m, fcorcos):
  return -jnp.stack((w_m, jnp.zeros_like(w_m)), axis=-1) * fcorcos


def pgrad_pressure_term(vtheta, grad_exner, exner, r_hat_m, h_grid, config):
  grad_p_term_1 = config["cp"] * vtheta[:, :, :, :, jnp.newaxis] * grad_exner
  grad_vtheta_exner = sphere_gradient_3d(vtheta * exner, h_grid, config) / r_hat_m
  grad_vtheta = sphere_gradient_3d(vtheta, h_grid, config) / r_hat_m
  grad_p_term_2 = config["cp"] * (grad_vtheta_exner - exner[:, :, :, :, jnp.newaxis] * grad_vtheta)
  return -(grad_p_term_1 + grad_p_term_2) / 2.0


def pgrad_phi_term(mu, grad_phi_i, r_hat_m):
  pgf_gradphi_m = interface_to_model_vec(mu[:, :, :, :, jnp.newaxis] * grad_phi_i)
  pgf_gradphi_m /= r_hat_m[:, :, :, :, jnp.newaxis]
  return -pgf_gradphi_m


def w_advection_term(v_over_r_hat_i, grad_w_i):
  v_grad_w_i = (v_over_r_hat_i[:, :, :, 0] * grad_w_i[:, :, :, 0] +
                v_over_r_hat_i[:, :, :, 1] * grad_w_i[:, :, :, 1])
  return -v_grad_w_i


def w_metric_term(u, r_m):
  v_sq_over_r_i = model_to_interface(u / r_m)
  return (v_sq_over_r_i[:, :, :, 0] + v_sq_over_r_i[:, :, :, 1])


def w_nct_term(u_i, fcorcos):
  return -u_i[:, :, :, :, 0] * fcorcos


def w_buoyancy_term(g, mu):
  return -g * (1 - mu)


def phi_advection_term(v_over_r_hat_i, grad_phi_i):
  v_grad_phi_i = (v_over_r_hat_i[:, :, :, 0] * grad_phi_i[:, :, :, 0] +
                  v_over_r_hat_i[:, :, :, 1] * grad_phi_i[:, :, :, 1])
  return -v_grad_phi_i


def phi_acceleration_v_term(g, w_i):
  return g * w_i


def vtheta_divergence_term(u, vtheta_dpi, vtheta, div_dp, dpi, r_hat_m, h_grid, config):
  v_vtheta = u * vtheta_dpi[:, :, :, :, jnp.newaxis]
  v_vtheta /= r_hat_m
  div_v_vtheta = sphere_divergence_3d(v_vtheta, h_grid, config) / 2.0
  grad_vtheta = sphere_gradient_3d(vtheta, h_grid, config)
  grad_vtheta /= r_hat_m

  div_v_vtheta += (vtheta * div_dp + (dpi * (u[:, :, :, :, 0] * grad_vtheta[:, :, :, :, 0] +
                                             u[:, :, :, :, 1] * grad_vtheta[:, :, :, :, 1]))) / 2.0
  return -div_v_vtheta


def dpi_divergence_term(div_dp):
  return -div_dp


def explicit_tend(state, h_grid, v_grid, config, hydrostatic=True, deep=False):
  dpi = state["dpi"]
  u = state["u"]
  w_i = state["w_i"]
  vtheta_dpi = state["vtheta_dpi"]

  (phi_i, phi, dpi_i, pnh, exner,
   r_hat_i, mu, r_hat_m, z, r_m, g,
   fcor, fcorcos, w_m,
   grad_w_i, grad_exner, vtheta,
   grad_phi_i, v_over_r_hat_i,
   div_dp, u_i) = calc_shared_quantities(state, h_grid,
                                         v_grid, config,
                                         hydrostatic=hydrostatic,
                                         deep=deep)

  u_tend = (vorticity_term(u, fcor, r_hat_m, h_grid, config) +
            grad_kinetic_energy_h_term(u, r_hat_m, h_grid, config) +
            pgrad_pressure_term(vtheta, grad_exner, exner, r_hat_m, h_grid, config) +
            pgrad_phi_term(mu, grad_phi_i, r_hat_m))
  if not hydrostatic:
    u_tend += (grad_kinetic_energy_v_term(w_i, r_hat_m, h_grid, config) +
               w_vorticity_correction_term(w_i, grad_w_i, r_hat_m))
    if deep:
      u_tend += (u_metric_term(u, w_m, r_m) +
                 u_nct_term(w_m, fcorcos))
  if not hydrostatic:
    w_tend = (w_advection_term(v_over_r_hat_i, grad_w_i) +
              w_buoyancy_term(g, mu))
    if deep:
      w_tend += (w_metric_term(u, r_m) +
                 w_nct_term(u_i, fcorcos))
  else:
    w_tend = 0.0
  if not hydrostatic:
    phi_tend = (phi_advection_term(v_over_r_hat_i, grad_phi_i) +
                phi_acceleration_v_term(g, w_i))
  else:
    phi_tend = 0.0

  vtheta_dpi_tend = vtheta_divergence_term(u, vtheta_dpi, vtheta, div_dp, dpi, r_hat_m, h_grid, config)
  dpi_tend = dpi_divergence_term(div_dp)
  return wrap_model_struct(u_tend, vtheta_dpi_tend, dpi_tend, state["phi_surf"], phi_tend, w_tend)


def calc_energy_quantities(state, h_grid, v_grid, config, deep=False):
  (phi_i, phi, dpi_i, pnh, exner,
   r_hat_i, mu, r_hat_m, z, r_m, g,
   fcor, fcorcos, w_m,
   grad_w_i, grad_exner, vtheta,
   grad_phi_i, v_over_r_hat_i,
   div_dp, u_i) = calc_shared_quantities(state, h_grid,
                                         v_grid, config,
                                         hydrostatic=False,
                                         deep=deep)
  dpi_i_integral = jnp.stack((dpi_i[:, :, :, 0] / 2.0,
                              dpi_i[:, :, :, 1:-1],
                              dpi_i[:, :, :, -1] / 2.0), axis=-1)
  u = state["u"]
  dpi = state["dpi"]
  w_i = state["w_i"]
  vtheta_dpi = state["vtheta_dpi"]
  u1 = u[:, :, :, :, 0]
  u2 = u[:, :, :, :, 0]
  grad_kinetic_energy_h = grad_kinetic_energy_h_term(u, r_hat_m, h_grid, config)
  dpi_divergence = dpi_divergence_term(div_dp)
  phi_acceleration_v = phi_acceleration_v_term(g, w_i)
  w_buoyancy = w_buoyancy_term(g, mu)
  pgrad_pressure = pgrad_pressure_term(vtheta, grad_exner, exner, r_hat_m, h_grid, config)
  pgrad_phi = pgrad_phi_term(mu, grad_phi_i, r_hat_m)
  vtheta_divergence = vtheta_divergence_term(u, vtheta_dpi, vtheta, div_dp, dpi, r_hat_m, h_grid, config)
  w_vorticity = w_vorticity_correction_term(w_i, grad_w_i, r_hat_m)
  w_advection = w_advection_term(v_over_r_hat_i, grad_w_i)
  u_metric = u_metric_term(u, w_m, r_m)
  w_metric = w_metric_term(u, r_m)
  u_nct = u_nct_term(w_m, fcorcos)
  w_nct = w_nct_term(u_i, fcorcos)
  grad_kinetic_energy_v = grad_kinetic_energy_v_term(w_i, r_hat_m, h_grid, config)
  vorticity = vorticity_term(u, fcor, r_hat_m, h_grid, config)
  phi_advection = phi_advection_term(v_over_r_hat_i, grad_phi_i)

  ke_ke_1_a = jnp.sum(dpi * (u1 * grad_kinetic_energy_h[:, :, :, :, 0] +
                             u1 * grad_kinetic_energy_h[:, :, :, :, 1]), axis=-1)
  ke_ke_1_b = jnp.sum(1.0 / 2.0 * (u1**2 + u2**2) * dpi_divergence, axis=-1)

  ke_ke_2_a = jnp.sum(dpi * (u1 * grad_kinetic_energy_v[:, :, :, :, 0] +
                             u1 * grad_kinetic_energy_v[:, :, :, :, 1]), axis=-1)
  ke_ke_2_b = jnp.sum(1.0 / 2.0 * w_m**2 * dpi_divergence, axis=-1)

  ke_pe_1_a = jnp.sum(dpi_i_integral * (w_buoyancy - mu * g), axis=-1)
  ke_pe_1_b = jnp.sum(dpi_i_integral * phi_acceleration_v, axis=-1)

  ke_ie_1_a = jnp.sum(dpi_i_integral * mu * phi_acceleration_v, axis=-1)
  ke_ie_1_b = jnp.sum(dpi_i_integral * w_i * (w_buoyancy + g), axis=-1)

  ke_ie_2_a = jnp.sum(dpi * (u1 * pgrad_pressure[:, :, :, :, 0] +
                             u2 * pgrad_pressure[:, :, :, :, 1]), axis=-1)
  ke_ie_2_b = jnp.sum(config["cp"] * exner * vtheta_divergence, axis=-1)

  ke_ie_3_a = jnp.sum(dpi * (u1 * pgrad_phi[:, :, :, :, 0] +
                             u2 * pgrad_phi[:, :, :, :, 1]), axis=-1)
  ke_ie_3_b = jnp.sum(dpi_i_integral * mu * phi_advection, axis=-1)

  ke_ke_3_a = jnp.sum(dpi * (u1 * w_vorticity[:, :, :, :, 0] +
                             u2 * w_vorticity[:, :, :, :, 1]), axis=-1)
  ke_ke_3_b = jnp.sum(dpi_i_integral * w_i * w_advection, axis=-1)

  ke_ke_4_a = jnp.sum(dpi * u1 * vorticity[:, :, :, :, 0], axis=-1)
  ke_ke_4_b = jnp.sum(dpi * u2 * vorticity[:, :, :, :, 1], axis=-1)

  pe_pe_1_a = jnp.sum(phi * dpi_divergence, axis=-1)
  pe_pe_1_b = jnp.sum(dpi_i_integral * phi_advection, axis=-1)

  ke_ke_5_a = jnp.sum(dpi * (u1 * u_metric[:, :, :, :, 0] +
                             u2 * u_metric[:, :, :, :, 1]), axis=-1)
  ke_ke_5_b = jnp.sum(dpi * w_i * w_metric, axis=-1)

  ke_ke_6_a = jnp.sum(dpi * u1 * u_nct, axis=-1)
  ke_ke_6_b = jnp.sum(dpi_i_integral * w_i * w_nct, axis=-1)

  tends = explicit_tend(state, h_grid, v_grid, config, hydrostatic=False, deep=deep)
  u_tend = tends["u"]
  ke_tend_emp = jnp.sum(dpi * (u1 * u_tend[:, :, :, :, 0] +
                               u2 * u_tend[:, :, :, :, 1]), axis=-1)

  ke_tend_emp += jnp.sum(dpi_i_integral * w_i * tends["w_i"],
                         axis=-1)

  ke_tend_emp += jnp.sum((u1**2 + u2**2) / 2.0 * tends["dpi"], axis=-1)

  pe_tend_emp = jnp.sum(phi * tends["dpi"], axis=-1)
  pe_tend_emp += jnp.sum(dpi_i_integral * tends["phi_i"], axis=-1)

  ie_tend_emp = jnp.sum(config["cp"] * exner * tends["vtheta_dpi"], axis=-1)
  ie_tend_emp = jnp.sum(mu * dpi_i_integral * tends["phi_i"], axis=-1)

  return {"pairs": {"ke_ke_1": (ke_ke_1_a, ke_ke_1_b),
                    "ke_ke_2": (ke_ke_2_a, ke_ke_2_b),
                    "ke_ke_3": (ke_ke_3_a, ke_ke_3_b),
                    "ke_ke_4": (ke_ke_4_a, ke_ke_4_b),
                    "ke_ke_5": (ke_ke_5_a, ke_ke_5_b),
                    "ke_ke_6": (ke_ke_6_a, ke_ke_6_b),
                    "ke_pe_1": (ke_pe_1_a, ke_pe_1_b),
                    "pe_pe_1": (pe_pe_1_a, pe_pe_1_b),
                    "ke_ie_1": (ke_ie_1_a, ke_ie_1_b),
                    "ke_ie_2": (ke_ie_2_a, ke_ie_2_b),
                    "ke_ie_3": (ke_ie_3_a, ke_ie_3_b)},
          "empirical": {"ke": ke_tend_emp,
                        "ie": ie_tend_emp,
                        "pe": pe_tend_emp}}


def lower_boundary_correction(state0, state1, dt, h_grid, v_grid, dims, hydrostatic=True):
  if hydrostatic:
    return state1
  else:
    assert False
