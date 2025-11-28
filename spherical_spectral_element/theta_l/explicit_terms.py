from ..config import jnp
from .infra import vel_model_to_interface, model_to_interface, interface_to_model, interface_to_model_vec
from .eos import get_mu

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
  lon = h_grid["physical_coordinates"][:, :, :, 1]
  
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
  vtheta = state["vtheta_dpi"]/dpi
  grad_phi_i = sphere_gradient_3d(phi_i, h_grid, config) 
  v_over_r_hat_i = model_to_interface(u / r_hat_m[:, :, :, jnp.newaxis])
  div_dp = sphere_divergence_3d(dpi[:, :, :, :, jnp.newaxis] * u /
                                r_hat_m[:, :, :, :, jnp.newaxis] , h_grid, config)
  return (phi_i, phi, dpi_i, pnh, exner,
          r_hat_i, mu, r_hat_m, z, r_m, g,
          fcor, fcorcos, w_m,
          grad_w_i, grad_exner, vtheta,
          grad_phi_i, v_over_r_hat_i,
          div_dp)
          

def vorticity_term(u, fcor, r_hat_m, h_grid):
  vort = sphere_vorticity_3d(u, h_grid, config)
  vort /= r_hat_m
  vort_term = np.stack((u[:, :, :, :, 1]  * (fcor[:, :, :, jnp.newaxis, jnp.newaxis] + vort),
                        -u[:, :, :, :, 0]  * (fcor[:, :, :, jnp.newaxis, jnp.newaxis] + vort)), axis=-1)
  return vort_term

def grad_kinetic_energy_h_term(u, r_hat_m, h_grid):
  grad_kinetic_energy = sphere_gradient_3d((u[:, :, :, :, 0]**2 + 
                                            u[:, :, :, :, 1]**2) / 2.0, h_grid, config)

  return -grad_kinetic_energy / r_hat_m

def grad_kinetic_energy_v_term(w_i, r_hat_m):
  w_sq_m = interface_to_model(w_i * w_i) / 2.0
  w2_grad_sph = sphere_gradient_3d(w_sq_m)/r_hat_m
  return -w2_grad_sph

def w_vorticity_correction(w_i, grad_w_i, r_hat_m):
  w_grad_w_m = interface_to_model_vec(w_i[:, :, :, :, jnp.newaxis] * grad_w_i)
  w_grad_w_m /= r_hat_m[:, :, :, :, jnp.newaxis]
  return w_grad_w_m

def u_metric_term(u, w_m, r_m):
  return -w_m[:, :, :, :, jnp.newaxis] * u / r_m[:, :, :, jnp.newaxis]

def u_nct(w_m, fcorcos):
  return -jnp.stack((w_m, jnp.zeros_like(w_m)), axis=-1) * fcorcos

def pgrad_pressure_term(vtheta, grad_exner, exner, r_hat_m, config):
  grad_p_term_1 = config["cp"] * vtheta[:, :, :, :, np.newaxis] * grad_exner
  
  grad_vtheta_exner = gradient_sphere_3d(vtheta * exner, h_grid, config) / r_hat_m
  grad_vtheta = gradient_sphere_3d(vtheta, h_grid, config) / r_hat_m
  
  grad_p_term_2 = config["cp"] * (grad_vtheta_exner - exner[:, :, :, :, np.newaxis] * vtheta)
  
  return -(grad_p_term_1 + grad_p_term_2) / 2.0

def pgrad_phi_term(mu, grad_phi_i, r_hat_m):
  pgf_gradphi_m = interface_to_model_vec(mu[:, :, :, :, jnp.newaxis] * grad_phi_i)
  pgf_gradphi_m /= r_hat_m[:, :, :, :, jnp.newaxis]
  return -pgf_gradphi_m


def w_advection(v_over_r_hat_i, grad_w_i):
  v_grad_w_i = (v_over_r_hat_i[:, :, :, 0] * grad_w_i[:, :, :, 0] +
                v_over_r_hat_i[:, :, :, 1] * grad_w_i[:, :, :, 1])
  return -v_grad_w_i


def w_metric_term(u, r_i):
  v_sq_over_r_i = model_to_interface(u / r_m)
  return (v_sq_over_r_i[:, :, :, 0] + v_sq_over_r_i[:, :, :, 1])


def w_nct_term(u_i, fcorcos):
  return -u_i[:, :, :, :, 0] * fcorcos


def w_buoyancy_term(g, mu):
  return -g*(1 - mu)


def phi_advection_term(v_over_r_hat_i, grad_phi_i):
  v_grad_phi_i = (v_over_r_hat_i[:, :, :, 0] * grad_phi_i[:, :, :, 0] +
                  v_over_r_hat_i[:, :, :, 1] * grad_phi_i[:, :, :, 1])
  return -v_grad_phi_i 


def phi_acceleration_v(g, w_i):
  return g * w_i


def vtheta_divergence_term(u, vtheta_dpi, vtheta, div_dp, dpi, r_hat_m):
  v_vtheta = u * vtheta_dpi[:,:,:,:,np.newaxis]
  v_vtheta /= r_hat_m
  div_v_vtheta = sphere_divergence_3d(v_vtheta, h_grid, config)/2.0
  grad_vtheta = sphere_gradient_3d(vtheta, h_grid, config)
  grad_vtheta /= r_hat_m

  div_v_vtheta += (vtheta * div_dp + (dpi * (u[:, :, :, :, 0] * grad_vtheta[:, :, :, :, 0] + 
                                            u[:, :, :, :, 1] * grad_vtheta[:, :, :, :, 1]))) / 2.0
  return -div_v_theta


def dpi_divergence_term(div_dp):
  return -div_dp
  


def explicit_tend(state, h_grid, v_grid, config, hydrostatic=False, deep=False):
  dpi = state["dpi"]
  u = state["u"]
  vtheta_dpi = state["vtheta_dpi"]
  vtheta = vtheta_dpi/dpi
  radius_earth = config["radius_earth"]

  (phi_i, phi, dpi_i, pnh, exner,
   r_hat_i, mu, r_hat_m, z, r_m, g,
   fcor, fcorcos, w_m,
   grad_w_i, grad_exner, vtheta,
   grad_phi_i, v_over_r_hat_i,
   div_dp) = calc_shared_quantities(state, h_grid,
                                    v_grid, config,
                                    hydrostatic=hydrostatic,
                                    deep=deep)

  u_i = vel_model_to_interface(u, dpi, dpi_i)

  u_tend = (vorticity_term(u, fcor, r_hat_m, h_grid) + 
            grad_kinetic_energy_h_term(u, r_hat_m, h_grid) +
            grad_kinetic_energy_v_term(w_i, r_hat_m) +
            w_vorticity_correction(w_i, grad_w_i, r_hat_m) +
            pgrad_pressure_term(vtheta, grad_exner, exner, r_hat_m, config) +
            pgrad_phi_term(mu, grad_phi_i, r_hat_m))
  if deep:
    u_tend += (u_metric_term(u, w_m, r_m) +
               u_nct(w_m, fcorcos))
  if not hydrostatic:
    w_tend = (w_advection(v_over_r_hat_i, grad_w_i) +
              w_buoyancy_term(g, mu))
    if deep:
      w_tend += (w_metric_term(u, r_i) +
                 w_nct_term(u_i, fcorcos))
  else:
    w_tend = 0.0
  if not hydrostatic:
    phi_tend = (phi_advection_term(v_over_r_hat_i, grad_phi_i) +
                phi_acceleration_v(g, w_i))
  else:
    phi_tend = 0.0

  vtheta_dpi_tens = vtheta_divergence_term(u, vtheta_dpi, vtheta, div_dp, dpi, r_hat_m)
  return wrap_model_struct(u_tens, vtheta_dpi_tens, dpi_tens, state["phi_surf"], phi_tend, w_tend)

def calc_energy_quantities(state, h_grid, v_grid, config, hydrostatic=True, deep=False):
  (phi_i, phi, dpi_i, pnh, exner,
   r_hat_i, mu, r_hat_m, z, r_m, g,
   fcor, fcorcos, w_m,
   grad_w_i, grad_exner, vtheta,
   grad_phi_i, v_over_r_hat_i,
   div_dp) = calc_shared_quantities(state, h_grid,
                                    v_grid, config,
                                    hydrostatic=hydrostatic,
                                    deep=deep)


def lower_boundary_correction(state0, state1, dt, h_grid, v_grid, dims, hydrostatic=True):
  if hydrostatic:
    return state1
  else:
    assert False