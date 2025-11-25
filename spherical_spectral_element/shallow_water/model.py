from ..config import np, npt
from ..assembly import dss_scalar
from ..operators import sphere_vorticity, sphere_gradient, sphere_divergence, sphere_laplacian_wk, sphere_vec_laplacian_wk

def create_state_struct(u,h,hs):
  return {"u": u, 
          "h": h,
          "hs":hs}

def get_config_sw(radius_earth=6371e3, earth_period=7.292e-5, gravity=9.81, alpha=0.0, ne=30):
  return {"radius_earth": radius_earth,
          "earth_period": earth_period,
          "gravity": gravity,
          "alpha": alpha,
          "nu": 2.5e15 * ((30.0 * radius_earth)/(ne * 6371e3))**3.2}
def dss_state(state, grid):
  u = dss_scalar(state["u"][:,:,:,0], grid)
  v = dss_scalar(state["u"][:,:,:,1], grid)
  h = dss_scalar(state["h"][:,:,:], grid)
  return create_state_struct(np.stack((u,v),axis=-1), h, state["hs"])


def calc_rhs(state_in, grid, config):
  abs_vort = sphere_vorticity(state_in["u"], grid, a=config["radius_earth"]) + 2 * config["earth_period"] * (
                      -np.cos(grid["physical_coords"][:,:,:,1])* np.cos(grid["physical_coords"][:,:,:,0]) * np.sin(config["alpha"]) +
                      np.sin(grid["physical_coords"][:,:,:,0]) * np.cos(config["alpha"]))
  energy = 0.5 * (state_in["u"][:,:,:,0]**2 + state_in["u"][:,:,:,1]**2 ) + config["gravity"] * (state_in["h"] + state_in["hs"])
  energy_grad = sphere_gradient(energy, grid, a=config["radius_earth"])
  u_tend =    abs_vort * state_in["u"][:,:,:,1] - energy_grad[:,:,:,0]
  v_tend = - abs_vort * state_in["u"][:,:,:,0] - energy_grad[:,:,:,1]
  h_tend = -sphere_divergence(state_in["h"][:,:,:,np.newaxis] * state_in["u"], grid, a=config["radius_earth"])
  return create_state_struct(np.stack((u_tend, v_tend), axis=-1), h_tend, state_in["hs"])

def calc_hypervis(state_in, grid, config):
  a = config["radius_earth"]
  u_tmp = sphere_vec_laplacian_wk(state_in["u"], grid, a=a, damp=True)
  h_tmp = sphere_laplacian_wk(state_in["h"][:,:,:], grid, a=a)
  lap1 = dss_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid)
  u_tmp = -config["nu"] * sphere_vec_laplacian_wk(lap1["u"], grid, a=a, damp=True)
  h_tmp = -config["nu"] * sphere_laplacian_wk(lap1["h"], grid, a=a)
  return dss_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid)

def advance_hypervis_euler(state_in, dt, grid, config, substeps=1):
  next_step = state_in
  for k in range(substeps):
    hvis_tend = calc_hypervis(next_step, grid, config)
    next_step = advance_state([next_step, hvis_tend], [1.0, dt/substeps])
  return next_step
    
def advance_state(states_in, coeffs):
  state_res = create_state_struct(states_in[0]["u"] * coeffs[0], states_in[0]["h"]*coeffs[0], states_in[0]["hs"])
  for state_idx in range(1, len(coeffs)):
    state = states_in[state_idx]
    coeff = coeffs[state_idx]
    state_res = create_state_struct(state_res["u"] + state["u"] * coeff, 
                                    state_res["h"] + state["h"] * coeff,
                                    state_res["hs"])
  return state_res
def advance_step_euler(state_in, dt, grid, config):
  state_tend = calc_rhs(state_in, grid, config)
  state_tend_c0 = dss_state(state_tend, grid)
  return advance_state([state_in, state_tend_c0], [1.0, dt])

def advance_step_ssprk3(state0, dt, grid, config):
  tend = calc_rhs(state0, grid, config)
  tend_c0 = dss_state(tend, grid)
  state1 = advance_state([state0, tend_c0], [1.0, dt])
  tend = calc_rhs(state1, grid, config)
  tend_c0 = dss_state(tend, grid)
  state2 = advance_state([state0, state1, tend_c0], [3.0/4.0, 1.0/4.0, 1.0/4.0 * dt])
  tend = calc_rhs(state2, grid, config)
  tend_c0 = dss_state(tend, grid)
  return advance_state([state0, state2, tend_c0], [1.0/3.0, 2.0/3.0, 2.0/3.0 * dt])


def simulate_sw(end_time, state_in, grid, config, ne, diffusion=False):
  dt = 100.0 * (30.0/ne) # todo: automatically calculate CFL from sw dispersion relation
  #hs = hs_fn(grid["physical_coords"][:,:,:,0], grid["physical_coords"][:,:,:,1])
  #u0 = u0_fn(grid["physical_coords"][:,:,:,0], grid["physical_coords"][:,:,:,1])
  #h0 = h0_fn(grid["physical_coords"][:,:,:,0], grid["physical_coords"][:,:,:,1])
  state_n = state_in
  t = 0.0
  times = np.arange(0.0, end_time, dt)
  for t in times:
    print(f"{t/end_time*100}%")
    state_tmp = advance_step_ssprk3(state_n, dt, grid, config)
    #state_tmp = advance_step_euler(state_n, dt, grid, config)
    if diffusion:
      state_np1 = advance_hypervis_euler(state_tmp, dt, grid, config, substeps=1)
    else:
      state_np1 = state_tmp
    state_n, state_np1 = state_np1, state_n
    assert(not np.any(np.isnan(state_n["u"])))
    assert(not np.any(np.isnan(state_n["h"])))
  #state_n["u"][:] -= state_np1["u"][:]
  #state_n["h"][:] -= state_np1["h"][:]
  return state_n
