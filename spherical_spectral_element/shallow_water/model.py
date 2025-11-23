from ..config import np, npt
from ..assembly import dss_scalar
from ..operators import sphere_vorticity, sphere_gradient, sphere_divergence, sphere_laplacian_wk, sphere_vec_laplacian_wk


class shallow_water_model:
  def __init__(self, panel_subdiv, grid, radius_earth=6371e3, omega=7.292e-5, gravity=9.81, update_u = True, alpha=0.0, diffusion=False):
    self.ts_workspace = []
    self.nstages = 0
    self.radius_earth = radius_earth
    self.earth_period = omega
    self.gravity = gravity
    self.grid = grid
    self.panel_subdiv = panel_subdiv
    self.alpha = alpha
    self.diffusion = diffusion
    if diffusion:
      self.nu = 2.5e15 * ((30.0 * self.radius_earth)/(self.panel_subdiv * 6371e3))**3.2
    else:
      self.nu = 0.0
  def create_state_struct(self):
    state = {"u": np.zeros((self.grid.num_elem, npt, npt, 2)),
            "h": np.zeros((self.grid.num_elem, npt, npt))}
    return state
  def dss_state(self, state):
    for comp_idx in range(2):
      state["u"][:,:,:,comp_idx] = dss_scalar(state["u"][:,:,:,comp_idx], self.grid)
    state["h"] = dss_scalar(state["h"][:,:,:], self.grid)

  def init_workspace(self, nstages):
    self.nstages = nstages
    self.ts_workspace = [self.create_state_struct() for _ in range(nstages)]
  def calc_rhs(self, state_in, state_out):
    abs_vort = sphere_vorticity(state_in["u"], self.grid, a=self.radius_earth) + 2 * self.earth_period * (
                        -np.cos(self.grid.physical_coords[:,:,:,1])* np.cos(self.grid.physical_coords[:,:,:,0]) * np.sin(self.alpha) +
                        np.sin(self.grid.physical_coords[:,:,:,0]) * np.cos(self.alpha))
    energy = 0.5 * (state_in["u"][:,:,:,0]**2 + state_in["u"][:,:,:,1]**2 ) + self.gravity * (state_in["h"] + self.hs)
    energy_grad = sphere_gradient(energy, self.grid, a=self.radius_earth)
    state_out["u"][:,:,:,0] =    abs_vort * state_in["u"][:,:,:,1] - energy_grad[:,:,:,0]
    state_out["u"][:,:,:,1] = - abs_vort * state_in["u"][:,:,:,0] - energy_grad[:,:,:,1]
    state_out["h"] = -sphere_divergence(state_in["h"][:,:,:,np.newaxis] * state_in["u"], self.grid, a=self.radius_earth)
  def calc_hypervis(self, state_in, state_out):
    workspace = self.create_state_struct()
    a = self.radius_earth
    workspace["u"][:,:,:,:] = sphere_vec_laplacian_wk(state_in["u"], self.grid, a=a, damp=True)
    workspace["h"][:,:,:] = sphere_laplacian_wk(state_in["h"][:,:,:], self.grid, a=a)
    self.dss_state(state_out)
    state_out["u"][:,:,:,:] = -self.nu * sphere_vec_laplacian_wk(workspace["u"], self.grid, a=a, damp=True)
    state_out["h"][:,:,:] = -self.nu * sphere_laplacian_wk(workspace["h"][:,:,:], self.grid, a=a)
    self.dss_state(state_out)
  def copy_state(self, state_to_copy, state_out):
    state_out["u"][:] = state_to_copy["u"][:]
    state_out["h"][:] = state_to_copy["h"][:]
  def advance_hypervis_euler(self, state_in, state_out, dt, substeps=1):
    workspace = self.create_state_struct()
    workspace2 = self.create_state_struct()
    self.copy_state(state_in, state_out)
    for k in range(substeps):
      self.calc_hypervis(state_out, workspace)
      self.advance_state([state_out, workspace], workspace2, [1.0, dt/substeps])
      self.copy_state(workspace2, state_out)
  def advance_step_euler(self, state_in, state_out, dt):
    assert(self.nstages >= 1)
    self.calc_rhs(state_in, self.ts_workspace[0])
    self.dss_state(self.ts_workspace[0])
    self.advance_state([state_in, self.ts_workspace[0]], state_out, [1.0, dt])

  def advance_state(self, states_in, state_out, coeffs):
    for field in state_out.keys():
      state_out[field][:] = 0
    for coeff, state in zip(coeffs, states_in):
      for field in state.keys():
        state_out[field] += coeff * state[field]
  def advance_step_ssprk3(self, state_in, state_out, dt):
    assert(self.nstages >= 3)
    rhs_struct = self.ts_workspace[0]
    self.calc_rhs(state_in, rhs_struct)
    self.dss_state(rhs_struct)
    self.advance_state([state_in, rhs_struct], self.ts_workspace[1], [1.0, dt])
    self.calc_rhs(self.ts_workspace[1], rhs_struct)
    self.dss_state(rhs_struct)
    self.advance_state([state_in, self.ts_workspace[1], rhs_struct], self.ts_workspace[2], [3.0/4.0, 1.0/4.0, 1.0/4.0 * dt])
    self.calc_rhs(self.ts_workspace[2], rhs_struct)
    self.dss_state(rhs_struct)
    self.advance_state([state_in, self.ts_workspace[2], rhs_struct], state_out, [1.0/3.0, 2.0/3.0, 2.0/3.0 * dt])

  def simulate(self, end_time, u0_fn, h0_fn, hs_fn):
    state_n = self.create_state_struct()
    state_np1 = self.create_state_struct()
    state_tmp = self.create_state_struct()
    dt = 100.0 * (30.0/self.panel_subdiv) # todo: automatically calculate CFL from sw dispersion relation
    self.hs = hs_fn(self.grid.physical_coords[:,:,:,0], self.grid.physical_coords[:,:,:,1])
    state_n["u"] = u0_fn(self.grid.physical_coords[:,:,:,0], self.grid.physical_coords[:,:,:,1])
    state_n["h"] = h0_fn(self.grid.physical_coords[:,:,:,0], self.grid.physical_coords[:,:,:,1])
    t = 0.0
    times = np.arange(0.0, end_time, dt)
    self.init_workspace(3)
    for t in times:
      print(f"{t/end_time*100}%")
      self.advance_step_ssprk3(state_n, state_tmp, dt)
      #self.advance_step_euler(state_n, state_tmp, dt)
      if self.diffusion:
        self.advance_hypervis_euler(state_tmp, state_np1, dt, substeps=1)
        #self.advance_hypervis_euler(state_n, state_np1, dt, substeps=1)
      else:
        state_np1, state_tmp = state_tmp, state_np1 
      print(np.max(np.abs(state_np1["u"] - state_tmp["u"])))
      state_n, state_np1 = state_np1, state_n 
      assert(not np.any(np.isnan(state_n["u"])))
      assert(not np.any(np.isnan(state_n["h"])))
    #state_n["u"][:] -= state_np1["u"][:]
    #state_n["h"][:] -= state_np1["h"][:]
    return t, state_n
