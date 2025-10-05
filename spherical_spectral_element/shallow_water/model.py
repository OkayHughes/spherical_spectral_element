from ..config import npt




class shallow_water_model:
  def __init__(self):
    self.ts_workspace = []
    self.nstages = 0
  def create_state_struct(self):
    state = {"u": np.zeros((NELEM, npt, npt, 2)),
            "h": np.zeros((NELEM, npt, npt))}
    return state
  def dss_state(self, state):
    for comp_idx in range(2):
      state["u"][:,:,:,comp_idx] = dss_scalar(state["u"][:,:,:,comp_idx])
    state["h"] = dss_scalar(state["h"][:,:,:])

  def init_workspace(self, nstages):
    self.nstages = nstages
    self.ts_workspace = [self.create_state_struct() for _ in range(nstages)]
  def calc_rhs(self, state_in, state_out):
    abs_vort = 1/rearth * sphere_vorticity(state_in["u"]) + 2 * earth_period * np.sin(gll_latlon[:,:,:,0])
    energy = 0.5 * (state_in["u"][:,:,:,0]**2 + state_in["u"][:,:,:,1]**2 ) + g * (state_in["h"] + self.hs)
    energy_grad = 1/rearth * sphere_gradient(energy)
    state_out["u"][:,:,:,0] = abs_vort * state_in["u"][:,:,:,1] - energy_grad[:,:,:,0]
    state_out["u"][:,:,:,1] = -abs_vort * state_in["u"][:,:,:,0] - energy_grad[:,:,:,1]
    state_out["h"] = -sphere_divergence(1/rearth * state_in["h"][:,:,:,np.newaxis] * state_in["u"])
  def calc_hypervis(self, state_in, state_out):
    workspace = self.create_state_struct()
    for comp_idx in range(2):
      workspace["u"][:,:,:,0] = sphere_laplacian(state_in["u"][:,:,:,0])
      state_out["u"][:,:,:,0] = sphere_laplacian(workspace["u"][:,:,:,0])
  def advance_hypervis_euler(self, state_in, state_out, dt, substeps=1):
    pass
  def advance_step_euler(self, state_in, state_out, dt):
    assert(self.nstages >= 1)
    self.calc_rhs(state_in, self.ts_workspace[0])
    self.dss_state(self.ts_workspace[0])
    self.advance_state(self, [state_in, self.ts_workspace[0]], state_out, [1.0, dt])

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
    dt = 1.0 # todo: automatically calculate CFL from sw dispersion relation
    self.hs = hs_fn(gll_latlon[:,:,:,0], gll_latlon[:,:,:,1])
    state_n["u"] = u0_fn(gll_latlon[:,:,:,0], gll_latlon[:,:,:,1])
    state_n["h"] = h0_fn(gll_latlon[:,:,:,0], gll_latlon[:,:,:,1])
    t = 0.0
    times = np.arange(0.0, end_time, dt)
    self.init_workspace(3)
    for t in tqdm(times):
      self.advance_step_ssprk3(state_n, state_np1, dt)
      state_n, state_np1 = state_np1, state_n
    return t, state_n
