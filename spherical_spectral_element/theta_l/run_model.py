from ..config import jnp
from .time_stepping import advance_euler


def simulate_theta(end_time, ne, state_in,
                   h_grid, v_grid, config,
                   dims, hydrostatic=True, deep=False,
                   diffusion=False, step_type="euler"):
  dt = 300.0 * (30.0 / ne)  # todo: automatically calculate CFL from sw dispersion relation
  state_n = state_in
  t = 0.0
  times = jnp.arange(0.0, end_time, dt)
  k = 0
  for t in times:
    print(f"{k/len(times-1)*100}%")
    if step_type == "euler":
      state_tmp = advance_euler(state_n, dt, h_grid, v_grid, dims, hydrostatic=True, deep=False)
      state_np1 = state_tmp
    # if diffusion:

    #   state_np1 = advance_hypervis_euler(state_tmp, dt, grid, config, dims, substeps=1)
    # else:
    #
    state_n, state_np1 = state_np1, state_n

    # versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["u"]))))
    # versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["h"]))))
    k += 1
