from model_state import init_model_struct
from explicit_terms import explicit_tendency_hydro

def advance_state(states, coefficients, h_grid, v_grid, advance_scalar=False):
  state_out = init_model_struct(h_grid, v_grid, states[0]["Qdp"].shape[-1])
  if not advance_scalar:
    state_out["Qdp"] = states[0]["Qdp"]
  for state, coefficient in zip(states, coefficients):
    state_out["u"] += state["u"] * coefficient
    state_out["vtheta_dpi"] += state["vtheta_dpi"] * coefficient
    state_out["dpi"] += state["dpi"] * coefficient
    if advance_scalar:
      state_out["Qdp"] += state["Qdp"] * coefficient
  return state_out

def ullrich_5stage(state_in, dt, h_grid, v_grid):
  u_tend = explicit_tendency_hydro(state_in, h_grid, v_grid)
  u1 = advance_state([state_in, u0_tend], [1.0, dt/5.0])
  u_tend = explicit_tendency_hydro(u1, h_grid, v_grid)
  u2 = advance_state([state_in, u_tend], [1.0, dt/5.0])
  u_tend = explicit_tendency_hydro(u2, h_grid, v_grid)
  u3 = advance_state([state_in, u_tend], [1.0, dt/3.0])
  u_tend = explicit_tendency_hydro(u3, h_grid, v_grid)
  u4 = advance_state([state_in, u_tend], [1.0, 2.0*dt/3.0])
  u_tend = explicit_tendency_hydro(u4, h_grid, v_grid)
  return advance_state([state_in, u1, u_tend], [-1.0/4.0, 5.0/4.0, 3.0*dt/4.0])

