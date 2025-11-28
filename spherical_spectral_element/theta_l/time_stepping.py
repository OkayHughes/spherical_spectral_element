from model_state import wrap_model_struc
from explicit_terms import explicit_tendency_hydro


def rfold_state(state1, state2, fold_coeff1, fold_coeff2):
  return wrap_model_struct(state1["u"] * fold_coeff1 + state2("u") * fold_coeff2,
                           state1["vtheta_dpi"] * fold_coeff1 + state2("vtheta_dpi") * fold_coeff2,
                           state1["dpi"] * fold_coeff1 + state2["vtheta_dpi"] * fold_coeff2,
                           state1["phi_surf"],
                           state1["phi_i"] * fold_coeff1 + state2["phi_i"]*fold_coeff2,
                           state1["w_i"] * fold_coeff1 + state2["w_i"]*fold_coeff2)


def advance_state(states, coefficients, nstages):
  for coeff_idx in range(nstages-1):
    state_out = rfold_state(states[coeff_idx],
                            states[coff_idx + 1],
                            coeffs[coeff_idx],
                            coeffs[coeff_idx + 1])
  return state_out


def advance_euler(state_in, dt, h_grid, v_grid, dims, hydrostatic=True, deep=False):
  u_tend = explicit_tendency(state_in, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_state(u_tend)
  u1 = advance_state([state_in, u_tend_c0], [1.0, dt], 2)
  return lower_boundary_correction(state_in, u1, dt, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)


def ullrich_5stage(state_in, dt, h_grid, v_grid, dims, hydrostatic=True):
  u_tend = explicit_tendency(state_in, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_state(u_tend)
  u1 = advance_state([state_in, u_tend_c0], [1.0, dt / 5.0], 2)
  u1 = lower_boundary_correction(state_in, u1, dt / 5.0, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend = explicit_tendency(u1, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_state(u_tend)
  u2 = advance_state([state_in, u_tend_c0], [1.0, dt / 5.0], 2)
  u2 = lower_boundary_correction(state_in, u2, dt / 5.0, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend = explicit_tendency(u2, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_state(u_tend)
  u3 = advance_state([state_in, u_tend_c0], [1.0, dt / 3.0], 2)
  u3 = lower_boundary_correction(state_in, u3, dt / 3.0, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend = explicit_tendency(u3, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_state(u_tend)
  u4 = advance_state([state_in, u_tend_c0], [1.0, 2.0 * dt / 3.0], 2)
  u4 = lower_boundary_correction(state_in, u4, 2.0 * dt / 3.0, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend = explicit_tendency(u4, h_grid, v_grid, hydrostatic=hydrostatic)
  u_tend_c0 = dss_state(u_tend)
  return advance_state([state_in, u1, u_tend_c0], [-1.0 / 4.0, 
                                                   5.0 / 4.0, 
                                                   3.0 * dt / 4.0], 3)

