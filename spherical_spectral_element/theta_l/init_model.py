from ..config import np
from model_state import init_model_struct

def p_from_z_monotonic(pressures, p_given_z, eps=1e-5, z_top=80e3):
  z_guesses=p_given_z(z_top * np.ones_like(pressures))
  not_converged = np.logical_not(np.abs((p_given_z(z_guesses) - pressures))/pressures < eps)
  frac = 0.5
  while np.any(not_converged):
    p_guess = p_given_z(z_guesses[not_converged]) 
    too_high = p_guess < pressures
    z_guesses[not_converged][too_high] -= frac * z_top
    z_guesses[not_converged][np.logical_not(too_high)] += frac * z_top
    not_converged = np.logical_not(np.abs((p_given_z(z_guesses) - pressures))/pressures < eps)
  return z_guesses


def init_model_z_hydro(z_surf_func, pi_surf_func, T_func, u_func, v_func, Q_funcs, h_grid, v_grid):
  initial_state = init_model_struct(h_grid, v_grid, len(Q_funcs))
  
