from ..config import np, use_jax, partial, vmap_1d_apply
from ..assembly import dss_scalar, dss_scalar_for


def wrap_model_struct(u, vtheta_dpi, dpi, phi_surf, phi_i=0.0, w_i=0.0):
  state = {"u": u,
           "vtheta_dpi": vtheta_dpi,
           "dpi": dpi,
           "phi_surf": phi_surf,
           "phi_i": phi_i,
           "w_i": w_i
           }
  return state



def dss_scalar_3d(variable, h_grid, dims, scaled=True):
  dss_onlyarg = lambda vec: dss_scalar(vec, h_grid, dims, scaled=scaled)
  return vmap_1d_apply(dss_onlyarg, variable, -1, -1)

def dss_scalar_3d_for(variable, h_grid, dims, scaled=True):
  levs = []
  for lev_idx in range(variable.shape[-1]):
    levs.append(dss_scalar_for(variable[:, :, :, lev_idx], h_grid))
  return np.stack(levs, axis=-1)



def dss_model_state(state_in, h_grid, dims, scaled=True, hydrostatic=True):
  u_dss = dss_scalar_3d(state_in["u"][:, :, :, :, 0], h_grid, dims, "num_model")
  v_dss = dss_scalar_3d(state_in["u"][:, :, :, :, 1], h_grid, dims, "num_model")
  vtheta_dpi_dss = dss_scalar_3d(state_in["vtheta_dpi"][:, :, :, :], h_grid, "num_model")
  dpi_dss = dss_scalar_3d(state_in["dpi"][:, :, :, :], h_grid, "num_model")
  if hydrostatic:
    w_i_dss = state_in["w_i"]
    phi_i_dss = state_in["phi_i"]
  else:
    w_i_dss = dss_scalar_3d(state_in["w_i"], h_grid, "num_interface")
    phi_i_dss = dss_scalar_3d(state_in["phi_i"], h_grid, "num_interface")
  return wrap_model_struct(np.stack((u_dss, v_dss), axis=-1),
                           vtheta_dpi_dss, dpi_dss, state_in["phi_surf"],
                           phi_i_dss,
                           w_i_dss)
