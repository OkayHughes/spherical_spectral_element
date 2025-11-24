from ..config import np
from infra import shape_model, get_tensor, shape_interface
from constants import Rgas, p0

def get_p_mid(state, h_grid, v_grid):
  p = np.cumsum(state["dp3d"], axis=-1) + v_grid["hybrid_a_i"][0] * v_grid["reference_pressure"]
  p -= 0.5 * state["dp3d"]
  return p
def get_balanced_phi(state, h_grid, v_grid):
  p = get_p_mid(state, h_grid, v_grid)
  dphi = -Rgas * state["vtheta_dpi"] * (p/p0)**(Rgas/cp-1.0)/p0
  phi_i = get_tensor_interface(h_grid, v_grid)
  phi_i[:, :, :, -1] = state["phi_s"]
  phi_i[:, :, :, :-1] = np.cumsum(dphi[:, :, :, ::-1], axis=-1)[:,:,::-1] + state["phi_s"][:,:,:,np.newaxis]
  return phi_i


