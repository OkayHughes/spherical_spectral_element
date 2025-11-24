from ..config import np, npt
from infra import get_tensor, get_tensor_model, get_tensor_interface, shape_2d

def init_model_struct(h_grid, v_grid, nQ):
  dim_model = shape_model(h_grid, v_grid)
  dim_2d = shape_2d(h_grid)
  state = {}
  state["u"] = get_tensor((*dim_model, 2))
  state["thetav_dpi"] = get_tensor_model(h_grid, v_grid)
  state["dpi"] = get_tensor_model(h_grid, v_grid)
  state["phi_s"] = get_tensor(dim_2d)
  state["Qdp"] = get_tensor((*dim_model, nQ))
  return state

def wrap_model_struct(u, vtheta_dpi, dpi, phi_s, Qdp):
  state = {"u": u,
           "vtheta_dpi": vtheta_dpi,
           "dpi": dpi
           "phi_s": phi_s,
           "Qdp": Qdp
           }
  return state