from ..config import np

def get_tensor(shape, device="cpu"):
  return np.zeros(shape)

def shape_model(hgrid, vgrid):
  dim_model = (hgrid.num_elem, npt, npt, vgrid.num_lev)
  return dim_model
def shape_interface(hgrid, vgrid):
  dim_model = (hgrid.num_elem, npt, npt, vgrid.num_levp)
  return dim_model

def shape_2d(hgrid):
  dim_2d = (hgrid.num_elem, npt, npt)
  return dim_2d

def get_tensor_model(h_grid, v_grid device="cpu"):
  return np.zeros(shape_model(h_grid, v_grid))

def get_tensor_interface(h_grid, v_grid device="cpu"):
  return np.zeros(shape_interface(h_grid, v_grid))