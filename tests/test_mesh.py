  
from .context import spherical_spectral_element
from spherical_spectral_element.config import npt
from spherical_spectral_element.cubed_sphere import face_topo, gen_cube_topo, edge_to_vert, gen_vert_redundancy
from spherical_spectral_element.grid_definitions import  TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE
from spherical_spectral_element.cubed_sphere import inv_elem_id_fn, elem_id_fn
from spherical_spectral_element.grid_definitions import TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE
from spherical_spectral_element.mesh import gen_bilinear_grid


def test_gen_bilinear_grid_cs():
  nx = 7
  face_connectivity, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  gll_pos, gll_pos_2d, gll_to_cube_jacobian, gll_to_cube_jacobian_inv, vert_redundancy_gll = gen_bilinear_grid(face_connectivity, face_position, face_position_2d, vert_redundancy)
  for elem_idx in vert_redundancy_gll.keys():
    for (i_idx, j_idx) in vert_redundancy_gll[elem_idx].keys():
      for elem_idx_pair, i_idx_pair, j_idx_pair in vert_redundancy_gll[elem_idx][(i_idx, j_idx)]:
        try:
          assert(np.max(np.abs(gll_position[elem_idx][(i_idx, j_idx)] - gll_position[elem_idx_pair][(i_idx_pair, j_idx_pair)])) < 1e-10)
        except:
          print(f"local: {(inv_elem_id_fn(nx, elem_idx), i_idx, j_idx)} {gll_pos[elem_idx][(i_idx, j_idx)]}")
          print(f"pair: {(inv_elem_id_fn(nx, elem_idx_pair), i_idx_pair, j_idx_pair)} {gll_pos[elem_idx_pair][(i_idx_pair, j_idx_pair)]}")
  # note: test is only valid on quasi-uniform grid
  for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
    for x_idx in range(nx):
      for y_idx in range(nx):
        for i_idx in range(npt):
          for j_idx in range(npt):
            num_neighbors = 0
            if ((x_idx == 0 and y_idx == 0 and i_idx==0 and j_idx==0) or
               (x_idx==0 and y_idx==nx-1 and i_idx==0 and j_idx==npt-1) or
               (x_idx==nx-1 and y_idx==nx-1 and i_idx==npt-1 and j_idx==npt-1) or
               (x_idx==nx-1 and y_idx==0 and i_idx==npt-1 and j_idx==0)):
              num_neighbors = 2
            elif ((i_idx==0 and j_idx == 0) or
                  (i_idx==0 and j_idx == npt-1) or
                  (i_idx==npt-1 and j_idx==0) or
                  (i_idx==npt-1 and j_idx==npt-1)):
              num_neighbors = 3

            if j_idx != 0 and j_idx!=npt-1:
              if i_idx == 0 or i_idx==npt-1:
                num_neighbors = 1
            if i_idx != 0 and i_idx!=npt-1:
              if j_idx==0 or j_idx==npt-1:
                num_neighbors = 1
            elem_idx = elem_id_fn(nx, face_idx, x_idx, y_idx)
            if (i_idx, j_idx) in vert_redundancy_gll[elem_idx].keys():
              assert(num_neighbors == len(vert_redundancy_gll[elem_idx][(i_idx, j_idx)]))
            else:
              assert(num_neighbors == 0)