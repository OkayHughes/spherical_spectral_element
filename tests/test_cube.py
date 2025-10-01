from .context import spherical_spectral_element
from spherical_spectral_element.config import np
from spherical_spectral_element.cubed_sphere import face_topo, gen_cube_topo, edge_to_vert
from spherical_spectral_element.cubed_sphere import  TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE
from spherical_spectral_element.cubed_sphere import inv_elem_id_fn, elem_id_fn
from spherical_spectral_element.cubed_sphere import TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE

def test_elem_id_fns():
  nx = 15
  for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
      for x_idx in range(nx):
        for y_idx in range(nx):
          f, x, y = inv_elem_id_fn(nx, elem_id_fn(nx, face_idx, x_idx, y_idx))
          assert(f == face_idx and x == x_idx and y == y_idx)

def test_face_topo():
  #test if topology is consistent
  for face_id in face_topo:
    face_info = face_topo[face_id]
    for edge_id in face_info:
      edge_info = face_info[edge_id]
      face_id_pair1, edge_id_pair1, direction_match1 = edge_info
      face_id_pair2, edge_id_pair2, direction_match2 = face_topo[face_id_pair1][edge_id_pair1]
      assert(face_id_pair2 == face_id)
      assert(edge_id_pair2 == edge_id)
      assert(direction_match1 == direction_match2)

def test_gen_cube_topo():
  nx = 15
  face_connectivity, face_position, face_position_2d = gen_cube_topo(nx)
  for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
    for x_idx in range(nx):
      for y_idx in range(nx):
        for edge_idx in [TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE]:
          elem_idx = elem_id_fn(nx, face_idx, x_idx, y_idx)
          idx_pair, edge_idx_pair, is_forwards = face_connectivity[elem_idx, edge_idx, :]
          face_idx_pair, x_idx_pair, y_idx_pair  = inv_elem_id_fn(nx, idx_pair)
          v0_local, v1_local = edge_to_vert(edge_idx)
          v0_pair, v1_pair = edge_to_vert(edge_idx_pair, is_forwards=is_forwards)
          err = False
          for v_local, v_pair in [(v0_local, v0_pair), (v1_local, v1_pair)]:
            pos_local = face_position[elem_idx, v_local, :]
            pos_pair = face_position[idx_pair, v_pair, :]
            if np.max(np.abs(pos_local - pos_pair)) > 1e-10:
              err = True
              print(f"elem_idx: {elem_idx}, idx_pair: {idx_pair}")
              print(f"pos1: {pos_local}, pos2: {pos_pair}")
          assert(not err)