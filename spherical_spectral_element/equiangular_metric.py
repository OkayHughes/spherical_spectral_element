from .config import np, npt, DEBUG

def gen_metric_terms_equiangular(cube_points, cube_points_2d, cube_redundancy):
  NFACES = cube_points.shape[0]
  top_face_mask = (np.abs(cube_points[:, 1, 1, 2] - 1.0) < 1e-10).reshape((NFACES, 1, 1))
  bottom_face_mask = (np.abs(cube_points[:, 1, 1, 2] - -1.0) < 1e-10).reshape((NFACES, 1, 1))
  front_face_mask = (np.abs(cube_points[:, 1, 1, 1] - 1.0) < 1e-10).reshape((NFACES, 1, 1))
  back_face_mask = (np.abs(cube_points[:, 1, 1, 1] - -1.0) < 1e-10).reshape((NFACES, 1, 1))
  left_face_mask = (np.abs(cube_points[:, 1, 1, 0] - -1.0) < 1e-10).reshape((NFACES, 1, 1))
  right_face_mask = (np.abs(cube_points[:, 1, 1, 0] - 1.0) < 1e-10).reshape((NFACES, 1, 1))
  gll_latlon = np.zeros(shape=(NFACES, npt, npt, 2))
  cube_to_sphere_jacobian = np.zeros(shape=(NFACES, npt, npt, 2, 2))
  if DEBUG:
    gll_latlon_pert = np.zeros(shape=(NFACES, npt, npt, 2))
    cube_points_pert = np.zeros_like(cube_points_2d)

  if DEBUG:
    n_mask = 0
    for m1, mask1 in enumerate([top_face_mask, bottom_face_mask, front_face_mask, back_face_mask, left_face_mask, right_face_mask]):
      for m2, mask2 in enumerate([top_face_mask, bottom_face_mask, front_face_mask, back_face_mask, left_face_mask, right_face_mask]):
        if m1 != m2:
          ct = np.sum(np.logical_and(mask1, mask2))
          assert(ct == 0)
      n_mask += np.sum(mask1)
    assert(n_mask == NFACES)
  def set_jac_eq(jac, lat, lon, mask, flip_x=1.0, flip_y=1.0):
    jac[:, :, :, 0, 1] += flip_x * np.cos(lon[:,:,:])**2 * mask
    jac[:, :, :, 0, 0] += flip_x * -1/4 * np.sin(2 * lon[:,:,:]) * np.sin(2 * lat[:,:,:]) * mask
    jac[:, :, :, 1, 0] += flip_y * np.cos(lon[:,:,:]) * np.cos(lat[:,:,:])**2 * mask
  def set_jac_pole(jac, lat, lon, mask, k, flip_x=1.0, flip_y=1.0):
    jac[:, :, :, 0, 1] += flip_x * k * np.cos(lon)*np.tan(lat) * mask
    jac[:, :, :, 0, 0] += flip_x * -k * np.sin(lon)* np.sin(lat)**2 * mask
    jac[:, :, :, 1, 1] += flip_y * np.sin(lon)* np.tan(lat) * mask
    jac[:, :, :, 1, 0] += flip_y * np.cos(lon)* np.sin(lat)**2 * mask

  def dlatlon_dcube(latlon_fn, latlon_idx, cube_idx, mask):
    gll_latlon_pert[:] = 0
    cube_points_pert[:] = cube_points_2d[:]
    cube_points_pert[:,:, :, cube_idx] *= 0.99999
    gll_latlon_pert[:, :, :, latlon_idx] += latlon_fn(cube_points_pert[:, :, :, 0] , cube_points_pert[:, :, :, 1]) * mask
    result = (gll_latlon_pert[:, :, :, latlon_idx] - gll_latlon[:, :, :, latlon_idx])/(cube_points_pert[:, :, :, cube_idx] - cube_points_2d[:, :, :, cube_idx])
    return result

  def test_face(lat_fn, lon_fn, mask):
    dlat_dx = dlatlon_dcube(lat_fn, 0, 0, mask)
    dlat_dy = dlatlon_dcube(lat_fn, 0, 1, mask)
    dlon_dx = dlatlon_dcube(lon_fn, 1, 0, mask)
    dlon_dy = dlatlon_dcube(lon_fn, 1, 1, mask)
    jac_tmp = np.zeros((2, 2))
    check1 = mask * (dlat_dx - cube_to_sphere_jacobian[:, :, :, 0, 0])
    check2 = mask * (dlat_dy - cube_to_sphere_jacobian[:, :, :, 1, 0])
    check3 = mask * (dlon_dx - cube_to_sphere_jacobian[:, :, :, 0, 1])
    check4 = mask * (dlon_dy - cube_to_sphere_jacobian[:, :, :, 1, 1])
    try:
      assert(np.max(check1) < 1e-3)
      assert(np.max(check2) < 1e-3)
      assert(np.max(check3) < 1e-3)
      assert(np.max(check4) < 1e-3)
    except:
      for face_idx in range(NFACES):
        if mask[face_idx]:
          i_idx, j_idx = (1, 1)
          jac_tmp[0, 0] = dlat_dx[face_idx, i_idx, j_idx]
          jac_tmp[0, 1] = dlon_dx[face_idx, i_idx, j_idx]
          jac_tmp[1, 0] = dlat_dy[face_idx, i_idx, j_idx]
          jac_tmp[1, 1] = dlon_dy[face_idx, i_idx, j_idx]
          print(f"Face: {face_idx}, Numerical jac: \n {jac_tmp}, \n Analytic jac: \n {cube_to_sphere_jacobian[face_idx, i_idx, j_idx, :, :]}")
  # front face
  front_lat = lambda x, y: (np.arctan2(y, np.sqrt(1 + x**2)))
  front_lon = lambda x, y: np.mod((np.arctan(x) + 2 * np.pi), 2 * np.pi)
  gll_latlon[:,:,:,0] += front_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * front_face_mask # (np.arctan2(cube_points[:, :, :, y_idx], np.sqrt(1 + cube_points[:, :, :, x_idx]**2))) * front_face_mask
  gll_latlon[:,:,:,1] += front_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * front_face_mask #np.mod((np.arctan(cube_points[:, :, :, x_idx]) + 2 * np.pi), 2 * np.pi) * front_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:,:,:,0],  gll_latlon[:,:,:,1], front_face_mask)
  if DEBUG:
    test_face(front_lat, front_lon, front_face_mask)

  # right face
  right_lat = lambda x, y: (np.arctan2(y, np.sqrt(1 + x**2)))
  right_lon = lambda x, y: np.arctan(x) + np.pi/2
  gll_latlon[:,:,:,0] += right_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * right_face_mask #(np.arctan2(cube_points[:, :, :, y_idx], np.sqrt(1 + cube_points[:, :, :, x_idx]**2))) * right_face_mask
  gll_latlon[:,:,:,1] += right_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * right_face_mask #(np.arctan(-cube_points[:, :, :, x_idx]) + np.pi/2) * right_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:,:,:,0], -np.pi/2 + gll_latlon[:,:,:,1], right_face_mask)
  if DEBUG:
    test_face(right_lat, right_lon, right_face_mask)

  # back face
  back_lat = lambda x, y: (np.arctan2(y, np.sqrt(1 + x**2)))
  back_lon = lambda x, y: (np.arctan(x) + np.pi )
  gll_latlon[:,:,:,0] += back_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * back_face_mask #(np.arctan2(cube_points[:, :, :, y_idx], np.sqrt(1 + cube_points[:, :, :, x_idx]**2))) * back_face_mask
  gll_latlon[:,:,:,1] += back_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * back_face_mask #(np.arctan(-cube_points[:, :, :, x_idx]) + np.pi ) * back_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:,:,:,0], -np.pi + gll_latlon[:,:,:,1], back_face_mask)
  if DEBUG:
    test_face(back_lat, back_lon, back_face_mask)

  # left face
  left_lat = lambda x, y: (np.arctan2(y, np.sqrt(1 + x**2)))
  left_lon = lambda x, y: (np.arctan(x) + 3 * np.pi/2)
  gll_latlon[:,:,:,0] += left_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * left_face_mask #(np.arctan2(cube_points[:, :, :, y_idx], np.sqrt(1 + cube_points[:, :, :, x_idx]**2))) * left_face_mask
  gll_latlon[:,:,:,1] += left_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * left_face_mask #(np.arctan(cube_points[:, :, :, x_idx]) + 3 * np.pi/2) * left_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:,:,:,0], -3 * np.pi/2 + gll_latlon[:,:,:,1], left_face_mask)
  if DEBUG:
    test_face(left_lat, left_lon, left_face_mask)

  # top face
  top_lat = lambda x, y: ( np.arctan2(1, np.sqrt(x**2 + y**2)))
  top_lon = lambda x, y: np.mod((np.arctan2(x, -y)), 2 * np.pi)
  gll_latlon[:,:,:,0] += top_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * top_face_mask #( np.arctan2(1, np.sqrt(cube_points[:, :, :, x_idx]**2 + cube_points[:, :, :, y_idx]**2))) * top_face_mask
  gll_latlon[:,:,:,1] += top_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * top_face_mask #np.mod((np.arctan2(cube_points[:, :, :, x_idx], cube_points[:, :, :, y_idx])), 2 * np.pi) * top_face_mask
  set_jac_pole(cube_to_sphere_jacobian, gll_latlon[:,:,:,0], gll_latlon[:,:,:,1], top_face_mask, 1.0)
  if DEBUG:
    test_face(top_lat, top_lon, top_face_mask)

  # bottom face
  bottom_lat = lambda x, y: ( -np.arctan2(1, np.sqrt(x**2 + y**2)))
  bottom_lon = lambda x, y: np.mod((np.arctan2(x, y)), 2 * np.pi)
  gll_latlon[:,:,:,0] += bottom_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * bottom_face_mask # ( -np.arctan2(1, np.sqrt(cube_points[:, :, :, x_idx]**2 + cube_points[:, :, :, y_idx]**2))) * bottom_face_mask
  gll_latlon[:,:,:,1] += bottom_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * bottom_face_mask # np.mod((np.arctan2(cube_points[:, :, :, x_idx], cube_points[:, :, :, y_idx])), 2 * np.pi) * bottom_face_mask
  set_jac_pole(cube_to_sphere_jacobian, gll_latlon[:,:,:,0], gll_latlon[:,:,:,1], bottom_face_mask, -1.0)

  if DEBUG:
    test_face(bottom_lat, bottom_lon, bottom_face_mask)

  cube_to_sphere_jacobian_inv = np.linalg.inv(cube_to_sphere_jacobian)
  return gll_latlon, cube_to_sphere_jacobian, cube_to_sphere_jacobian_inv



def generate_metric_terms(gll_latlon, cube_to_sphere_jacobian, cube_to_sphere_jacobian_inv, vert_redundancy_gll):
  gll_to_sphere_jacobian = np.einsum("fijpg,fijps->fijgs", cube_to_sphere_jacobian, gll_to_cube_jacobian)
  gll_to_sphere_jacobian[:, :, :, 1, :] *= np.cos(gll_latlon[:,:,:,0])[:, :, :, np.newaxis]
  gll_to_sphere_jacobian_inv = np.linalg.inv(gll_to_sphere_jacobian)

  rmetdet = np.linalg.det(gll_to_sphere_jacobian_inv)

  metdet = 1.0/rmetdet

  mass_mat = metdet.copy() * (g_weights[np.newaxis, :, np.newaxis] * g_weights[np.newaxis, np.newaxis, :]) # denominator for weighted sum

  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        mass_mat[remote_face_id, remote_i, remote_j] += metdet[local_face_idx, local_i, local_j] * (g_weights[local_i] * g_weights[local_j])

  inv_mass_mat = 1.0/mass_mat
  return gll_to_sphere_jacobian, gll_to_sphere_jacobian_inv, rmetdet, metdet, mass_mat, inv_mass_mat



