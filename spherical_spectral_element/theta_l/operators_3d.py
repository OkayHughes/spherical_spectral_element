from ..config import use_jax, partial, np
from ..operators import sphere_divergence, sphere_vorticity
from ..operators import sphere_gradient, sphere_laplacian_wk, sphere_vec_laplacian_wk


# use vmap if gpu jax, pmap if cpu jax, for loop otherwise

# currently violates design philosophy.
# Desired solution is implementing a dummy vmap that behaves simularly
if use_jax:
  from jax import vmap

  def sphere_divergence_3d(vector, h_grid, config):
    sph_op = partial(sphere_divergence, grid=h_grid, a=config["radius_earth"])
    return vmap(sph_op, in_axes=(-2), out_axes=(-1))(vector)

  def sphere_vorticity_3d(vector, h_grid, config):
    sph_op = partial(sphere_vorticity, grid=h_grid, a=config["radius_earth"])
    return vmap(sph_op, in_axes=(-2), out_axes=(-1))(vector)

  def sphere_laplacian_wk_3d(scalar, h_grid, config):
    sph_op = partial(sphere_laplacian_wk, grid=h_grid, a=config["radius_earth"])
    return vmap(sph_op, in_axes=(-1), out_axes=(-1))(scalar)

  def sphere_vec_laplacian_3d(vector, h_grid, config):
    sph_op = partial(sphere_vec_laplacian_wk, grid=h_grid, a=config["radius_earth"])
    return vmap(sph_op, in_axes=(-2), out_axes=(-2))(vector)

  def sphere_gradient_3d(scalar, h_grid, config):
    sph_op = partial(sphere_gradient, grid=h_grid, a=config["radius_earth"])
    return vmap(sph_op, in_axes=(-1), out_axes=(-2))(scalar)

else:
  def sphere_divergence_3d(vector, h_grid, config):
    levs = []
    for lev_idx in range(vector.shape[-2]):
      levs.append(sphere_divergence(vector[:, :, :, lev_idx, :], grid=h_grid, a=config["radius_earth"]))
    return np.stack(levs, axis=-1)

  def sphere_vorticity_3d(vector, h_grid, config):
    levs = []
    for lev_idx in range(vector.shape[-2]):
      levs.append(sphere_vorticity(vector[:, :, :, lev_idx, :], grid=h_grid, a=config["radius_earth"]))
    return np.stack(levs, axis=-1)

  def sphere_vec_laplacian_wk_3d(vector, h_grid, config):
    levs = []
    for lev_idx in range(vector.shape[-2]):
      levs.append(sphere_vec_laplacian_wk(vector[:, :, :, lev_idx, :], grid=h_grid, a=config["radius_earth"]))
    return np.stack(levs, axis=-2)

  def sphere_laplacian_wk_3d(scalar, h_grid, config):
    levs = []
    for lev_idx in range(scalar.shape[-1]):
      levs.append(sphere_laplacian_wk(scalar[:, :, :, lev_idx], grid=h_grid, a=config["radius_earth"]))
    return np.stack(levs, axis=-1)

  def sphere_gradient_3d(scalar, h_grid, config):
    levs = []
    for lev_idx in range(scalar.shape[-1]):
      levs.append(sphere_gradient(scalar[:, :, :, lev_idx], grid=h_grid, a=config["radius_earth"]))
    return np.stack(levs, axis=-2)
