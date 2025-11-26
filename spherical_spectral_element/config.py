import numpy as np


DEBUG = True
npt = 4

is_jax = True
use_double = True

if use_double:
  eps = 1e-8
else:
  eps = 1e-6

if is_jax:
  import jax.numpy as jnp
  import jax
  print(jax.devices("cpu"))
  # device = jax.devices("METAL")[0]
  jax.config.update("jax_default_device", jax.devices("cpu")[0])
  # jax.config.update("jax_default_device", jax.devices("METAL")[0])
  jax.config.update("jax_enable_x64", True)

  def jax_wrapper(x):
    return jnp.array(x)

  def jax_unwrapper(x):
    return np.asarray(x)
  jit = jax.jit
else:
  import numpy as jnp

  def jax_wrapper(x):
    return x

  def jax_unwrapper(x):
    return x

  def jit(func, *_, **__):
    return func
