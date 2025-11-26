# potentially cupy could be loaded as np, or we could 
# write a wrapper around torch that behaves how we would expect numpy to
import numpy as np
import numpy as jnp
DEBUG=True
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
  #device = jax.devices("METAL")[0]
  jax.config.update("jax_default_device", jax.devices("cpu")[0])
  #jax.config.update("jax_default_device", jax.devices("METAL")[0])
  jax.config.update("jax_enable_x64", True)
  jax_wrapper = lambda x: jnp.array(x)
  jax_unwrapper = lambda x: np.asarray(x)
  jit = jax.jit
else:
  jax_wrapper = lambda x: x
  jax_unwrapper = lambda x: x
  def jit(func, *_, **__):
    return func