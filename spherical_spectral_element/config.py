import numpy as np


DEBUG = True
npt = 4

use_jax = True
use_cpu = True
use_double = True

if use_double:
  eps = 1e-11
else:
  eps = 1e-6

if use_jax:
  import jax.numpy as jnp
  import jax
  if use_cpu:
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
  if use_double:
    jax.config.update("jax_enable_x64", True)

  def jax_wrapper(x):
    return jnp.array(x)

  def jax_unwrapper(x):
    return np.asarray(x)
  jit = jax.jit

  def versatile_assert(should_be_true):
    return

  from jax.tree_util import Partial as partial
else:
  import numpy as jnp

  def jax_wrapper(x):
    return x

  def jax_unwrapper(x):
    return x

  def jit(func, *_, **__):
    return func

  def versatile_assert(should_be_true):
    assert should_be_true

  from functools import partial
