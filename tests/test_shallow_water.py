
if TESTING:
  u0 = 2 * np.pi * rearth / (12 * 24 * 60 * 60)
  h0 = 2.94e4/g
  def williamson_tc2_u(lat, lon):
    wind = np.zeros((*lat.shape, 2))
    wind[:,:,:,0] =  u0 * np.cos(lat)
    return wind
  def williamson_tc2_h(lat, lon):
    h = np.zeros_like(lat)
    h += h0
    h -= (rearth * earth_period * u0 + u0**2/2.0)/g * np.sin(lat)**2
    return h
  def williamson_tc2_hs(lat, lon):
    return np.zeros_like(lat)
  model = shallow_water_model()
  T = 2000.0
  t, final_state = model.simulate(T, williamson_tc2_u, williamson_tc2_h, williamson_tc2_hs)
  plt.figure()
  plt.title("U at time {t}")
  plt.tricontourf(gll_latlon[:,:,:,1].flatten(), gll_latlon[:,:,:,0].flatten(), final_state["u"][:,:,:,0].flatten())
  plt.colorbar()
  plt.show()
  plt.figure()
  plt.title("V at time {t}")
  plt.tricontourf(gll_latlon[:,:,:,1].flatten(), gll_latlon[:,:,:,0].flatten(), final_state["u"][:,:,:,1].flatten())
  plt.colorbar()
  plt.show()
  plt.figure()
  plt.title("h at time {t}")
  plt.tricontourf(gll_latlon[:,:,:,1].flatten(), gll_latlon[:,:,:,0].flatten(), final_state["h"].flatten())
  plt.colorbar()
  plt.show()

