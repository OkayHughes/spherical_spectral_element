
def init_config(Rgas=287.0,
                radius_earth=6371e3,
                period_earth=7.292e-5,
                gravity=9.81,
                p0=1e5,
                cp=1005.0,
                Rvap=461.50):
  return {"Rgas": Rgas,
          "Rvap": Rvap,
          "cp": cp,
          "gravity": gravity,
          "radius_earth": radius_earth,
          "period_earth": period_earth,
          "p0": p0}
