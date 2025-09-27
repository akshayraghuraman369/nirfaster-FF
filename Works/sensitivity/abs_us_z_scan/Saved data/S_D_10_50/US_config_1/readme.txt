# UltraSound pos and size

x_cen,y_cen,z_cen = [30,30, z]  #centre of the US, z(depth) is varied
height = 5  #height of cylinder
dia = 4     #diameter of cylinder


# Source-detector position

mesh.source = ff.base.optode(np.array([10,30,57]))
mesh.meas = ff.base.optode(np.array([50,30,57]))


# US freq and pressure

fa = 2e6  # frequency of US (Hz)
P0 = 1e6  # US pressue(pascal) , this value is converted into N/mm2 before assigning the pressure value to nodes