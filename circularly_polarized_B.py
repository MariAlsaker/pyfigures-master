import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

w = 33.78E6*2*np.pi
T = 1/w * 2*np.pi
t = np.linspace(0, T, 10, endpoint=False)
x = np.cos(w*t)
y = np.cos(w*t+np.pi/2)# Faseforskyvning 85 grader i stedet for 90: *0.94444)
figure = plt.figure(figsize=[8,8])
ax = figure.add_subplot(111)
ax.hlines([-1.5], xmin=[-2], xmax=[+2], colors=["k"])
ax.vlines([-1.5], ymin=[-2], ymax=[+2], colors=["k"])
cmap = mpl.cm.get_cmap("plasma")
c_num = np.linspace(0,1,len(t), endpoint=False)
colors = [cmap(num) for num in c_num]
ypos = 1.76
for i, val in enumerate(t):
    qv = ax.quiver(0, 0, x[i], y[i], color = colors[i], scale_units="xy", angles="xy", scale=1)
    ax.quiverkey(qv, 1.6, ypos, 0.4, f"B-field t = {val*1E9:.1f}ns,\n (x,y)=({x[i]:.1f},{y[i]:.1f})", coordinates="data")
    ypos = ypos-0.4
ax.set_title("Field of quadrature coil")
ax.set_xlabel("x-direction")
ax.set_ylabel("y-direction")
plt.legend()
plt.show()