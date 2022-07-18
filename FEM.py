import numpy as np
from functions import *
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

startTime = datetime.now()

d1 = 1
d2 = 2
p = 9
m = 9
R = 0.2
element_type = 'D2TR3N'  # D2QU4N D2TR3N
defV = 0.1

NL, EL = void_mesh(d1, d2, p, m, R, element_type)

BC_flag = 'extension'  # extension expansion shear

(ENL, DOFs, DOCs) = assign_BCs(NL, BC_flag, defV)

K = assemble_stiffness1(ENL, EL, NL)

Fp = assemble_forces1(ENL, NL)
Up = assemble_displacements1(ENL, NL)

K_reduced = K[0:DOFs, 0:DOFs]  # K_UU in 1D
K_UP = K[0:DOFs, DOFs:DOCs + DOFs]
K_PU = K[DOFs:DOCs + DOFs, 0:DOFs]
K_PP = K[DOFs:DOCs + DOFs, DOFs:DOCs + DOFs]

F = Fp - (K_UP @ Up)  # np.matmul()
Uu = np.linalg.solve(K_reduced, F)  # Shortcut
Fu = (K_PU @ Uu) + (K_PP @ Up)

ENL = update_nodes(ENL, Uu, Fu, NL)

###########################################################

(stress_xx, stress_xy, stress_yx, stress_yy,
 strain_xx, strain_xy, strain_yx, strain_yy,
 disp_x, disp_y, X, Y) = post_process(NL, EL, ENL)

# Normalize color values for the colormap

# The below code only for xx values but we can copy paste for yy and etc. to plot others

# TODO this is x
# stress_xxNormalized = (stress_xx - stress_xx.min()) / (stress_xx.max() - stress_xx.min())
# disp_xNormalized = (disp_x - disp_x.min()) / (disp_x.max() - disp_x.min())

# TODO this is y
stress_xxNormalized = (stress_yy - stress_yy.min()) / (stress_yy.max() - stress_yy.min())
disp_xNormalized = (disp_y - disp_y.min()) / (disp_y.max() - disp_y.min())


# Plot for each element

# import matplotlib.colors as mcolors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N  # what is this .N ???

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap


fig_1 = plt.figure(1)
plt.title('Stress XX')
axstress_xx = fig_1.add_subplot(111)

for i in range(np.size(EL, 0)):
    x = X[:, i]
    y = Y[:, i]
    c = stress_xxNormalized[:, i]

    # Setting colormap boundaries
    cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())

    # Plot the colors
    t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')

    # t = axstress_xx.tricontourf(x, y, c, cmap=cmap, levels=10)

    # Plot the blacklines 'k-' is the black line

    p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)

    # In our project we need to be able to close on and off those blacklines
    # In our project we need to be able to change colormap
    # those functions are on line 112-115

    """ 
    Some notes from Mert Şölen:
    
    You can collect the lines and colors as variables to turn them on/off in your GUI
    
    For instance:
    
    -I collected black lines (edge colors) in the plist ---> used to display/hide black lines in another function
    
    -I collected colormaps for each element in cmaplis ---> used to redraw the plot when exaggeration and when color map changes
    
    -I collected color values for each element in triplist ---> used to redraw the plot when colormap changes
    
    -I collected the min and max values in mmList ---> Used to redraw the plot when colormap changes
    
    Reach me out if you need support!
    
    """

    # pList.append(p)
    # cmapList.append(cmap)
    # tripList.append(t)
    # mmList.append((c.min(), c.maiyiyimx()))


fig_2 = plt.figure(2)
plt.title('Displacement X')
axdisp_x = fig_2.add_subplot(111)

for i in range(np.size(EL, 0)):
    x = X[:, i]
    y = Y[:, i]
    c = disp_xNormalized[:, i]
    cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
    t = axdisp_x.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
    # t = axdisp_x.tricontourf(x,y,c, cmap = cmap, levels= 10)

    p = axdisp_x.plot(x, y, 'k', linewidth=0.5)

plt.show()

print(datetime.now() - startTime)
