import numpy as np
import math
from post_functions import *


def post_process(NL, EL, ENL):
    PD = np.size(NL, 1)
    NoE = np.size(EL, 0)
    NPE = np.size(EL, 1)

    scale = 1  # magnify the deflection

    disp, stress, strain = element_post_process(NL, EL, ENL)

    stress_xx = np.zeros([NPE, NoE])
    stress_xy = np.zeros([NPE, NoE])
    stress_yx = np.zeros([NPE, NoE])
    stress_yy = np.zeros([NPE, NoE])

    strain_xx = np.zeros([NPE, NoE])
    strain_xy = np.zeros([NPE, NoE])
    strain_yx = np.zeros([NPE, NoE])
    strain_yy = np.zeros([NPE, NoE])

    disp_x = np.zeros([NPE, NoE])
    disp_y = np.zeros([NPE, NoE])

    X = np.zeros([NPE, NoE])
    Y = np.zeros([NPE, NoE])

    if NPE in [3, 4]:  # D2QU4N and D2TR3N #add others such as 8 or 9, look at line 59 and 75

        # Calculate X and Y coordinates of the nodes

        X = ENL[EL - 1, 0] + scale * ENL[EL - 1, 4 * PD]
        Y = ENL[EL - 1, 1] + scale * ENL[EL - 1, 4 * PD + 1]

        X = X.T
        Y = Y.T

        # [[xx, xy],
        # [yx,yy]

        # [:,:] is to compensate for D2TR3N since it only has 1 gauss

        stress_xx[:, :] = stress[:, :, 0, 0].T
        stress_xy[:, :] = stress[:, :, 0, 1].T
        stress_yx[:, :] = stress[:, :, 1, 0].T
        stress_yy[:, :] = stress[:, :, 1, 1].T

        strain_xx[:, :] = strain[:, :, 0, 0].T
        strain_xy[:, :] = strain[:, :, 0, 1].T
        strain_yx[:, :] = strain[:, :, 1, 0].T
        strain_yy[:, :] = strain[:, :, 1, 1].T

        disp_x = disp[:, :, 0, 0].T
        disp_y = disp[:, :, 1, 0].T

        # You may need to write other if blocks for D2QU8N, D2QU9N, D2TR6N.
        # For those part we will code it ourselfs.

        return (stress_xx, stress_xy, stress_yx, stress_yy,
                strain_xx, strain_xy, strain_yx, strain_yy,
                disp_x, disp_y, X, Y)


def element_post_process(NL, EL, ENL):
    PD = np.size(NL, 1)
    NoE = np.size(EL, 0)
    NPE = np.size(EL, 1)

    GPE = 1
    if NPE == 3:  # D2TR3N
        GPE = 1

    if NPE == 4:  # D2QU4N
        GPE = 4

    if NPE == 6:
        GPE = 3

    if NPE == 8 or 9:
        GPE = 9

    # Here add other if blocks for D2QU8N, D2QU9N, D2TR6N

    # NoE: Specifies the element I'm looking at
    # NPE: Specifies the node on the element I'm looking at
    # PD: Displacement is a vector that has x and y components
    # Specifies the direction I'm looking
    # 1: Added to make it similar to stress and strain matrices

    disp = np.zeros([NoE, NPE, PD, 1])  # Disp is written on the nodes
    # Stress and strain are noramlly supposed to mapped on the Gauss Points
    # However, due to shortcomings of Python and MATLAB, we will cgeat
    # by mapping these on the corners as well.

    # Briefly, stress&strain are calculated on the Gauss Points

    stress = np.zeros([NoE, GPE, PD, PD])
    strain = np.zeros([NoE, GPE, PD, PD])

    for e in range(1, NoE + 1):  # First, find the displacements. By using
        n1 = EL[e - 1, 0: NPE]

        # Assign displacements to the corresponding nodes

        for i in range(1, NPE + 1):
            for j in range(1, PD + 1):
                disp[e - 1, i - 1, j - 1, 0] = ENL[n1[i - 1] - 1, 4 * PD + j - 1]

        # Specify the corners of the elements(Just like in the stiffness calculayions)

        x = np.zeros([NPE, PD])
        x[0:NPE, 0:PD] = NL[n1[0:NPE] - 1, 0:PD]

        # Specify the displacements for these corners [Required for strain]

        u = np.zeros([PD, NPE])
        for i in range(1, NPE + 1):
            for j in range(1, PD + 1):
                u[j - 1, i - 1] = ENL[n1[i - 1] - 1, 4 * PD + j - 1]

        # Coordinates of the corners trnasposed (Just like in the stiffness calculations)
        coor = x.T

        # Going over the gauss points since every gauss point will have their own

        for gp in range(1, GPE + 1):

            # String for each gauss point (2x2 matrix)
            epsilon = np.zeros([PD, PD])

            # Going over the nodes in the element
            for i in range(1, NPE + 1):
                # The process for the shape functions is the same is in stiffness
                J = np.zeros([PD, PD])

                grad = np.zeros([PD, NPE])

                (xi, eta, alpha) = GaussPoint(NPE, GPE, gp)

                grad_nat = grad_N_nat(NPE, xi, eta)

                J = coor @ grad_nat.T

                grad = np.linalg.inv(J).T @ grad_nat

                # Calculate strain
                # Define dyadic in another function
                epsilon = epsilon + 1 / 2 * (dyad(grad[:, i - 1], u[:, i - 1]) + dyad(u[:, i - 1], grad[:, i - 1])
                                             ) + dyad(u[:, i - 1], grad[:, i - 1])

        # Initialize stress as a 2x2 matrix

        sigma = np.zeros([PD, PD])

        # The same logic as in the stiffness calculation (4 directions)
        # sigma = E * epsilon

        for a in range(1, PD + 1):
            for b in range(1, PD + 1):
                for c in range(1, PD + 1):
                    for d in range(1, PD + 1):
                        sigma[a - 1, b - 1] = sigma[a - 1, b - 1] + constitutive(a, b, c, d
                                                                                 ) * epsilon[c - 1, d - 1]

        # Compile the results. Remember the sizes of stress and strain matrix

        for a in range(1, PD + 1):
            for b in range(1, PD + 1):
                strain[e - 1, gp - 1, a - 1, b - 1] = epsilon[a - 1, b - 1]
                strain[e - 1, gp - 1, a - 1, b - 1] = sigma[a - 1, b - 1]

    return disp, stress, strain


def dyad(u, v):
    # Takes two matrices
    # Shpaes them to row matrices
    u = u.reshape(len(u), 1)  # Mert writes as u = u.reshape(len(v),1) which is same since len v and u are the same
    v = u.reshape(len(v), 1)

    PD = 2

    # Matrix multiplication
    A = u @ v.T

    return A

# def GaussPoint(NPE, GPE, gp):
# He does not show below this point, this is from probably
# previous scripts in Computations file functions script, where we define GuassPoints
# grad_N_nat and constitutive are also from there.
# I add the file to the script. You can import it yourself.

def GaussPoint(NPE, GPE, gp):
    if NPE == 3:
        if GPE == 1:
            if gp == 1:
                xi = 1 / 3
                eta = 1 / 3
                alpha = 1
    if NPE == 6:

        if GPE == 3:
            if gp == 1:
                xi = 1 / 6
                eta = 1 / 6
                alpha = 1 / 3

            if gp == 2:
                xi = 4 / 6
                eta = 1 / 6
                alpha = 1 / 3

            if gp == 3:
                xi = 1 / 6
                eta = 4 / 6
                alpha = 1 / 3

    if NPE == 4:
        if GPE == 1:
            if gp == 1:
                xi = 0
                eta = 0
                alpha = 4

        if GPE == 4:
            if gp == 1:
                xi = -1 / math.sqrt(3)
                eta = -1 / math.sqrt(3)
                alpha = 1

            if gp == 2:
                xi = 1 / math.sqrt(3)
                eta = -1 / math.sqrt(3)
                alpha = 1

            if gp == 3:
                xi = 1 / math.sqrt(3)
                eta = 1 / math.sqrt(3)
                alpha = 1

            if gp == 4:
                xi = -1 / math.sqrt(3)
                eta = 1 / math.sqrt(3)
                alpha = 1

    if NPE == 8 or 9:

        if GPE == 9:

            if gp == 1:
                xi = -math.sqrt(3 / 5)
                eta = -math.sqrt(3 / 5)
                alpha = 25 / 81

            if gp == 2:
                xi = math.sqrt(3 / 5)
                eta = -math.sqrt(3 / 5)
                alpha = 25 / 81

            if gp == 3:
                xi = math.sqrt(3 / 5)
                eta = math.sqrt(3 / 5)
                alpha = 25 / 81

            if gp == 4:
                xi = -math.sqrt(3 / 5)
                eta = math.sqrt(3 / 5)
                alpha = 25 / 81

            if gp == 5:
                xi = 0
                eta = -math.sqrt(3 / 5)
                alpha = 40 / 81

            if gp == 6:
                xi = math.sqrt(3 / 5)
                eta = 0
                alpha = 40 / 81

            if gp == 7:
                xi = 0
                eta = math.sqrt(3 / 5)
                alpha = 40 / 81

            if gp == 8:
                xi = -math.sqrt(3 / 5)
                eta = 0
                alpha = 40 / 81

            if gp == 9:
                xi = 0
                eta = 0
                alpha = 64 / 81

    return (xi, eta, alpha)


def grad_N_nat(NPE, xi, eta):
    PD = 2
    result = np.zeros([PD, NPE])

    if NPE == 3:
        result[0, 0] = 1
        result[0, 1] = 0
        result[0, 2] = -1

        result[1, 0] = 0
        result[1, 1] = 1
        result[1, 2] = -1

    if NPE == 4:
        result[0, 0] = -1 / 4 * (1 - eta)
        result[0, 1] = 1 / 4 * (1 - eta)
        result[0, 2] = 1 / 4 * (1 + eta)
        result[0, 3] = -1 / 4 * (1 + eta)

        result[1, 0] = -1 / 4 * (1 - xi)
        result[1, 1] = -1 / 4 * (1 + xi)
        result[1, 2] = 1 / 4 * (1 + xi)
        result[1, 3] = 1 / 4 * (1 - xi)

    if NPE == 8:
        result[0, 0] = 1 / 4 * (1 - eta)*(2*xi+eta)
        result[0, 1] = 1 / 4 * (1 - eta)*(2*xi-eta)
        result[0, 2] = 1 / 4 * (1 + eta)*(2*xi+eta)
        result[0, 3] = 1 / 4 * (1 + eta)*(2*xi-eta)
        result[0, 4] = -xi*(1-eta)
        result[0, 5] = 1 / 2 * (1 - eta)*(1+eta)
        result[0, 6] = -xi*(1+eta)
        result[0, 7] = -1 / 2 * (1 - eta)*(1+eta)

        result[1, 0] = 1 / 4 * (1 - xi)*(xi+2*eta)
        result[1, 1] = 1 / 4 * (1 + xi)*(-xi-eta)
        result[1, 2] = 1 / 4 * (1 + xi)*(xi+eta)
        result[1, 3] = 1 / 4 * (1 - xi)*(-eta)
        result[1, 5] = -xi*(1-eta)
        result[1, 6] = 1 / 2 * (1 - eta)*(1+eta)
        result[1, 7] = -xi*(1+eta)
        result[1, 8] = -1 / 2 * (1 - eta)*(1+eta)





    return result


def constitutive(i, j, k, l):
    E = 8 / 3
    nu = 1 / 3

    C = (E / (2 * (1 + nu))) * (delta(i, l) * delta(j, k) + delta(i,
                                                                  k) * delta(j, l)) + (E * nu) / (1 - nu ** 2) * delta(
        i, j) * delta(k, l)

    return C


def delta(i, j):
    if i == j:
        delta = 1
    else:
        delta = 0

    return delta