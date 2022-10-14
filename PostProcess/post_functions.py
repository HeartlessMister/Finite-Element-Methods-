import numpy as np
import math


def assign_BCs(NL, BC_flag, defV):
    NoN = np.size(NL, 0)
    PD = np.size(NL, 1)

    ENL = np.zeros([NoN, 6 * PD])

    ENL[:, 0:PD] = NL

    if BC_flag == "extension":

        for i in range(0, NoN):

            if ENL[i, 0] == 0:

                ENL[i, 2] = -1
                ENL[i, 3] = -1
                ENL[i, 8] = -defV
                ENL[i, 9] = 0

            elif ENL[i, 0] == 1:

                ENL[i, 2] = -1
                ENL[i, 3] = -1
                ENL[i, 8] = defV
                ENL[i, 9] = 0
            else:
                ENL[i, 2] = 1
                ENL[i, 3] = 1

                ENL[i, 10] = 0
                ENL[i, 11] = 0

    if BC_flag == "expansion":

        for i in range(0, NoN):

            if ENL[i, 0] == 0 or ENL[i, 0] == 1 or ENL[i, 1] == 0 or ENL[i, 1] == 1:

                ENL[i, 2] = -1
                ENL[i, 3] = -1

                ENL[i, 8] = defV * ENL[i, 0]
                ENL[i, 9] = defV * ENL[i, 1]

            else:
                ENL[i, 2] = 1
                ENL[i, 3] = 1

                ENL[i, 10] = 0
                ENL[i, 11] = 0

    if BC_flag == "shear":

        for i in range(0, NoN):

            if ENL[i, 1] == 0:

                ENL[i, 2] = -1
                ENL[i, 3] = -1

                ENL[i, 8] = 0
                ENL[i, 9] = 0

            elif ENL[i, 1] == 1:

                ENL[i, 2] = -1
                ENL[i, 3] = -1

                ENL[i, 8] = defV
                ENL[i, 9] = 0

            else:
                ENL[i, 2] = 1
                ENL[i, 3] = 1

                ENL[i, 10] = 0
                ENL[i, 11] = 0

    # This is actually assign BC function

    DOFs = 0
    DOCs = 0

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == -1:
                DOCs -= 1
                ENL[i, 2 * PD + j] = DOCs
            else:
                DOFs += 1
                ENL[i, 2 * PD + j] = DOFs

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, 2 * PD + j] < 0:
                ENL[i, 3 * PD + j] = abs(ENL[i, 2 * PD + j]) + DOFs
            else:
                ENL[i, 3 * PD + j] = abs(ENL[i, 2 * PD + j])

    DOCs = abs(DOCs)

    return (ENL, DOFs, DOCs)


def assemble_displacements(ENL, NL):
    NoN = np.size(NL, 0)
    PD = np.size(NL, 1)
    DOC = 0
    Up = []

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == -1:
                DOC += 1
                Up.append(ENL[i, 4 * PD + j])

    Up = np.vstack([Up]).reshape(-1, 1)
    return Up


def assemble_forces(ENL, NL):
    NoN = np.size(NL, 0)
    PD = np.size(NL, 1)
    DOF = 0
    Fp = []

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == 1:
                DOF += 1
                Fp.append(ENL[i, 5 * PD + j])

    Fp = np.vstack([Fp]).reshape(-1, 1)
    return Fp


def update_nodes(ENL, U_u, Fu, NL):
    NoN = np.size(NL, 0)
    PD = np.size(NL, 1)
    DOFs = 0
    DOCs = 0

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == 1:
                DOFs += 1
                ENL[i, 4 * PD + j] = U_u[DOFs - 1]
            else:
                DOCs += 1
                ENL[i, 5 * PD + j] = Fu[DOCs - 1]
    return ENL


def element_stiffness(n1, NL):
    NPE = np.size(n1, 0)
    PD = np.size(NL, 1)

    x = np.zeros([NPE, PD])
    x[0:NPE, 0:PD] = NL[n1[0:NPE] - 1, 0:PD]

    K = np.zeros([NPE * PD, NPE * PD])

    coor = x.T

    if NPE == 3:
        GPE = 1

    if NPE == 4:
        GPE = 4

    for i in range(1, NPE + 1):
        for j in range(1, NPE + 1):

            k = np.zeros([PD, PD])

            for gp in range(1, GPE + 1):

                (xi, eta, alpha) = GaussPoint(NPE, GPE, gp)

                grad_nat = grad_N_nat(NPE, xi, eta)

                J = coor @ grad_nat.T  # np.matmul(.)

                grad = np.linalg.inv(J).T @ grad_nat

                for a in range(1, PD + 1):
                    for c in range(1, PD + 1):
                        for b in range(1, PD + 1):
                            for d in range(1, PD + 1):
                                # Impelement 2 if blocks one for triangular elements and another for quadrangular elements.

                                k[a - 1, c - 1] = k[a - 1, c - 1] + grad[b - 1, i - 1] * constitutive(a, b, c, d) * \
                                                  grad[d - 1, j - 1] * np.linalg.det(J) * alpha

            K[((i - 1) * PD + 1) - 1:i * PD, ((j - 1) * PD + 1) - 1:j * PD] = k

    return K

def assemble_stiffness(ENL, EL, NL):
    NoE = np.size(EL, 0)
    NPE = np.size(EL, 1)
    NoN = np.size(NL, 0)
    PD = np.size(NL, 1)

    K = np.zeros([NoN * PD, NoN * PD])

    for i in range(1, NoE + 1):
        n1 = EL[i - 1, 0:NPE]
        k = element_stiffness(n1, NL)
        for r in range(0, NPE):
            for p in range(0, PD):
                for q in range(0, NPE):
                    for s in range(0, PD):
                        row = ENL[n1[r] - 1, p + 3 * PD]
                        column = ENL[n1[q] - 1, s + 3 * PD]
                        value = k[r * PD + p, q * PD + s]
                        K[int(row) - 1, int(column) - 1] = K[int(row) - 1, int(column) - 1] + value
    return K




def GaussPoint(NPE, GPE, gp):
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

def void_mesh(d1, d2, p, m, R, element_type):
    PD = 2
    q = np.array([[0, 0], [d1, 0], [0, d2], [d1, d2]])
    NoN = 2 * (p + 1) * (m + 1) + 2 * (p - 1) * (m + 1)
    NoE = 4 * p * m
    NPE = 4

    # Nodes
    NL = np.zeros([NoN, PD])
    a = (q[1, 0] - q[0, 0]) / p
    b = (q[2, 1] - q[0, 1]) / p

    # Region 1
    coor11 = np.zeros([(p + 1) * (m + 1), PD])
    for i in range(1, p + 2):
        coor11[i - 1, 0] = q[0, 0] + (i - 1) * a
        coor11[i - 1, 1] = q[0, 1]

    for i in range(1, p + 2):
        coor11[m * (p + 1) + i - 1, 0] = R * np.cos((5 * math.pi / 4) + (i - 1) * ((math.pi / 2) / p)) + d1 / 2
        coor11[m * (p + 1) + i - 1, 1] = R * np.sin((5 * math.pi / 4) + (i - 1) * ((math.pi / 2) / p)) + d2 / 2

    for i in range(1, m):
        for j in range(1, p + 2):
            dx = (coor11[m * (p + 1) + j - 1, 0] - coor11[j - 1, 0]) / m
            dy = (coor11[m * (p + 1) + j - 1, 1] - coor11[j - 1, 1]) / m

            coor11[i * (p + 1) + j - 1, 0] = coor11[(i - 1) * (p + 1) + j - 1, 0] + dx
            coor11[i * (p + 1) + j - 1, 1] = coor11[(i - 1) * (p + 1) + j - 1, 1] + dy

    # Region 2
    coor22 = np.zeros([(p + 1) * (m + 1), PD])
    for i in range(1, p + 2):
        coor22[i - 1, 0] = q[2, 0] + (i - 1) * a
        coor22[i - 1, 1] = q[2, 1]

    for i in range(1, p + 2):
        coor22[m * (p + 1) + i - 1, 0] = R * np.cos((3 * math.pi / 4) - (i - 1) * ((math.pi / 2) / p)) + d1 / 2
        coor22[m * (p + 1) + i - 1, 1] = R * np.sin((3 * math.pi / 4) - (i - 1) * ((math.pi / 2) / p)) + d2 / 2

    for i in range(1, m):
        for j in range(1, p + 2):
            dx = (coor22[m * (p + 1) + j - 1, 0] - coor22[j - 1, 0]) / m
            dy = (coor22[m * (p + 1) + j - 1, 1] - coor22[j - 1, 1]) / m

            coor22[i * (p + 1) + j - 1, 0] = coor22[(i - 1) * (p + 1) + j - 1, 0] + dx
            coor22[i * (p + 1) + j - 1, 1] = coor22[(i - 1) * (p + 1) + j - 1, 1] + dy

    # Region 3
    coor33 = np.zeros([(p - 1) * (m + 1), PD])
    for i in range(1, p):
        coor33[i - 1, 0] = q[0, 0]
        coor33[i - 1, 1] = q[0, 1] + i * b

    for i in range(1, p):
        coor33[m * (p - 1) + i - 1, 0] = R * np.cos((5 * math.pi / 4) - i * ((math.pi / 2) / p)) + d1 / 2
        coor33[m * (p - 1) + i - 1, 1] = R * np.sin((5 * math.pi / 4) - i * ((math.pi / 2) / p)) + d2 / 2

    for i in range(1, m):
        for j in range(1, p):
            dx = (coor33[m * (p - 1) + j - 1, 0] - coor33[j - 1, 0]) / m
            dy = (coor33[m * (p - 1) + j - 1, 1] - coor33[j - 1, 1]) / m

            coor33[i * (p - 1) + j - 1, 0] = coor33[(i - 1) * (p - 1) + j - 1, 0] + dx
            coor33[i * (p - 1) + j - 1, 1] = coor33[(i - 1) * (p - 1) + j - 1, 1] + dy

    # Region 4
    coor44 = np.zeros([(p - 1) * (m + 1), PD])
    for i in range(1, p):
        coor44[i - 1, 0] = q[1, 0]
        coor44[i - 1, 1] = q[1, 1] + i * b

    for i in range(1, p):
        coor44[m * (p - 1) + i - 1, 0] = R * np.cos((7 * math.pi / 4) + i * ((math.pi / 2) / p)) + d1 / 2
        coor44[m * (p - 1) + i - 1, 1] = R * np.sin((7 * math.pi / 4) + i * ((math.pi / 2) / p)) + d2 / 2

    for i in range(1, m):
        for j in range(1, p):
            dx = (coor44[m * (p - 1) + j - 1, 0] - coor44[j - 1, 0]) / m
            dy = (coor44[m * (p - 1) + j - 1, 1] - coor44[j - 1, 1]) / m

            coor44[i * (p - 1) + j - 1, 0] = coor44[(i - 1) * (p - 1) + j - 1, 0] + dx
            coor44[i * (p - 1) + j - 1, 1] = coor44[(i - 1) * (p - 1) + j - 1, 1] + dy

    # Reordering the nodes
    for i in range(1, m + 2):
        NL[(i - 1) * 4 * p: i * 4 * p, :] = np.vstack([
            coor11[(i - 1) * (p + 1): i * (p + 1), :],
            coor44[(i - 1) * (p - 1): i * (p - 1), :],
            np.flipud(coor22[(i - 1) * (p + 1): i * (p + 1), :]),
            np.flipud(coor33[(i - 1) * (p - 1): i * (p - 1), :]),
        ])

    # Element
    EL = np.zeros([NoE, NPE])

    for i in range(1, m + 1):
        for j in range(1, 4 * p + 1):
            if j == 1:
                EL[(i - 1) * (4 * p) + j - 1, 0] = (i - 1) * (4 * p) + j
                EL[(i - 1) * (4 * p) + j - 1, 1] = EL[(i - 1) * (4 * p) + j - 1, 0] + 1
                EL[(i - 1) * (4 * p) + j - 1, 3] = EL[(i - 1) * (4 * p) + j - 1, 0] + 4 * p
                EL[(i - 1) * (4 * p) + j - 1, 2] = EL[(i - 1) * (4 * p) + j - 1, 3] + 1
            elif j == 4 * p:
                EL[(i - 1) * (4 * p) + j - 1, 0] = i * 4 * p
                EL[(i - 1) * (4 * p) + j - 1, 1] = (i - 1) * (4 * p) + 1
                EL[(i - 1) * (4 * p) + j - 1, 2] = EL[(i - 1) * (4 * p) + j - 1, 0] + 1
                EL[(i - 1) * (4 * p) + j - 1, 3] = EL[(i - 1) * (4 * p) + j - 1, 0] + 4 * p
            else:
                EL[(i - 1) * (4 * p) + j - 1, 0] = EL[(i - 1) * (4 * p) + j - 2, 1]
                EL[(i - 1) * (4 * p) + j - 1, 3] = EL[(i - 1) * (4 * p) + j - 2, 2]
                EL[(i - 1) * (4 * p) + j - 1, 2] = EL[(i - 1) * (4 * p) + j - 1, 3] + 1
                EL[(i - 1) * (4 * p) + j - 1, 1] = EL[(i - 1) * (4 * p) + j - 1, 0] + 1

    if element_type == "D2TR3N":
        NPE_new = 3
        NoE_new = 2 * NoE
        EL_new = np.zeros([NoE_new, NPE_new])

        for i in range(1, NoE + 1):
            EL_new[2 * (i - 1), 0] = EL[i - 1, 0]
            EL_new[2 * (i - 1), 1] = EL[i - 1, 1]
            EL_new[2 * (i - 1), 2] = EL[i - 1, 2]

            EL_new[2 * (i - 1) + 1, 0] = EL[i - 1, 0]
            EL_new[2 * (i - 1) + 1, 1] = EL[i - 1, 2]
            EL_new[2 * (i - 1) + 1, 2] = EL[i - 1, 3]

        EL = EL_new

    EL = EL.astype(int)

    return NL, EL