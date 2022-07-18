import math
import numpy as np


def assign_bcs(NL, ENL):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    DOFs = 0
    DOCs = 0

    for i in range(0, NoN):
        for j in range(0, PD):
            # Diriclet
            if ENL[i, PD + j] == -1:
                DOCs -= 1
                ENL[i, 2 * PD + j] = DOCs
            # Neuman
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

    return ENL, DOFs, DOCs


def assemble_stiffness(ENL, EL, NL, E, A):
    NoE = np.size(EL, 0)
    NPE = np.size(EL, 1)
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    K = np.zeros([NoN * PD, NoN * PD])
    for i in range(0, NoE):
        nl = EL[i, 0:NPE]
        k = element_stiffness(nl, ENL, E, A)
        for r in range(0, NPE):
            for p in range(0, PD):
                for q in range(0, NPE):
                    for s in range(0, PD):
                        row = ENL[nl[r] - 1, p + 3 * PD]
                        col = ENL[nl[q] - 1, s + 3 * PD]
                        val = k[r * PD + p, q * PD + s]
                        K[int(row) - 1, int(col) - 1] = K[int(row) - 1, int(col) - 1] + val

    return K


def element_stiffness(nl, ENL, E, A):
    X1 = ENL[nl[0] - 1, 0]
    Y1 = ENL[nl[0] - 1, 1]
    X2 = ENL[nl[1] - 1, 0]
    Y2 = ENL[nl[1] - 1, 1]

    L = math.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2)

    C = (X2 - X1) / L
    S = (Y2 - Y1) / L

    k = (E * A) / L * np.array([
        [C ** 2, C * S, -C ** 2, -C * S],
        [C * S, S ** 2, -C * S, -S ** 2],
        [-C ** 2, -C * S, C ** 2, C * S],
        [-C * S, -S ** 2, C * S, S ** 2],
    ])

    return k


def assemble_forces(ENL, NL):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)
    DOF = 0

    Fp = []

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == 1:
                DOF += 1
                Fp.append(ENL[i, 5 * PD + j])
    Fp = np.vstack([Fp]).reshape(-1, 1)

    return Fp


def assemble_displacements(ENL, NL):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)
    DOC = 0

    Up = []

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == -1:
                DOC += 1
                Up.append(ENL[i, 4 * PD + j])
    Up = np.vstack([Up]).reshape(-1, 1)

    return Up


def update_nodes(ENL, U_u, NL, Fu):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

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


def uniform_mesh(d1: float, d2: float, p: int, m: int, element_type):
    PD = 2
    q = np.array([[0, 0], [d1, 0], [0, d2], [d1, d2]])  # 4 corners
    NoN = (p + 1) * (m + 1)
    NoE = p * m
    NPE = 4

    # Nodes
    NL = np.zeros([NoN, PD])
    a = (q[1, 0] - q[0, 0]) / p  # Increment in the horizontal direction
    b = (q[2, 1] - q[0, 1]) / m  # Increment in the vertical direction
    n = 0  # This allows to go thorugh rows in node list NL.

    for i in range(1, m + 2):
        for j in range(1, p + 2):
            NL[n, 0] = q[0, 0] + (j - 1) * a  # for x values
            NL[n, 1] = q[0, 1] + (i - 1) * b  # for y values
            n += 1

    # Elements
    EL = np.zeros([NoE, NPE])
    for i in range(1, m + 1):
        for j in range(1, p + 1):
            index = (i - 1) * p + j - 1
            if j == 1:
                EL[index, 0] = (i - 1) * (p + 1) + j
                EL[index, 1] = EL[(i - 1) * p + j - 1, 0] + 1
                EL[index, 3] = EL[(i - 1) * p + j - 1, 0] + (p + 1)
                EL[index, 2] = EL[(i - 1) * p + j - 1, 3] + 1
            else:
                EL[index, 0] = EL[(i - 1) * p + j - 2, 1]
                EL[index, 3] = EL[(i - 1) * p + j - 2, 2]
                EL[index, 1] = EL[(i - 1) * p + j - 1, 0] + 1
                EL[index, 2] = EL[(i - 1) * p + j - 1, 3] + 1

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


def void_mesh(d1, d2, p, m, R, element_type, shape_type):
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

    if shape_type == "circle":
        for i in range(1, p + 2):
            coor11[m * (p + 1) + i - 1, 0] = R * np.cos((5 * math.pi / 4) + (i - 1) * ((math.pi / 2) / p)) + d1 / 2
            coor11[m * (p + 1) + i - 1, 1] = R * np.sin((5 * math.pi / 4) + (i - 1) * ((math.pi / 2) / p)) + d2 / 2

    if shape_type == "square":
        for i in range(1, p + 2):
            coor11[m * (p + 1) + i - 1, 0] = (d1 - R) / 2 + (i - 1) * R / p
            coor11[m * (p + 1) + i - 1, 1] = (d2 - R) / 2

    if shape_type == "rhombus":
        for i in range(1, p + 2):
            coor11[m * (p + 1) + i - 1, 0] = (d1 - R) / 2 + (i - 1) * R / p
            if (d2 - R) / 2 - (i - 1) * R / p < (d2 - R) / 2 - R / 2:
                coor11[m * (p + 1) + i - 1, 1] = (d2 - R) / 2 - R + (i - 1) * R / p
            else:
                coor11[m * (p + 1) + i - 1, 1] = (d2 - R) / 2 - (i - 1) * R / p

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

    if shape_type == "circle":
        for i in range(1, p + 2):
            coor22[m * (p + 1) + i - 1, 0] = R * np.cos((3 * math.pi / 4) - (i - 1) * ((math.pi / 2) / p)) + d1 / 2
            coor22[m * (p + 1) + i - 1, 1] = R * np.sin((3 * math.pi / 4) - (i - 1) * ((math.pi / 2) / p)) + d2 / 2

    if shape_type == "square":
        for i in range(1, p + 2):
            coor22[m * (p + 1) + i - 1, 0] = (d1 - R) / 2 + (i - 1) * R / p
            coor22[m * (p + 1) + i - 1, 1] = (d2 - R) / 2 + R

    if shape_type == "rhombus":
        for i in range(1, p + 2):
            coor22[m * (p + 1) + i - 1, 0] = (d1 - R) / 2 + (i - 1) * R / p
            coor22[m * (p + 1) + i - 1, 1] = (d2 - R) / 2 + R

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

    if shape_type == "circle":
        for i in range(1, p):
            coor33[m * (p - 1) + i - 1, 0] = R * np.cos((5 * math.pi / 4) - i * ((math.pi / 2) / p)) + d1 / 2
            coor33[m * (p - 1) + i - 1, 1] = R * np.sin((5 * math.pi / 4) - i * ((math.pi / 2) / p)) + d2 / 2

    if shape_type == "square":
        for i in range(1, p):
            coor33[m * (p - 1) + i - 1, 0] = (d1 - R) / 2
            coor33[m * (p - 1) + i - 1, 1] = (d2 - R) / 2 + (i - 1) * R / p

    if shape_type == "rhombus":
        for i in range(1, p):
            coor33[m * (p - 1) + i - 1, 0] = (d1 - R) / 2
            coor33[m * (p - 1) + i - 1, 1] = (d2 - R) / 2 + (i - 1) * R / p

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

    if shape_type == "circle":
        for i in range(1, p):
            coor44[m * (p - 1) + i - 1, 0] = R * np.cos((7 * math.pi / 4) + i * ((math.pi / 2) / p)) + d1 / 2
            coor44[m * (p - 1) + i - 1, 1] = R * np.sin((7 * math.pi / 4) + i * ((math.pi / 2) / p)) + d2 / 2

    if shape_type == "square":
        for i in range(1, p):
            coor44[m * (p - 1) + i - 1, 0] = (d1 - R) / 2 + R
            coor44[m * (p - 1) + i - 1, 1] = (d2 - R) / 2 + (i - 1) * R / p

    if shape_type == "rhombus":
        for i in range(1, p):
            coor44[m * (p - 1) + i - 1, 0] = (d1 - R) / 2 + R
            coor44[m * (p - 1) + i - 1, 1] = (d2 - R) / 2 + (i - 1) * R / p

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

    if element_type == "D2TR6N":
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

    if element_type == "D2TR8N" or element_type == "D2TR9N":
        # EL = np.hstack([EL, np.zeros([EL, 4])])
        #
        # for i in range(1, NoE + 1):
        #     data = NL[EL[i, [0, 1, 2, 3]] - 1, :]
        #     avg_1 = (data[0] + data[1]) / 2
        #     avg_2 = (data[1] + data[2]) / 2
        #     avg_3 = (data[2] + data[3]) / 2
        #     avg_4 = (data[3] + data[0]) / 2
        #
        #     try:
        #         index1 = np.where(np.all(NL == avg_1, axis=1))[0][0]
        #     except:
        #         index1 = np.size(NL, 0)
        #
        #     try:
        #         index2 = np.where(np.all(NL == avg_2, axis=1))[0][0]
        #     except:
        #         index2 = np.size(NL, 0)
        #
        #     try:
        #         index3 = np.where(np.all(NL == avg_3, axis=1))[0][0]
        #     except:
        #         index3 = np.size(NL, 0)
        #
        #     try:
        #         index4 = np.where(np.all(NL == avg_4, axis=1))[0][0]
        #     except:
        #         index4 = np.size(NL, 0)
        #
        #     if index1:
        #         EL[i, 5] = index1
        #     else:
        #         EL[i, 5] = index1
        #         NL = np.vstack([NL, avg_1])
        #
        #     if index2:
        #         EL[i, 5] = index2
        #     else:
        #         EL[i, 5] = index2
        #         NL = np.vstack([NL, avg_2])
        #
        #     if index3:
        #         EL[i, 5] = index3
        #     else:
        #         EL[i, 5] = index3
        #         NL = np.vstack([NL, avg_3])
        #
        #     if index4:
        #         EL[i, 5] = index4
        #     else:
        #         EL[i, 5] = index4
        #         NL = np.vstack([NL, avg_4])


        NPE_new = 4
        NoE_new = 2 * NoE
        EL_new = np.zeros([NoE_new, NPE_new])

        for i in range(1, NoE + 1):
            EL_new[2 * (i - 1), 0] = EL[i - 1, 1]
            EL_new[2 * (i - 1), 1] = EL[i - 1, 0]
            EL_new[2 * (i - 1), 2] = EL[i - 1, 3]
            EL_new[2 * (i - 1), 3] = EL[i - 1, 2]

            EL_new[2 * (i - 1) + 1, 0] = EL[i - 1, 1]
            EL_new[2 * (i - 1) + 1, 1] = EL[i - 1, 0]
            EL_new[2 * (i - 1) + 1, 2] = EL[i - 1, 3]
            EL_new[2 * (i - 1) + 1, 3] = EL[i - 1, 2]

        EL = EL_new

        # NPE_new = 3
        # NoE_new = 2 * NoE
        # EL_new = np.zeros([NoE_new, NPE_new])
        #
        # for i in range(1, NoE + 1):
        #     EL_new[2 * (i - 1), 0] = EL[i - 1, 0]
        #     EL_new[2 * (i - 1), 1] = EL[i - 1, 1]
        #     EL_new[2 * (i - 1), 2] = EL[i - 1, 2]
        #
        #     EL_new[2 * (i - 1) + 1, 0] = EL[i - 1, 0]
        #     EL_new[2 * (i - 1) + 1, 1] = EL[i - 1, 2]
        #     EL_new[2 * (i - 1) + 1, 2] = EL[i - 1, 3]

        EL = EL_new

    if element_type == "D2QU4N":
        NPE_new = 4
        NoE_new = 2 * NoE
        EL_new = np.zeros([NoE_new, NPE_new])

        for i in range(1, NoE + 1):
            EL_new[2 * (i - 1), 0] = EL[i - 1, 1]
            EL_new[2 * (i - 1), 1] = EL[i - 1, 0]
            EL_new[2 * (i - 1), 2] = EL[i - 1, 3]
            EL_new[2 * (i - 1), 3] = EL[i - 1, 2]

            EL_new[2 * (i - 1) + 1, 0] = EL[i - 1, 1]
            EL_new[2 * (i - 1) + 1, 1] = EL[i - 1, 0]
            EL_new[2 * (i - 1) + 1, 2] = EL[i - 1, 3]
            EL_new[2 * (i - 1) + 1, 3] = EL[i - 1, 2]

        EL = EL_new




        # EL = np.hstack([EL, np.zeros([EL, 2])])
        #
        # for i in range(1, NoE + 1):
        #     data = NL[EL[i, [0, 1]] - 1, :]
        #     avg_1 = (data[0] + data[1]) / 2
        #
        #     try:
        #         index1 = np.where(np.all(NL == avg_1, axis=1))[0][0]
        #     except:
        #         index1 = np.size(NL, 0)
        #
        #     if index1:
        #         EL[i, 5] = index1
        #     else:
        #         EL[i, 5] = index1
        #         NL = np.vstack([NL, avg_1])

        # NPE_new = 3
        # NoE_new = 2 * NoE
        # EL_new = np.zeros([NoE_new, NPE_new])
        #
        # for i in range(1, NoE + 1):
        #     EL_new[2 * (i - 1), 0] = EL[i - 1, 0]
        #     EL_new[2 * (i - 1), 1] = EL[i - 1, 1]
        #     EL_new[2 * (i - 1), 2] = EL[i - 1, 2]
        #
        #     EL_new[2 * (i - 1) + 1, 0] = EL[i - 1, 0]
        #     EL_new[2 * (i - 1) + 1, 1] = EL[i - 1, 2]
        #     EL_new[2 * (i - 1) + 1, 2] = EL[i - 1, 3]

        # EL = EL_new

    EL = EL.astype(int)

    return NL, EL


def stiffness(x, GPE):
    NPE = np.size(x, 0)
    PD = np.size(x, 1)

    K = np.zeros([NPE * PD, NPE * PD])

    coor = x.T

    for i in range(1, NPE + 1):
        for j in range(1, NPE + 1):

            k = np.zeros([PD, PD])

            for gp in range(1, GPE + 1):

                J = np.zeros([PD, PD])

                grad = np.zeros([PD, NPE])

                (xi, eta, alpha) = GaussPoint(NPE, GPE, gp)

                grad_nat = grad_N_nat(NPE, xi, eta)

                J = coor @ grad_nat.T  # np.matmul(.)

                grad = np.linalg.inv(J).T @ grad_nat

                for a in range(1, PD + 1):
                    for c in range(1, PD + 1):
                        for b in range(1, PD + 1):
                            for d in range(1, PD + 1):

                                if NPE == 4:  # 4 kenarlı ise
                                    k[a - 1, c - 1] = k[a - 1, c - 1] + grad[b - 1, i - 1] * constitutive(a, b, c, d) * \
                                                      grad[d - 1, j - 1] * np.linalg.det(J) * alpha

                                if NPE == 3:  # 3gen ise
                                    k[a - 1, c - 1] = k[a - 1, c - 1] + grad[b - 1, i - 1] * constitutive(a, b, c, d) * \
                                                      grad[d - 1, j - 1] * np.linalg.det(J) * alpha * 1 / 2

                K[((i - 1) * PD + 1) - 1:i * PD, ((j - 1) * PD + 1) - 1:j * PD] = k

    return K


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

    if NPE == 8 or NPE == 9:
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
        result[0, 0] = 1 / 4 * (1 - eta) * (2 * xi + eta)
        result[0, 1] = 1 / 4 * (1 - eta) * (2 * xi - eta)
        result[0, 2] = 1 / 4 * (1 + eta) * (2 * xi + eta)
        result[0, 3] = 1 / 4 * (1 + eta) * (2 * xi - eta)
        result[0, 4] = -xi * (1 - eta)
        result[0, 5] = 1 / 2 * (1 - eta) * (1 + eta)
        result[0, 6] = -xi * (1 + eta)
        result[0, 7] = -1 / 2 * (1 - eta) * (1 + eta)

        result[1, 0] = 1 / 4 * (1 - xi) * (xi + 2 * eta)
        result[1, 1] = 1 / 4 * (1 + xi) * (-xi + 2 * eta)
        result[1, 2] = 1 / 4 * (1 + xi) * (xi + 2 * eta)
        result[1, 3] = 1 / 4 * (1 - xi) * (-xi + 2 * eta)
        result[1, 4] = -1 / 2 * (1 - eta) * (1 + eta)
        result[1, 5] = -1 * (1 + xi) * eta
        result[1, 6] = 1 / 2 * (1 - xi) * (1 + xi)
        result[1, 7] = -1 * (1 - xi) * eta

    if NPE == 9:
        result[0, 0] = 1 / 4 * (1 - 2 * xi) * (1 - eta) * eta
        result[0, 1] = -1 / 4 * (1 + 2 * xi) * (1 - eta) * eta
        result[0, 2] = 1 / 4 * (1 + 2 * xi) * (1 + eta) * eta
        result[0, 3] = -1 / 4 * (1 - 2 * xi) * (1 + eta) * eta
        result[0, 4] = xi * eta * (1 - eta)
        result[0, 5] = 1 / 2 * (1 + 2 * xi) * (1 - eta) * (1 + eta)
        result[0, 6] = -xi * eta * (1 + eta)
        result[0, 7] = -1 / 2 * (1 - 2 * xi) * (1 - eta) * (1 + eta)
        result[0, 8] = -2 * xi * (1 - eta) * (1 + eta)

        result[1, 0] = 1 / 4 * (1 - xi)
        xi(1 - 2 * eta)
        result[1, 1] = -1 / 4 * (1 + xi)
        xi(1 - 2 * eta)
        result[1, 2] = 1 / 4 * (1 + xi)
        xi(1 + 2 * eta)
        result[1, 3] = -1 / 4 * (1 - xi)
        xi(1 + 2 * eta)
        result[1, 4] = 1 / 2 * (1 - xi)(1 + xi)(2 * eta - 1)
        result[1, 5] = -1 * (1 + xi) * xi * eta
        result[1, 6] = 1 / 2 * (1 - xi) * (1 + xi) * (1 + 2 * eta)
        result[1, 7] = (1 - xi) * xi * eta
        result[1, 8] = -2 * (1 - xi) * (1 + xi) * eta

    if NPE == 6:
        result[0, 0] = -1 + 4 * xi
        result[0, 1] = 0
        result[0, 2] = -3 + 4 * xi + 4 * eta
        result[0, 3] = 4 * eta
        result[0, 4] = -4 * eta
        result[0, 5] = -4 * (-1 + eta + 2 * xi)

        result[1, 0] = 0
        result[1, 1] = -1 + 4 * eta
        result[1, 2] = -3 + 4 * xi + 4 * eta
        result[1, 3] = 4 * xi
        result[1, 4] = -4 * (-1 + 2 * eta + xi)
        result[1, 5] = -4 * xi

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


def post_process(NL, EL, ENL, scale):
    PD = np.size(NL, 1)
    NoE = np.size(EL, 0)
    NPE = np.size(EL, 1)

    disp, stress, strain = element_post_process(NL, EL, ENL)

    if NPE in [3, 4]:  # D2QU4N and D2TR3N #add others such as 8 or 9, look at line 59 and 75
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

    if NPE in [6]:
        NPE = NPE / 2
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

        # Calculate X and Y coordinates of the nodes
        X = ENL[EL[:, [0, 1, 2]] - 1, 0] + scale * ENL[EL[:, [0, 1, 2]] - 1, 4 * PD]
        Y = ENL[EL[:, [0, 1, 2]] - 1, 1] + scale * ENL[EL[:, [0, 1, 2]] - 1, 4 * PD + 1]

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

    if NPE in [8, 9]:
        NPE = 4
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

        # Calculate X and Y coordinates of the nodes
        X = ENL[EL[:, [0, 1, 2, 3]] - 1, 0] + scale * ENL[EL[:, [0, 1, 2, 3]] - 1, 4 * PD]
        Y = ENL[EL[:, [0, 1, 2, 3]] - 1, 1] + scale * ENL[EL[:, [0, 1, 2, 3]] - 1, 4 * PD + 1]

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
    if NPE == 8 or NPE == 9:
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
                stress[e - 1, gp - 1, a - 1, b - 1] = sigma[a - 1, b - 1]

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


def assemble_displacements1(ENL, NL):
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


def assemble_forces1(ENL, NL):
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


def assemble_stiffness1(ENL, EL, NL):
    NoE = np.size(EL, 0)
    NPE = np.size(EL, 1)
    NoN = np.size(NL, 0)
    PD = np.size(NL, 1)

    # GPE nerde varsa
    # eger NPE 6 ise TR6N ise GPE 3
    # eger NPE 8 ise QU8N ise GPE 9
    # eger NPE 9 ise QU9N ise GPE 9

    K = np.zeros([NoN * PD, NoN * PD])

    for i in range(1, NoE + 1):
        n1 = EL[i - 1, 0:NPE]
        k = element_stiffness1(n1, NL)
        for r in range(0, NPE):
            for p in range(0, PD):
                for q in range(0, NPE):
                    for s in range(0, PD):
                        row = ENL[n1[r] - 1, p + 3 * PD]
                        column = ENL[n1[q] - 1, s + 3 * PD]
                        value = k[r * PD + p, q * PD + s]
                        K[int(row) - 1, int(column) - 1] = K[int(row) - 1, int(column) - 1] + value
    return K


def element_stiffness1(n1, NL):
    NPE = np.size(n1, 0)
    PD = np.size(NL, 1)

    x = np.zeros([NPE, PD])
    x[0:NPE, 0:PD] = NL[n1[0:NPE] - 1, 0:PD]

    K = np.zeros([NPE * PD, NPE * PD])

    coor = x.T

    GPE = 4
    if NPE == 3:
        GPE = 1

    if NPE == 4:
        GPE = 4

    for i in range(1, NPE + 1):

        for j in range(1, NPE + 1):

            k = np.zeros([PD, PD])

            for gp in range(1, GPE + 1):

                J = np.zeros([PD, PD])

                grad = np.zeros([PD, NPE])

                (xi, eta, alpha) = GaussPoint(NPE, GPE, gp)

                grad_nat = grad_N_nat(NPE, xi, eta)

                J = coor @ grad_nat.T  # np.matmul(.)

                grad = np.linalg.inv(J).T @ grad_nat

                for a in range(1, PD + 1):
                    for c in range(1, PD + 1):
                        for b in range(1, PD + 1):
                            for d in range(1, PD + 1):
                                # Implement 2 if blocks one for triangular elements and another for quadrangular elements.

                                if NPE == 4:  # 4 kenarlı ise
                                    k[a - 1, c - 1] = k[a - 1, c - 1] + grad[b - 1, i - 1] * constitutive(a, b, c, d) * \
                                                      grad[d - 1, j - 1] * np.linalg.det(J) * alpha

                                if NPE == 3:  # 3gen ise
                                    k[a - 1, c - 1] = k[a - 1, c - 1] + grad[b - 1, i - 1] * constitutive(a, b, c, d) * \
                                                      grad[d - 1, j - 1] * np.linalg.det(J) * alpha * 1 / 2

            K[((i - 1) * PD + 1) - 1:i * PD, ((j - 1) * PD + 1) - 1:j * PD] = k

    return K
