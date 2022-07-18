import random
import tkinter as tk
from tkinter.ttk import *
from tkinter import messagebox
from tkinter.filedialog import askopenfile, asksaveasfile
import numpy as np
from functions import *
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from fem_types import Node, Element
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import copy

# Config
DEBUG = True

# Values
NL = np.array([])
EL = np.array([])
DorN = np.array([])
Fu = np.array([])
U_u = np.array([])
ENL = np.array([])  # X Y XBC YBC TEMP_X TEMP_Y GLOBAL_X GLOBAL_Y DISP_X DISP_Y FORCE_X FORCE_Y
E, A = 210_000_000_000, 0.01  # 210 GPA
SCALE = 0.01
node_labels_active = True
element_labels_active = True
node_list = []
element_list = []
iter_list = []


def generate_node():
    global node_list
    x_value = entry_node_x.get()
    y_value = entry_node_y.get()

    try:
        x_value = float(x_value)
        y_value = float(y_value)
    except ValueError:
        messagebox.showwarning("Node Generation Error", "Enter a float")
        return

    match = [n for n in node_list if n.x == x_value and n.y == y_value]

    if match:
        messagebox.showwarning("Node Generation Error", "There is not a node here!")
        return

    index = len(node_list) + 1
    node_list.append(Node(index=index, x=x_value, y=y_value))
    label_node_number1.configure(text=str(len(node_list)))

    entry_node_x.delete(0, tk.END)
    entry_node_y.delete(0, tk.END)
    entry_node_x.focus_set()

    draw_node_element()


def generate_element():
    global element_list
    l_value = entry_element_x.get()
    r_value = entry_element_y.get()

    try:
        l_value = int(l_value)
        r_value = int(r_value)
    except ValueError:
        messagebox.showwarning("Element Generation Error", "Enter a integer")
        return

    if l_value == r_value:
        messagebox.showwarning("Element Generation Error", "Select a valid node!")
        return

    left_node = [n for n in node_list if n.index == l_value]
    right_node = [n for n in node_list if n.index == r_value]

    if not left_node or not right_node:
        messagebox.showwarning("Element Generation Error", "Select a valid node!")
        return

    match = [e for e in element_list if
             (e.l_node == l_value and e.r_node == r_value) or (e.l_node == r_value and e.r_node == l_value)]

    if match:
        messagebox.showwarning("Element Generation Error", "There is a element here!")
        return

    index = len(element_list) + 1
    element_list.append(Element(index=index, l_node=l_value, r_node=r_value))
    label_element_number1.configure(text=str(len(element_list)))

    entry_element_x.delete(0, tk.END)
    entry_element_y.delete(0, tk.END)
    entry_element_x.focus_set()

    draw_node_element()


def generate_bc():
    global node_list
    index = entry_bc.get()
    direction = var_direction.get()
    bc_type = var_type.get()
    magnitude = entry_magnitude.get()

    try:
        index = int(index)
    except ValueError:
        messagebox.showwarning("BC Generation Error", "Enter a node")
        return

    try:
        magnitude = float(magnitude)
    except ValueError:
        messagebox.showwarning("BC Generation Error", "Enter a float magnitude")
        return

    if not node_list:
        messagebox.showwarning("BC Generation Error", "Enter at least 1 node to add a BC!")
        return

    match = [n for n in node_list if n.index == index]

    if not match:
        messagebox.showwarning("BC Generation Error", "There is a not node here!")
        return

    node: Node = [n for n in node_list if n.index == index][0]

    if direction == "X":
        if bc_type == "Force":
            node.bc_f_x = magnitude
            node.bc_d_x = None
        else:
            node.bc_f_x = None
            node.bc_d_x = magnitude
    else:
        if bc_type == "Force":
            node.bc_f_y = magnitude
            node.bc_d_y = None
        else:
            node.bc_f_y = None
            node.bc_d_y = magnitude

    entry_bc.delete(0, tk.END)
    entry_magnitude.delete(0, tk.END)
    entry_bc.focus_set()

    draw_node_element()


def clear():
    global node_list, element_list
    node_list = []
    element_list = []
    draw_node_element()


def clear_bc():
    global node_list
    for node in node_list:
        node.bc_f_x = None
        node.bc_d_x = None
        node.bc_f_y = None
        node.bc_d_y = None
    draw_node_element()


def iterate_process():
    global node_list, element_list, iter_list
    global U_u, Fu, NL, EL, DorN, ENL, E, A

    E = entry_modulus.get()
    A = entry_area.get()

    try:
        E = float(E)
        A = float(A)
    except ValueError:
        messagebox.showwarning("Run Process Error", "Enter valid material properties!")
        return

    if len(node_list) < 2 or len(element_list) < 1:
        messagebox.showwarning("Process Error", "There needs to be at least 2 nodes and 1 element!")
        return

    iter_list = []
    iter_var.set([i[0] for i in iter_list])
    run_process()

    # for i in range(2):

    data = ENL[:, [0, 1, 2, 3, 8, 9]]
    for k, node in enumerate(data):
        x, y, bcx, bcy, dispx, dispy = node

        # if k == 5:
        #     break

        print("Node:", k)

        compare_list = []
        for j in range(4):
            if bcx == -1 and bcy == -1:
                # print("Fixed in both dir")
                break

            if bcx != -1:
                if j == 0:
                    node_list[k].x += 0.25
                elif j == 1:
                    node_list[k].x -= 0.25
                elif j == 2:
                    node_list[k].x -= 0.25
                elif j == 3:
                    node_list[k].x += 0.25

            if bcy != -1:
                if j == 0:
                    node_list[k].y += 0.25
                elif j == 1:
                    node_list[k].y += 0.25
                elif j == 2:
                    node_list[k].y -= 0.25
                elif j == 3:
                    node_list[k].y -= 0.25

            # run_process()
            run_process_without_side_effects()
            x_scatter_f, y_scatter_f, stress_list, forcex_array = calculate_scatter_force()

            # Max
            max_stress = 0
            for f in forcex_array:
                if abs(f[0]) > abs(max_stress):
                    max_stress = f[0]

            # print(forcex_array)
            # print(len(forcex_array))
            # print(max_stress)
            # print(f"Item {j}", max_stress)

            compare_list.append(max_stress)

            # print(ENL[:, [0, 1, 2, 3, 8, 9]][k])

        if bcx == -1 and bcy == -1:
            # run_process()
            continue

        # max_index, max_value = min(enumerate([abs(x) for x in compare_list]), key=operator.itemgetter(1))

        max_index = -1
        max_value = 0
        for i, z in enumerate(compare_list):
            if abs(z) > max_value:
                max_index = i
                max_value = abs(z)

        if max_index != -1:
            if bcx != -1:
                if max_index == 0:
                    node_list[k].x += 0.25
                elif max_index == 1:
                    node_list[k].x -= 0.25
                elif max_index == 2:
                    node_list[k].x -= 0.25
                elif max_index == 3:
                    node_list[k].x += 0.25

            if bcy != -1:
                if max_index == 0:
                    node_list[k].y += 0.25
                elif max_index == 1:
                    node_list[k].y += 0.25
                elif max_index == 2:
                    node_list[k].y -= 0.25
                elif max_index == 3:
                    node_list[k].y -= 0.25

        # print(compare_list)
        # print("Max:", max_index)
        # print()

        run_process()


        # data = ENL[:, [0, 1, 2, 3, 8, 9]]  # Current
        # max_node = None  # Current max displacement node
        # direction = "x"
        # max_disp = 0
        #
        # print("iter:", i)
        #
        # # Select which node to move
        # for node in data:
        #     x, y, bcx, bcy, dispx, dispy = node
        #     # print("Selected:", dispx, dispy)
        #
        #     if bcx != -1 and abs(dispx) > max_disp:
        #         direction = "x"
        #         max_disp = abs(dispx)
        #         max_node = node
        #
        #     if bcy != -1 and abs(dispy) > max_disp:
        #         direction = "y"
        #         max_disp = abs(dispy)
        #         max_node = node
        #
        # sign = -1

        # Compare with previous iteration
        # if i > 0:  # Index 0 problem
        #     previous_data = iter_list[-2][1][:, [0, 1, 2, 3, 8, 9]]
        #     for index, prev_node in enumerate(previous_data):
        #         if prev_node[0] == max_node[0] and prev_node[1] == max_node[1]:
        #             prev_x, prev_y, prev_bcx, prev_bcy, prev_dispx, prev_dispy = prev_node
        #             # print("Max:", max_node[4], max_node[5])
        #             # print("Prev:", prev_dispx, prev_dispy)
        #
        #             if direction == "x":
        #                 sign = 1 if max_node[4] > prev_node[4] else -1
        #             else:
        #                 sign = 1 if max_node[5] > prev_node[5] else -1
        #             break

        # print()

        # Move node
        # for n in node_list:
        #     if n.x == max_node[0] and n.y == max_node[1]:
        #         if direction == "x":
        #             n.x = n.x + sign * 0.1
        #         else:
        #             n.y = n.y + sign * 0.1
        #         break

        # run_process()


def run_process():
    global U_u, Fu, NL, EL, DorN, ENL, E, A, node_list, element_list

    E = entry_modulus.get()
    A = entry_area.get()

    try:
        E = float(E)
        A = float(A)
    except ValueError:
        messagebox.showwarning("Run Process Error", "Enter valid material properties!")
        return

    tmp_fu_list = []
    tmp_uu_list = []
    tmp_dorn_list = []

    if len(node_list) < 2 or len(element_list) < 1:
        messagebox.showwarning("Process Error", "There needs to be at least 2 nodes and 1 element!")
        return

    for node in node_list:
        x_f = 0 if node.bc_f_x is None else node.bc_f_x
        y_f = 0 if node.bc_f_y is None else node.bc_f_y
        tmp_fu_list.append([x_f, y_f])

        x_d = 0 if node.bc_d_x is None else node.bc_d_x
        y_d = 0 if node.bc_d_y is None else node.bc_d_y
        tmp_uu_list.append([x_d, y_d])

        x_dn_type = 1 if node.bc_d_x is None else -1
        y_dn_type = 1 if node.bc_d_y is None else -1
        tmp_dorn_list.append([x_dn_type, y_dn_type])

    NL = np.array([[i.x, i.y] for i in node_list])
    EL = np.array([[i.l_node, i.r_node] for i in element_list])
    DorN = np.array([i for i in tmp_dorn_list])
    Fu = np.array([i for i in tmp_fu_list])
    U_u = np.array([i for i in tmp_uu_list])

    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    ENL = np.zeros([NoN, 6 * PD])

    ENL[:, 0:PD] = NL[:, :]
    ENL[:, PD:2 * PD] = DorN[:, :]

    ENL, DOFs, DOCs = assign_bcs(NL, ENL)

    K = assemble_stiffness(ENL, EL, NL, E, A)

    ENL[:, 4 * PD: 5 * PD] = U_u[:, :]
    ENL[:, 5 * PD: 6 * PD] = Fu[:, :]

    # U_u = U_u.flatten()
    # Fu = Fu.flatten()

    Fp = assemble_forces(ENL, NL)
    Up = assemble_displacements(ENL, NL)

    K_UU = K[0: DOFs, 0: DOFs]
    K_UP = K[0: DOFs, DOFs: DOFs + DOCs]
    K_PU = K[DOFs: DOFs + DOCs, 0: DOFs]
    K_PP = K[DOFs: DOFs + DOCs, DOFs: DOFs + DOCs]

    F = Fp - np.matmul(K_UP, Up)
    U_u = np.matmul(np.linalg.inv(K_UU), F)
    Fu = np.matmul(K_PU, U_u) + np.matmul(K_PP, Up)

    ENL = update_nodes(ENL, U_u, NL, Fu)

    iter_list.append(
        (f"Iter {len(iter_list)}", copy.deepcopy(ENL), copy.deepcopy(NL), copy.deepcopy(EL), copy.deepcopy(node_list),
         copy.deepcopy(element_list)))
    iter_var.set([i[0] for i in iter_list])

    run_post_process()


def run_process_without_side_effects():
    global U_u, Fu, NL, EL, DorN, ENL, E, A, node_list, element_list

    E = entry_modulus.get()
    A = entry_area.get()

    try:
        E = float(E)
        A = float(A)
    except ValueError:
        messagebox.showwarning("Run Process Error", "Enter valid material properties!")
        return

    tmp_fu_list = []
    tmp_uu_list = []
    tmp_dorn_list = []

    if len(node_list) < 2 or len(element_list) < 1:
        messagebox.showwarning("Process Error", "There needs to be at least 2 nodes and 1 element!")
        return

    for node in node_list:
        x_f = 0 if node.bc_f_x is None else node.bc_f_x
        y_f = 0 if node.bc_f_y is None else node.bc_f_y
        tmp_fu_list.append([x_f, y_f])

        x_d = 0 if node.bc_d_x is None else node.bc_d_x
        y_d = 0 if node.bc_d_y is None else node.bc_d_y
        tmp_uu_list.append([x_d, y_d])

        x_dn_type = 1 if node.bc_d_x is None else -1
        y_dn_type = 1 if node.bc_d_y is None else -1
        tmp_dorn_list.append([x_dn_type, y_dn_type])

    NL = np.array([[i.x, i.y] for i in node_list])
    EL = np.array([[i.l_node, i.r_node] for i in element_list])
    DorN = np.array([i for i in tmp_dorn_list])
    Fu = np.array([i for i in tmp_fu_list])
    U_u = np.array([i for i in tmp_uu_list])

    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    ENL = np.zeros([NoN, 6 * PD])

    ENL[:, 0:PD] = NL[:, :]
    ENL[:, PD:2 * PD] = DorN[:, :]

    ENL, DOFs, DOCs = assign_bcs(NL, ENL)

    K = assemble_stiffness(ENL, EL, NL, E, A)

    ENL[:, 4 * PD: 5 * PD] = U_u[:, :]
    ENL[:, 5 * PD: 6 * PD] = Fu[:, :]

    # U_u = U_u.flatten()
    # Fu = Fu.flatten()

    Fp = assemble_forces(ENL, NL)
    Up = assemble_displacements(ENL, NL)

    K_UU = K[0: DOFs, 0: DOFs]
    K_UP = K[0: DOFs, DOFs: DOFs + DOCs]
    K_PU = K[DOFs: DOFs + DOCs, 0: DOFs]
    K_PP = K[DOFs: DOFs + DOCs, DOFs: DOFs + DOCs]

    F = Fp - np.matmul(K_UP, Up)
    U_u = np.matmul(np.linalg.inv(K_UU), F)
    Fu = np.matmul(K_PU, U_u) + np.matmul(K_PP, Up)

    ENL = update_nodes(ENL, U_u, NL, Fu)

    return ENL, NL, EL, node_list, element_list


def run_post_process():
    x_scatter_d, y_scatter_d, color_x_d, dispx_array = calculate_scatter_displacement()
    x_scatter_f, y_scatter_f, stress_list, forcex_array = calculate_scatter_force()
    draw_post_process_displacement(x_scatter_d, y_scatter_d, color_x_d, dispx_array)
    draw_post_process_force(x_scatter_f, y_scatter_f, stress_list, forcex_array)


def calculate_scatter_displacement():
    global ENL, EL, SCALE
    coor = []
    dispx_array = []

    if not ENL.any():
        return

    for i in range(np.size(NL, 0)):
        dispx = ENL[i, 8]
        dispy = ENL[i, 9]

        x = ENL[i, 0] + dispx * SCALE
        y = ENL[i, 1] + dispy * SCALE

        dispx_array.append(dispx)
        coor.append(np.array([x, y]))

    coor = np.vstack(coor)
    dispx_array = np.vstack(dispx_array)

    x_scatter = []
    y_scatter = []
    color_x = []

    for i in range(0, np.size(EL, 0)):
        x1 = coor[EL[i, 0] - 1, 0]
        x2 = coor[EL[i, 1] - 1, 0]
        y1 = coor[EL[i, 0] - 1, 1]
        y2 = coor[EL[i, 1] - 1, 1]

        dispx_el = np.array([
            dispx_array[EL[i, 0] - 1],
            dispx_array[EL[i, 1] - 1],
        ])

        if x1 == x2:
            x = np.linspace(x1, x2, 200)
            y = np.linspace(y1, y2, 200)
        else:
            m = (y2 - y1) / (x2 - x1)
            x = np.linspace(x1, x2, 200)
            y = m * (x - x1) + y1

        x_scatter.append(x)
        y_scatter.append(y)

        color_x.append(np.linspace(np.abs(dispx_el[0]), np.abs(dispx_el[1]), 200))

    x_scatter = np.vstack([x_scatter]).flatten()
    y_scatter = np.vstack([y_scatter]).flatten()
    color_x = np.vstack([color_x]).flatten()
    return x_scatter, y_scatter, color_x, dispx_array


def calculate_scatter_force():
    global ENL, EL, SCALE
    coor = []
    forcex_array = []

    if not ENL.any():
        return

    for i in range(np.size(NL, 0)):
        forcex = ENL[i, 8]
        forcey = ENL[i, 9]

        x = ENL[i, 0] + forcex * SCALE
        y = ENL[i, 1] + forcey * SCALE

        forcex_array.append(forcex)
        coor.append(np.array([x, y]))

    coor = np.vstack(coor)
    forcex_array = np.vstack(forcex_array)

    x_scatter = []
    y_scatter = []
    color_x = []
    stress_list = []

    for i in range(0, np.size(EL, 0)):
        x1 = coor[EL[i, 0] - 1, 0]
        x2 = coor[EL[i, 1] - 1, 0]
        y1 = coor[EL[i, 0] - 1, 1]
        y2 = coor[EL[i, 1] - 1, 1]

        dlx = ENL[EL[i, 0] - 1, 8] - ENL[EL[i, 1] - 1, 8]
        dly = ENL[EL[i, 0] - 1, 9] - ENL[EL[i, 1] - 1, 9]
        dl = math.sqrt(dlx ** 2 + dly ** 2)

        lx = ENL[EL[i, 0] - 1, 0] - ENL[EL[i, 1] - 1, 0]
        ly = ENL[EL[i, 0] - 1, 1] - ENL[EL[i, 1] - 1, 1]
        L = math.sqrt(lx ** 2 + ly ** 2)

        stress = E * dl / L
        stress_list.append(np.linspace(stress, stress, 200))

        dispx_el = np.array([
            forcex_array[EL[i, 0] - 1],
            forcex_array[EL[i, 1] - 1],
        ])

        if x1 == x2:
            x = np.linspace(x1, x2, 200)
            y = np.linspace(y1, y2, 200)
        else:
            m = (y2 - y1) / (x2 - x1)
            x = np.linspace(x1, x2, 200)
            y = m * (x - x1) + y1

        x_scatter.append(x)
        y_scatter.append(y)

        color_x.append(np.linspace(np.abs(dispx_el[0]), np.abs(dispx_el[1]), 200))

    x_scatter = np.vstack([x_scatter]).flatten()
    y_scatter = np.vstack([y_scatter]).flatten()
    # color_x = np.vstack([color_x]).flatten()
    stress_list = np.vstack([stress_list]).flatten()
    return x_scatter, y_scatter, stress_list, forcex_array


def draw_node_element():
    global node_labels_active, element_labels_active, node_list, element_list

    for child in frame_preprocess_right.winfo_children():
        child.destroy()

    label_node_number1.configure(text=str(len(node_list)))
    label_element_number1.configure(text=str(len(element_list)))

    node_figure = Figure(figsize=(6, 5), dpi=100)
    ax_dispx = node_figure.add_subplot(111)

    ax_dispx.axes.set_aspect("equal")
    ax_dispx.axes.set_adjustable("datalim")

    # ax_dispx = node_figure.add_subplot(111, aspect="equal")
    canvas1 = FigureCanvasTkAgg(node_figure, master=frame_preprocess_right)
    toolbar2 = NavigationToolbar2Tk(canvas1, frame_preprocess_right)
    toolbar2.update()
    toolbar2.pack()
    canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    for node in node_list:
        # Node markers
        if node_labels_active:
            ax_dispx.plot(node.x, node.y, "o", markersize=14, markeredgecolor="k", markerfacecolor="r", zorder=10)
            ax_dispx.annotate(node.index, xy=(node.x, node.y), size=8, ha="center", va="center", zorder=15)

        # BCs
        if node.bc_f_x and node.bc_f_x != 0:
            if node.bc_f_x > 0:
                ax_dispx.arrow(node.x - 0.5, node.y, 0.3, 0, head_width=0.05, head_length=0.1, fc='k', ec='k',
                               zorder=15)
                ax_dispx.annotate(f"{abs(node.bc_f_x)}N", xy=(node.x - 0.5, node.y), xycoords="data",
                                  va="center", ha="center", zorder=15,
                                  bbox=dict(boxstyle="round", fc="w"))
            else:
                ax_dispx.arrow(node.x + 0.5, node.y, -0.3, 0, head_width=0.05, head_length=0.1, fc='k', ec='k',
                               zorder=15)
                ax_dispx.annotate(f"{abs(node.bc_f_x)}N", xy=(node.x + 0.5, node.y), xycoords="data",
                                  va="center", ha="center", zorder=15,
                                  bbox=dict(boxstyle="round", fc="w"))

        if node.bc_f_y and node.bc_f_y != 0:
            if node.bc_f_y > 0:
                ax_dispx.arrow(node.x, node.y - 0.5, 0, 0.3, head_width=0.05, head_length=0.1, fc='k', ec='k',
                               zorder=15)
                ax_dispx.annotate(f"{abs(node.bc_f_y)}N", xy=(node.x, node.y - 0.5), xycoords="data",
                                  va="center", ha="center", zorder=15,
                                  bbox=dict(boxstyle="round", fc="w"))
            else:
                ax_dispx.arrow(node.x, node.y + 0.5, 0, -0.3, head_width=0.05, head_length=0.1, fc='k', ec='k',
                               zorder=15)
                ax_dispx.annotate(f"{abs(node.bc_f_y)}N", xy=(node.x, node.y + 0.5), xycoords="data",
                                  va="center", ha="center", zorder=15,
                                  bbox=dict(boxstyle="round", fc="w"))
        if node.bc_d_x == 0:
            x = np.array([[node.x - 0.1, node.y], [node.x - 0.3, node.y - 0.1], [node.x - 0.3, node.y + 0.1]])
            t1 = plt.Polygon(x, color="blue")
            node_figure.gca().add_patch(t1)

        if node.bc_d_y == 0:
            x = np.array([[node.x, node.y - 0.1], [node.x + 0.1, node.y - 0.3], [node.x - 0.1, node.y - 0.3]])
            t1 = plt.Polygon(x, color="blue")
            node_figure.gca().add_patch(t1)

    # Element lines
    for element in element_list:
        # Element markers
        left_node = [n for n in node_list if n.index == element.l_node][0]
        right_node = [n for n in node_list if n.index == element.r_node][0]

        el_x = (left_node.x + right_node.x) / 2
        el_y = (left_node.y + right_node.y) / 2
        if element_labels_active:
            ax_dispx.plot(el_x, el_y, "s", markersize=14, markeredgecolor="k", markerfacecolor="y", zorder=10)
            ax_dispx.annotate(element.index, xy=(el_x, el_y), size=8, ha="center", va="center", zorder=15)

        ax_dispx.plot((left_node.x, right_node.x), (left_node.y, right_node.y), linewidth=8, zorder=8, color="gray")

    canvas1.draw()


def draw_post_process_displacement(x_scatter, y_scatter, color_x, dispx_array):
    global node_list, element_list

    for child in frame_postprocess_left.winfo_children():
        child.destroy()

    node_figure = Figure(figsize=(5, 5), dpi=100)
    ax_dispx = node_figure.add_subplot(111)
    ax_dispx.set_title("Displacement")

    cmap = plt.get_cmap("jet")
    ax_dispx.scatter(x_scatter, y_scatter, c=color_x, cmap=cmap, s=10, edgecolor="none", zorder=10)
    ax_dispx.axes.set_aspect("equal")
    ax_dispx.axes.set_adjustable("datalim")
    norm_x = Normalize(np.abs(dispx_array.min()), np.abs(dispx_array.max()))
    node_figure.colorbar(ScalarMappable(norm=norm_x, cmap=cmap))

    for element in element_list:
        left_node = [n for n in node_list if n.index == element.l_node][0]
        right_node = [n for n in node_list if n.index == element.r_node][0]
        ax_dispx.plot((left_node.x, right_node.x), (left_node.y, right_node.y), linewidth=4, zorder=8, color="gray")

    canvas1 = FigureCanvasTkAgg(node_figure, master=frame_postprocess_left)
    toolbar1 = NavigationToolbar2Tk(canvas1, frame_postprocess_left)
    toolbar1.update()
    toolbar1.pack()
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.TOP)
    # canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def draw_post_process_force(x_scatter, y_scatter, stress_list, dispx_array):
    global node_list, element_list

    for child in frame_postprocess_middle.winfo_children():
        child.destroy()

    node_figure = Figure(figsize=(5, 5), dpi=100)
    ax_force = node_figure.add_subplot(111)
    ax_force.axes.set_aspect("equal")
    ax_force.axes.set_adjustable("datalim")
    ax_force.set_title("Force")

    cmap = plt.get_cmap("jet")
    ax_force.scatter(x_scatter, y_scatter, c=stress_list, cmap=cmap, s=10, edgecolor="none", zorder=10)
    norm_x = Normalize(np.abs(dispx_array.min()), np.abs(dispx_array.max()))
    node_figure.colorbar(ScalarMappable(norm=norm_x, cmap=cmap))

    for element in element_list:
        left_node = [n for n in node_list if n.index == element.l_node][0]
        right_node = [n for n in node_list if n.index == element.r_node][0]
        ax_force.plot((left_node.x, right_node.x), (left_node.y, right_node.y), linewidth=4, zorder=8, color="gray")

    canvas2 = FigureCanvasTkAgg(node_figure, master=frame_postprocess_middle)
    toolbar2 = NavigationToolbar2Tk(canvas2, frame_postprocess_middle)
    toolbar2.update()
    toolbar2.pack()
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.TOP)


def load_preset1():
    global node_list, element_list
    clear()
    node_list = [
        Node(index=1, x=0.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=0, bc_d_y=0),
        Node(index=2, x=1.0, y=0.0, bc_f_x=0, bc_f_y=None, bc_d_x=None, bc_d_y=0),
        Node(index=3, x=0.5, y=1.0, bc_f_x=0, bc_f_y=-20, bc_d_x=None, bc_d_y=None),
    ]
    element_list = [
        Element(index=1, l_node=1, r_node=2),
        Element(index=2, l_node=2, r_node=3),
        Element(index=3, l_node=3, r_node=1),
    ]
    draw_node_element()


def load_preset2():
    global node_list, element_list
    clear()
    node_list = [
        Node(index=1, x=0.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=0.0, bc_d_y=0.0),
        Node(index=2, x=1.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=3, x=2.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=4, x=3.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=5, x=4.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=6, x=5.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=7, x=6.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=8, x=6.0, y=1.0, bc_f_x=None, bc_f_y=-50.0, bc_d_x=None, bc_d_y=None),
        Node(index=9, x=5.0, y=1.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=10, x=4.0, y=1.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=11, x=3.0, y=1.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=12, x=2.0, y=1.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=13, x=1.0, y=1.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=14, x=0.0, y=1.0, bc_f_x=None, bc_f_y=None, bc_d_x=0.0, bc_d_y=None)
    ]
    element_list = [
        Element(index=1, l_node=1, r_node=2),
        Element(index=2, l_node=2, r_node=3),
        Element(index=3, l_node=3, r_node=4),
        Element(index=4, l_node=4, r_node=5),
        Element(index=5, l_node=5, r_node=6),
        Element(index=6, l_node=6, r_node=7),
        Element(index=7, l_node=7, r_node=8),
        Element(index=8, l_node=8, r_node=9),
        Element(index=9, l_node=9, r_node=10),
        Element(index=10, l_node=10, r_node=11),
        Element(index=11, l_node=11, r_node=12),
        Element(index=12, l_node=12, r_node=13),
        Element(index=13, l_node=13, r_node=14),
        Element(index=14, l_node=14, r_node=1),
        Element(index=15, l_node=13, r_node=2),
        Element(index=16, l_node=12, r_node=3),
        Element(index=17, l_node=11, r_node=4),
        Element(index=18, l_node=10, r_node=5),
        Element(index=19, l_node=9, r_node=6),
        Element(index=20, l_node=14, r_node=2),
        Element(index=21, l_node=13, r_node=3),
        Element(index=22, l_node=12, r_node=4),
        Element(index=23, l_node=11, r_node=5),
        Element(index=24, l_node=10, r_node=6),
        Element(index=25, l_node=9, r_node=7)
    ]

    draw_node_element()


def load_preset6():
    global node_list, element_list
    clear()
    node_list = [
        Node(index=1, x=0.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=0.0, bc_d_y=0.0),
        Node(index=2, x=1.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=3, x=3.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=4, x=5.0, y=0.0, bc_f_x=None, bc_f_y=-100_000, bc_d_x=None, bc_d_y=None),
        Node(index=5, x=7.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=6, x=9.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=7, x=10.0, y=0.0, bc_f_x=None, bc_f_y=None, bc_d_x=0.0, bc_d_y=0.0),
        Node(index=8, x=1.0, y=2.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=9, x=3.0, y=2.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=10, x=5.0, y=2.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=11, x=7.0, y=2.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
        Node(index=12, x=9.0, y=2.0, bc_f_x=None, bc_f_y=None, bc_d_x=None, bc_d_y=None),
    ]
    element_list = [
        Element(index=1, l_node=1, r_node=2),
        Element(index=2, l_node=2, r_node=3),
        Element(index=3, l_node=3, r_node=4),
        Element(index=4, l_node=4, r_node=5),
        Element(index=5, l_node=5, r_node=6),
        Element(index=6, l_node=6, r_node=7),

        Element(index=7, l_node=1, r_node=8),
        Element(index=8, l_node=7, r_node=12),

        Element(index=9, l_node=2, r_node=8),
        Element(index=10, l_node=3, r_node=9),
        Element(index=11, l_node=4, r_node=10),
        Element(index=12, l_node=5, r_node=11),
        Element(index=13, l_node=6, r_node=12),

        Element(index=14, l_node=8, r_node=9),
        Element(index=15, l_node=9, r_node=10),
        Element(index=16, l_node=10, r_node=11),
        Element(index=17, l_node=11, r_node=12),

        Element(index=18, l_node=2, r_node=9),
        Element(index=19, l_node=3, r_node=8),
        Element(index=20, l_node=4, r_node=9),
        Element(index=21, l_node=3, r_node=10),
        Element(index=22, l_node=4, r_node=11),
        Element(index=23, l_node=5, r_node=10),
        Element(index=24, l_node=5, r_node=12),
        Element(index=25, l_node=6, r_node=11)
    ]

    draw_node_element()


def popupwin_preset3():
    # Create a Toplevel window
    top = tk.Toplevel(root)
    top.geometry("250x150")

    label = tk.Label(top, text="Format: 'd1,d2,p,m' Ex: '1,1,4,3'")
    label.pack()
    entry = tk.Entry(top, width=20)
    entry.pack()
    tk.Button(top, text="Insert", command=lambda: load_preset3(entry.get(), top)).pack(pady=5, side=tk.TOP)
    tk.Button(top, text="Cancel", command=lambda: top.destroy()).pack(pady=5, side=tk.TOP)


def load_preset3(data: str, top):
    global node_list, element_list
    clear()
    preset_data = data.split(",")
    if len(preset_data) != 4:
        return

    d1 = float(preset_data[0])
    d2 = float(preset_data[1])
    p = int(preset_data[2])
    m = int(preset_data[3])

    element_type = "D2TR3N"
    NL, EL = uniform_mesh(d1, d2, p, m, element_type)

    # Nodes
    for index, n in enumerate(NL):
        node_list.append(Node(index=index + 1, x=n[0], y=n[1]))

    # Element
    dim_type = 3
    e_index = 1
    for index, e in enumerate(EL):
        for j in range(dim_type):
            l_index = j
            r_index = j + 1 if j != dim_type - 1 else 0
            l_value, r_value = e[l_index], e[r_index]
            match = [e for e in element_list if
                     (e.l_node == l_value and e.r_node == r_value) or (e.l_node == r_value and e.r_node == l_value)]
            if not match:
                element_list.append(Element(index=e_index, l_node=l_value, r_node=r_value))
                e_index += 1

    draw_node_element()
    top.destroy()


# def load_preset4():
#     global node_list, element_list
#     clear()
#     d1 = 1
#     d2 = 1
#     p = 4
#     m = 3
#     element_type = "D2QU4N"
#     NL, EL = uniform_mesh(d1, d2, p, m, element_type)
#
#     # Nodes
#     for index, n in enumerate(NL):
#         node_list.append(Node(index=index + 1, x=n[0], y=n[1]))
#
#     # Element
#     dim_type = 4
#     e_index = 1
#     for index, e in enumerate(EL):
#         for j in range(dim_type):
#             l_index = j
#             r_index = j + 1 if j != dim_type - 1 else 0
#             l_value, r_value = e[l_index], e[r_index]
#             match = [e for e in element_list if
#                      (e.l_node == l_value and e.r_node == r_value) or (e.l_node == r_value and e.r_node == l_value)]
#             if not match:
#                 element_list.append(Element(index=e_index, l_node=l_value, r_node=r_value))
#                 e_index += 1
#
#     draw_node_element()


def popupwin_preset5():
    # Create a Toplevel window
    top = tk.Toplevel(root)
    top.geometry("250x150")

    label = tk.Label(top, text="Format: 'd1,d2,p,m,R' Ex: '1,1,4,3,0.2'")
    label.pack()
    entry = tk.Entry(top, width=20)
    entry.pack()
    tk.Button(top, text="Insert", command=lambda: load_preset5(entry.get(), top)).pack(pady=5, side=tk.TOP)
    tk.Button(top, text="Cancel", command=lambda: top.destroy()).pack(pady=5, side=tk.TOP)


def load_preset5(data: str, top):
    global node_list, element_list
    clear()
    preset_data = data.split(",")
    if len(preset_data) != 5:
        return

    d1 = float(preset_data[0])
    d2 = float(preset_data[1])
    p = int(preset_data[2])
    m = int(preset_data[3])
    R = float(preset_data[4])

    element_type = "D2TR3N"
    NL, EL = void_mesh(d1, d2, p, m, R, element_type)

    # Nodes
    for index, n in enumerate(NL):
        node_list.append(Node(index=index + 1, x=n[0], y=n[1]))

    # Element
    dim_type = 3
    e_index = 1
    for index, e in enumerate(EL):
        for j in range(dim_type):
            l_index = j
            r_index = j + 1 if j != dim_type - 1 else 0
            l_value, r_value = e[l_index], e[r_index]
            match = [e for e in element_list if
                     (e.l_node == l_value and e.r_node == r_value) or (e.l_node == r_value and e.r_node == l_value)]
            if not match:
                element_list.append(Element(index=e_index, l_node=l_value, r_node=r_value))
                e_index += 1

    draw_node_element()
    top.destroy()


# def load_preset6():
#     global node_list, element_list
#     clear()
#     d1 = 1
#     d2 = 1
#     p = 4
#     m = 3
#     R = 0.2
#     element_type = "D2QU4N"
#     NL, EL = void_mesh(d1, d2, p, m, R, element_type)
#
#     # Nodes
#     for index, n in enumerate(NL):
#         node_list.append(Node(index=index + 1, x=n[0], y=n[1]))
#
#     # Element
#     dim_type = 4
#     e_index = 1
#     for index, e in enumerate(EL):
#         for j in range(dim_type):
#             l_index = j
#             r_index = j + 1 if j != dim_type - 1 else 0
#             l_value, r_value = e[l_index], e[r_index]
#             match = [e for e in element_list if
#                      (e.l_node == l_value and e.r_node == r_value) or (e.l_node == r_value and e.r_node == l_value)]
#             if not match:
#                 element_list.append(Element(index=e_index, l_node=l_value, r_node=r_value))
#                 e_index += 1
#
#     draw_node_element()


def load_file():
    global node_list, element_list
    filename = askopenfile()
    if not filename:
        return

    with open(filename.name, "r", encoding="utf-8") as file:
        clear()
        for line in file:
            data = line.strip().split("\t")
            if data[0] == "N":
                node_list.append(
                    Node(
                        index=int(data[1]),
                        x=float(data[2]),
                        y=float(data[3]),
                        bc_f_x=None if data[4] == "None" else float(data[4]),
                        bc_f_y=None if data[5] == "None" else float(data[5]),
                        bc_d_x=None if data[6] == "None" else float(data[6]),
                        bc_d_y=None if data[7] == "None" else float(data[7])
                    )
                )
            else:
                element_list.append(Element(index=int(data[1]), l_node=int(data[2]), r_node=int(data[3])))
    draw_node_element()


def save_file():
    global node_list, element_list
    filename = asksaveasfile(filetypes=(("TXT Files", "*.txt"), ("All files", "*.txt")), defaultextension=".txt")
    if not filename:
        return

    with open(filename.name, "w", encoding="utf-8") as file:
        for n in node_list:
            file.write(f"N\t{n.index}\t{n.x}\t{n.y}\t{n.bc_f_x}\t{n.bc_f_y}\t{n.bc_d_x}\t{n.bc_d_y}\n")
        for e in element_list:
            file.write(f"E\t{e.index}\t{e.l_node}\t{e.r_node}\n")


def switch_slider(x):
    global slider1, SCALE
    value = 10 ** (slider1.get() - 3)
    # value = float(slider1.get())
    SCALE = value
    var_scale.set(f"Scale Factor: {SCALE}")
    if not ENL.any():
        return
    run_post_process()


def switch_node_active(var):
    global node_labels_active
    node_labels_active = var
    draw_node_element()


def switch_element_active(var):
    global element_labels_active
    element_labels_active = var
    draw_node_element()


def show_about():
    messagebox.showinfo(title="About",
                        message="This program, which has been created as a part of our ME 362 Project, is meant to be a tool that utilizes FEM in such a way that it can show the displacements of any given truss structure, input by the user, along with the boundary conditions and acting forces. The program, as stated before, can show displacements, and it can exaggerate said displacements so that it is easier to see them.\n\nMade by:\nCan Gürsu\nTuna Atasoy\nYunus Özkan")


def items_selected(event):
    global ENL, NL, EL, node_list, element_list
    selected_langs = listbox.get(tk.ANCHOR)

    if selected_langs == "":
        return
    data = [i for i in iter_list if i[0] == selected_langs][0]
    iteration, ENL, NL, EL, node_list, element_list = data

    draw_node_element()
    run_post_process()


root = tk.Tk()
root.title("FEM Project 1")
root.geometry("1000x600")
root.wm_minsize(1000, 600)
root.state('zoomed')
root.iconbitmap('./icon.ico')

style = Style(root)
style.theme_use("vista")

# Frame Preprocess Generate
frame_preprocess_generate = tk.Frame(root)
frame_preprocess_generate.place(relwidth=0.50, relheight=0.5, relx=0.00, rely=0.00)

# Frame Preprocess Right
frame_preprocess_right = tk.Frame(root)
frame_preprocess_right.place(relwidth=0.50, relheight=0.5, relx=0.50, rely=0.00)

# Frame Postprocess Left
frame_postprocess_left = tk.Frame(root)
frame_postprocess_left.place(relwidth=0.35, relheight=0.5, relx=0.00, rely=0.5)

# Frame Postprocess Middle
frame_postprocess_middle = tk.Frame(root)
frame_postprocess_middle.place(relwidth=0.35, relheight=0.5, relx=0.35, rely=0.5)

# Frame Postprocess Right
frame_postprocess_right = tk.Frame(root)
frame_postprocess_right.place(relwidth=0.30, relheight=0.5, relx=0.70, rely=0.5)

var_node_label = tk.BooleanVar()
var_node_label.set(True)
var_element_label = tk.BooleanVar()
var_element_label.set(True)

frame_preprocess_node = tk.Frame(frame_preprocess_generate)
frame_preprocess_node.pack(fill=tk.X, pady=16)

frame_preprocess_element = tk.Frame(frame_preprocess_generate)
frame_preprocess_element.pack(fill=tk.X, pady=32)

frame_preprocess_bc = tk.Frame(frame_preprocess_generate)
frame_preprocess_bc.pack(fill=tk.X, pady=2)
frame_preprocess_bc1 = tk.Frame(frame_preprocess_generate)
frame_preprocess_bc1.pack(fill=tk.X, pady=2)

frame_preprocess_material = tk.Frame(frame_preprocess_generate)
frame_preprocess_material.pack(fill=tk.X, pady=32)

frame_preprocess_slider = tk.Frame(frame_preprocess_generate)
frame_preprocess_slider.pack(fill=tk.X, pady=8)

# Nodes
label_node = tk.Label(frame_preprocess_node, text="Generate Nodes", anchor="w", width=15)
label_node.pack(side=tk.LEFT)
label_node_x = tk.Label(frame_preprocess_node, text="X :", anchor="e", width=3)
label_node_x.pack(side=tk.LEFT)
entry_node_x = tk.Entry(frame_preprocess_node, width=3)
entry_node_x.pack(side=tk.LEFT)
label_node_y = tk.Label(frame_preprocess_node, text="Y :", anchor="e", width=3)
label_node_y.pack(side=tk.LEFT)
entry_node_y = tk.Entry(frame_preprocess_node, width=3)
entry_node_y.pack(side=tk.LEFT)
btn_node = tk.Button(frame_preprocess_node, text="Generate", command=generate_node)
btn_node.pack(side=tk.LEFT, padx=10)
label_node_number = tk.Label(frame_preprocess_node, text="Node #:", width=10, anchor="e")
label_node_number.pack(side=tk.LEFT)
label_node_number1 = tk.Label(frame_preprocess_node, text=str(len(node_list)))
label_node_number1.pack(side=tk.LEFT)

# Element
label_element = tk.Label(frame_preprocess_element, text="Generate Elements", anchor="w", width=15)
label_element.pack(side=tk.LEFT)
label_element_x = tk.Label(frame_preprocess_element, text="X :", anchor="e", width=3)
label_element_x.pack(side=tk.LEFT)
entry_element_x = tk.Entry(frame_preprocess_element, width=3)
entry_element_x.pack(side=tk.LEFT)
label_element_y = tk.Label(frame_preprocess_element, text="Y :", anchor="e", width=3)
label_element_y.pack(side=tk.LEFT)
entry_element_y = tk.Entry(frame_preprocess_element, width=3)
entry_element_y.pack(side=tk.LEFT)
btn_element = tk.Button(frame_preprocess_element, text="Generate", command=generate_element)
btn_element.pack(side=tk.LEFT, padx=10)
label_element_number = tk.Label(frame_preprocess_element, text="Element #:", width=10, anchor="e")
label_element_number.pack(side=tk.LEFT)
label_element_number1 = tk.Label(frame_preprocess_element, text=str(len(element_list)))
label_element_number1.pack(side=tk.LEFT)

# Boundary Conditions
label_bc = tk.Label(frame_preprocess_bc, text="Generate BC", anchor="w", width=15)
label_bc.pack(side=tk.LEFT)
label_bc_no = tk.Label(frame_preprocess_bc, text="Node ID :", anchor="e", width=10)
label_bc_no.pack(side=tk.LEFT)
entry_bc = tk.Entry(frame_preprocess_bc, width=3)
entry_bc.pack(side=tk.LEFT, padx=10)

var_direction = tk.StringVar()
var_direction.set("X")
radioX = tk.Radiobutton(frame_preprocess_bc, variable=var_direction, text="X", value="X", width=6, anchor="w")
radioX.pack(side=tk.LEFT)

radioY = tk.Radiobutton(frame_preprocess_bc, variable=var_direction, text="Y", value="Y", width=10, anchor="w")
radioY.pack(side=tk.LEFT)

label_bc = tk.Label(frame_preprocess_bc1, text="", anchor="w", width=15)
label_bc.pack(side=tk.LEFT)
label_magnitude = tk.Label(frame_preprocess_bc1, text="Magnitude :", anchor="e", width=10)
label_magnitude.pack(side=tk.LEFT)
entry_magnitude = tk.Entry(frame_preprocess_bc1, width=3)
entry_magnitude.pack(side=tk.LEFT, padx=10)

var_type = tk.StringVar()
var_type.set("Force")
radioForce = tk.Radiobutton(frame_preprocess_bc1, variable=var_type, text="Force", value="Force", width=6, anchor="w")
radioForce.pack(side=tk.LEFT)

radioDisplacement = tk.Radiobutton(frame_preprocess_bc1, variable=var_type, text="Displacement", value="Displacement",
                                   width=10, anchor="w")
radioDisplacement.pack(side=tk.LEFT)

btn_bc_generate = tk.Button(frame_preprocess_bc, text="Generate", command=generate_bc)
btn_bc_generate.pack(side=tk.LEFT, padx=20)

# Material
label_material = tk.Label(frame_preprocess_material, text="Material Properties", anchor="w", width=15)
label_material.pack(side=tk.LEFT)
label_modulus = tk.Label(frame_preprocess_material, text="E :", anchor="e", width=3)
label_modulus.pack(side=tk.LEFT)
entry_modulus = tk.Entry(frame_preprocess_material, width=12)
entry_modulus.insert(0, str(E))
entry_modulus.pack(side=tk.LEFT)
label_area = tk.Label(frame_preprocess_material, text="A :", anchor="e", width=3)
label_area.pack(side=tk.LEFT)
entry_area = tk.Entry(frame_preprocess_material, width=10)
entry_area.insert(0, str(A))
entry_area.pack(side=tk.LEFT)
btn_run_process = tk.Button(frame_preprocess_material, text="RUN", command=run_process)
btn_run_process.pack(side=tk.LEFT, padx=64)
btn_iterate = tk.Button(frame_preprocess_material, text="ITERATE", command=iterate_process)
btn_iterate.pack(side=tk.LEFT, padx=32)

# Slider
var_scale = tk.StringVar()
var_scale.set(f"Scale Factor: {SCALE}")
label_slider = tk.Label(frame_preprocess_slider, textvariable=var_scale, width=20)
label_slider.pack(side=tk.LEFT)
slider1 = tk.Scale(frame_preprocess_slider, from_=1, to=5, orient=tk.HORIZONTAL, width=20, length=300,
                   command=switch_slider)
slider1.pack(side=tk.LEFT)

# Menubar
menubar = tk.Menu(root)

# Top level menus
file_menu = tk.Menu(menubar, tearoff=0)
edit_menu = tk.Menu(menubar, tearoff=0)
window_menu = tk.Menu(menubar, tearoff=0)
help_menu = tk.Menu(menubar, tearoff=0)

menubar.add_cascade(label="File", menu=file_menu)
menubar.add_cascade(label="Edit", menu=edit_menu)
menubar.add_cascade(label="Window", menu=window_menu)
menubar.add_cascade(label="Help", menu=help_menu)

# File menu
file_menu.add_command(label="Load", command=load_file)
file_menu.add_command(label="Save", command=save_file)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Edit Menu
preset_menu = tk.Menu(edit_menu, tearoff=0)
edit_menu.add_cascade(label="Presets", menu=preset_menu)
edit_menu.add_separator()
edit_menu.add_command(label="Clear", command=clear)
edit_menu.add_command(label="Clear BC", command=clear_bc)

preset_menu.add_command(label="Preset 1 - Triangle", command=load_preset1)
preset_menu.add_command(label="Preset 2 - Long beam", command=load_preset2)
preset_menu.add_command(label="Preset 3 - Bridge", command=load_preset6)
preset_menu.add_command(label="Preset 4 - Uniform mesh - D2TR3N", command=popupwin_preset3)
# preset_menu.add_command(label="Preset 4 - Uniform mesh - D2QU4N", command=load_preset4)
preset_menu.add_command(label="Preset 5 - Void mesh - D2TR3N", command=popupwin_preset5)
# preset_menu.add_command(label="Preset 6 - Void mesh - D2QU4N", command=load_preset6)

# Window Menu
window_menu.add_checkbutton(label="Node Labels", onvalue=True, offvalue=False, variable=var_node_label,
                            command=lambda: switch_node_active(var_node_label.get()))
window_menu.add_checkbutton(label="Element Labels", onvalue=True, offvalue=False, variable=var_element_label,
                            command=lambda: switch_element_active(var_element_label.get()))

# Help menu
help_menu.add_command(label="About", command=show_about)

root.config(menu=menubar)

# Initialize functions
draw_node_element()

node_force = Figure(figsize=(5, 5), dpi=100)
ax_force = node_force.add_subplot(111)
canvas1 = FigureCanvasTkAgg(node_force, master=frame_postprocess_middle)
toolbar2 = NavigationToolbar2Tk(canvas1, frame_postprocess_middle)
toolbar2.update()
toolbar2.pack()
canvas1.draw()
canvas1.get_tk_widget().pack(side=tk.TOP)

node_disp = Figure(figsize=(5, 5), dpi=100)
ax_disp = node_disp.add_subplot(111)
canvas2 = FigureCanvasTkAgg(node_disp, master=frame_postprocess_left)
toolbar3 = NavigationToolbar2Tk(canvas2, frame_postprocess_left)
toolbar3.update()
toolbar3.pack()
canvas2.draw()
canvas2.get_tk_widget().pack(side=tk.TOP)

iter_var = tk.StringVar(value=iter_list)
label_iter = tk.Label(frame_postprocess_right, text="Iteration List", anchor="w", width=15)
label_iter.pack()
listbox = tk.Listbox(frame_postprocess_right, listvariable=iter_var, height=15, selectmode='extended')
listbox.pack()

listbox.bind('<<ListboxSelect>>', items_selected)

root.mainloop()
