import random
import tkinter as tk
from tkinter import ttk
from tkinter.ttk import *
from tkinter import messagebox
import numpy as np
from functions import *
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from fem_types import Node, Element
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from PIL import ImageTk, Image

# Config
DEBUG = True

# Values
NL = np.array([])
EL = np.array([])
DorN = np.array([])
Fu = np.array([])
U_u = np.array([])
ENL = np.array([])  # X Y XBC YBC TEMP_X TEMP_Y GLOBAL_X GLOBAL_Y DISP_X DISP_Y FORCE_X FORCE_Y
E, A = 80_000_000_000 / 3, 0.01  # 210 GPA
SCALE = 0.01
node_labels_active = False
element_labels_active = False
node_list = []
element_list = []


def generate_mesh():
    global node_list, element_list, element_type
    clear()

    d1 = float(entry_a.get())
    d2 = float(entry_b.get())
    p = int(entry_c.get())
    m = int(entry_d.get())
    R = float(entry_e.get())

    # element_type = "D2TR3N"
    el_type = element_type.get()
    shape_type = shape.get()
    NL, EL = void_mesh(d1, d2, p, m, R, el_type, shape_type)

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
    label_node_number1.configure(text=str(len(node_list)))
    label_element_number1.configure(text=str(len(element_list)))


def reset_mesh():
    clear()
    label_node_number1.configure(text=str(len(node_list)))
    label_element_number1.configure(text=str(len(element_list)))


def clear():
    global node_list, element_list
    node_list = []
    element_list = []
    draw_node_element()


def run_process():
    global U_u, Fu, NL, EL, DorN, ENL, E, A, node_list, element_list

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


def run_post_process():
    global node_list, element_list, element_type

    d1 = float(entry_a.get())
    d2 = float(entry_b.get())
    p = int(entry_c.get())
    m = int(entry_d.get())
    R = float(entry_e.get())
    defV = 1 / 3

    # element_type = "D2TR3N"
    el_type = element_type.get()
    shape_type = shape.get()
    NL, EL = void_mesh(d1, d2, p, m, R, el_type, shape_type)

    # BC_flag = 'extension'  # extension expansion shear
    deformation = deformation_type.get()
    if deformation == 1:
        BC_flag = "extension"
    elif deformation == 2:
        BC_flag = "expansion"
    else:
        BC_flag = "shear"

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

    # scale = float(var_scale.get().split(" ")[2])
    scale = 0.5

    (stress_xx, stress_xy, stress_yx, stress_yy,
     strain_xx, strain_xy, strain_yx, strain_yy,
     disp_x, disp_y, X, Y) = post_process(NL, EL, ENL, scale)

    graph = graph_type.get()

    for child in frame_analysis.winfo_children():
        child.destroy()

    node_figure = Figure(figsize=(5, 5), dpi=100)
    ax_dispx = node_figure.add_subplot(111)

    if "stress" in graph:
        ax_dispx.set_title(graph)
        chosen_stress = stress_xx
        if graph == "stress xx":
            chosen_stress = stress_xx
        elif graph == "stress xy":
            chosen_stress = stress_xy
        elif graph == "stress yx":
            chosen_stress = stress_yx
        elif graph == "stress yy":
            chosen_stress = stress_yy

        stress_xxNormalized = (chosen_stress - chosen_stress.min()) / (chosen_stress.max() - chosen_stress.min())

        cmap = plt.get_cmap("jet")
        norm_x = Normalize(np.abs(chosen_stress.min()), np.abs(chosen_stress.max()))
        node_figure.colorbar(ScalarMappable(norm=norm_x, cmap=cmap))

        for i in range(np.size(EL, 0)):
            x = X[:, i]
            y = Y[:, i]
            c = stress_xxNormalized[:, i]

            # Setting colormap boundaries
            color = color_map.get()
            cmap = truncate_colormap(plt.get_cmap(color), c.min(), c.max())
            # Plot the colors
            ax_dispx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
            ax_dispx.plot(x, y, 'k-', linewidth=0.5)

    if "disp" in graph:
        ax_dispx.set_title('Displacement X')
        chosen_stress = stress_xx
        if graph == "disp x":
            chosen_stress = stress_xx
        elif graph == "disp y":
            chosen_stress = stress_xy

        disp_xNormalized = (chosen_stress - chosen_stress.min()) / (chosen_stress.max() - chosen_stress.min())

        cmap = plt.get_cmap("jet")
        norm_x = Normalize(np.abs(chosen_stress.min()), np.abs(chosen_stress.max()))
        node_figure.colorbar(ScalarMappable(norm=norm_x, cmap=cmap))

        for i in range(np.size(EL, 0)):
            x = X[:, i]
            y = Y[:, i]
            c = disp_xNormalized[:, i]
            cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
            ax_dispx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')

            ax_dispx.plot(x, y, 'k', linewidth=0.5)

    canvas1 = FigureCanvasTkAgg(node_figure, master=frame_analysis)
    toolbar1 = NavigationToolbar2Tk(canvas1, frame_analysis)
    toolbar1.update()
    toolbar1.pack()
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.TOP)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N  # what is this .N ???

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap


def draw_node_element():
    global node_labels_active, element_labels_active, node_list, element_list

    for child in frame_mesh.winfo_children():
        child.destroy()

    # label_node_number1.configure(text=str(len(node_list)))
    # label_element_number1.configure(text=str(len(element_list)))

    node_figure = Figure(figsize=(6, 5), dpi=100)
    ax_dispx = node_figure.add_subplot(111)

    ax_dispx.axes.set_aspect("equal")
    ax_dispx.axes.set_adjustable("datalim")

    # ax_dispx = node_figure.add_subplot(111, aspect="equal")
    canvas1 = FigureCanvasTkAgg(node_figure, master=frame_mesh)
    toolbar2 = NavigationToolbar2Tk(canvas1, frame_mesh)
    toolbar2.update()
    toolbar2.pack()
    # canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas1.get_tk_widget().place(relwidth=1.0, relheight=0.9, relx=0.0, rely=0.0)

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

        ax_dispx.plot((left_node.x, right_node.x), (left_node.y, right_node.y), linewidth=2, zorder=8, color="black")

    canvas1.draw()


def draw_post_process_displacement(x_scatter, y_scatter, color_x, dispx_array):
    global node_list, element_list

    for child in frame_analysis.winfo_children():
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

    canvas1 = FigureCanvasTkAgg(node_figure, master=frame_analysis)
    toolbar1 = NavigationToolbar2Tk(canvas1, frame_analysis)
    toolbar1.update()
    toolbar1.pack()
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.TOP)
    # canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def draw_post_process_force(x_scatter, y_scatter, stress_list, dispx_array):
    global node_list, element_list

    for child in frame_mesh.winfo_children():
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

    canvas2 = FigureCanvasTkAgg(node_figure, master=frame_analysis)
    toolbar2 = NavigationToolbar2Tk(canvas2, frame_analysis)
    toolbar2.update()
    toolbar2.pack()
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.TOP)


def switch_slider(x):
    global slider1, SCALE
    value = 10 ** (slider1.get() - 3)
    # value = float(slider1.get())
    SCALE = value
    var_scale.set(f"Scale Factor: {SCALE}")
    if not ENL.any():
        return
    run_post_process()


def call_post_process(x):
    run_post_process()


def set_shape(x):
    global frame_image, shape
    shape.set(x)
    # for child in frame_image.winfo_children():
    #     child.destroy()
    # frame_image = tk.Frame(frame_dimensions, highlightbackground="gray", highlightthickness=1)
    # frame_image.place(relwidth=0.60, relheight=0.60, relx=0.00, rely=0.30)
    # img = ImageTk.PhotoImage(Image.open(f"{x}.jpeg").resize((150, 150)))
    # print(img)
    # label = Label(frame_image, image=img)
    # label.pack()


root = tk.Tk()
root.title("FEM Project 2")
root.geometry("1000x600")
root.wm_minsize(1000, 600)
root.state('zoomed')
root.iconbitmap('./icon.ico')

style = Style(root)
style.theme_use("vista")

# Frame Inclusion
frame_inclusion = tk.Frame(root, highlightbackground="gray", highlightthickness=1)
frame_inclusion.place(relwidth=0.20, relheight=0.4, relx=0.00, rely=0.00)

shape = tk.StringVar()
shape.set("circle")
lbl_inclusion = tk.Label(frame_inclusion, text="Include Shape")
lbl_inclusion.place(relwidth=1.0, relheight=0.2, relx=0.00, rely=0.00)
btn_inclusion_circle = ttk.Button(frame_inclusion, text="Circle", command=lambda: set_shape("circle"))
btn_inclusion_circle.place(relwidth=0.8, relheight=0.2, relx=0.10, rely=0.20)
btn_inclusion_square = ttk.Button(frame_inclusion, text="Square", command=lambda: set_shape("square"))
btn_inclusion_square.place(relwidth=0.8, relheight=0.2, relx=0.10, rely=0.45)
btn_inclusion_rhombus = ttk.Button(frame_inclusion, text="Rhombus", command=lambda: set_shape("rhombus"))
btn_inclusion_rhombus.place(relwidth=0.8, relheight=0.2, relx=0.10, rely=0.70)

# Frame Dimensions
frame_dimensions = tk.Frame(root, highlightbackground="gray", highlightthickness=1)
frame_dimensions.place(relwidth=0.20, relheight=0.4, relx=0.20, rely=0.00)

lbl_dimensions = tk.Label(frame_dimensions, text="Dimensions and Number of Partitions")
lbl_dimensions.place(relwidth=1.0, relheight=0.2, relx=0.00, rely=0.00)
frame_image = tk.Frame(frame_dimensions, highlightbackground="gray", highlightthickness=1)
frame_image.place(relwidth=0.60, relheight=0.60, relx=0.00, rely=0.30)
img = ImageTk.PhotoImage(Image.open("circle.jpeg").resize((150, 150)))
label = Label(frame_image, image=img)
label.pack()

lbl_a = tk.Label(frame_dimensions, text="a")
lbl_a.place(relwidth=0.1, relheight=0.1, relx=0.60, rely=0.35)
entry_a = tk.Entry(frame_dimensions)
entry_a.insert(0, str(1))
entry_a.place(relwidth=0.3, relheight=0.1, relx=0.70, rely=0.35)
lbl_b = tk.Label(frame_dimensions, text="b")
lbl_b.place(relwidth=0.1, relheight=0.1, relx=0.60, rely=0.45)
entry_b = tk.Entry(frame_dimensions)
entry_b.insert(0, str(1))
entry_b.place(relwidth=0.3, relheight=0.1, relx=0.70, rely=0.45)
lbl_c = tk.Label(frame_dimensions, text="c")
lbl_c.place(relwidth=0.1, relheight=0.1, relx=0.60, rely=0.55)
entry_c = tk.Entry(frame_dimensions)
entry_c.insert(0, str(4))
entry_c.place(relwidth=0.3, relheight=0.1, relx=0.70, rely=0.55)
lbl_d = tk.Label(frame_dimensions, text="d")
lbl_d.place(relwidth=0.1, relheight=0.1, relx=0.60, rely=0.65)
entry_d = tk.Entry(frame_dimensions)
entry_d.insert(0, str(4))
entry_d.place(relwidth=0.3, relheight=0.1, relx=0.70, rely=0.65)
lbl_e = tk.Label(frame_dimensions, text="e")
lbl_e.place(relwidth=0.1, relheight=0.1, relx=0.60, rely=0.75)
entry_e = tk.Entry(frame_dimensions)
entry_e.insert(0, str(0.2))
entry_e.place(relwidth=0.3, relheight=0.1, relx=0.70, rely=0.75)

# Frame Type
frame_type = tk.Frame(root, highlightbackground="gray", highlightthickness=1)
frame_type.place(relwidth=0.20, relheight=0.4, relx=0.40, rely=0.00)

lbl_type = tk.Label(frame_type, text="Element Type")
lbl_type.place(relwidth=1.0, relheight=0.2, relx=0.00, rely=0.00)
# var1 = tk.IntVar()
# cb_type = tk.Checkbutton(frame_type, text='Inclusion', variable=var1, onvalue=1, offvalue=0, command=print_selection)
# cb_type.place(relwidth=1.0, relheight=0.1, relx=0.00, rely=0.20)

options = [
    "D2TR3N",
    "D2QU4N",
    "D2TR6N",
    "D2TR8N",
    "D2TR9N",
]
element_type = tk.StringVar()
element_type.set(options[0])
drop_type = tk.OptionMenu(frame_type, element_type, *options)
drop_type.place(relwidth=0.80, relheight=0.1, relx=0.10, rely=0.30)

# Frame Mesh
frame_mesh = tk.Frame(root, highlightbackground="gray", highlightthickness=1)
frame_mesh.place(relwidth=0.20, relheight=0.4, relx=0.60, rely=0.00)

# Frame Generate
frame_generate = tk.Frame(root, highlightbackground="gray", highlightthickness=1)
frame_generate.place(relwidth=0.20, relheight=0.4, relx=0.80, rely=0.00)

lbl_generate = tk.Label(frame_generate, text="Generate")
lbl_generate.place(relwidth=1.0, relheight=0.2, relx=0.00, rely=0.00)
btn_generate = ttk.Button(frame_generate, text="Generate Mesh", command=generate_mesh)
btn_generate.place(relwidth=0.8, relheight=0.2, relx=0.10, rely=0.20)
btn_generate_reset = ttk.Button(frame_generate, text="Reset Mesh", command=reset_mesh)
btn_generate_reset.place(relwidth=0.8, relheight=0.2, relx=0.10, rely=0.40)

label_node_number = tk.Label(frame_generate, text="Node #:")
label_node_number.place(relwidth=0.8, relheight=0.1, relx=0.10, rely=0.60)
label_node_number1 = tk.Label(frame_generate, text=str(len(node_list)))
label_node_number1.place(relwidth=0.8, relheight=0.1, relx=0.10, rely=0.70)
label_element_number = tk.Label(frame_generate, text="Element #:")
label_element_number.place(relwidth=0.8, relheight=0.1, relx=0.10, rely=0.80)
label_element_number1 = tk.Label(frame_generate, text=str(len(element_list)))
label_element_number1.place(relwidth=0.8, relheight=0.1, relx=0.10, rely=0.90)

# Frame Material
frame_material = tk.Frame(root, highlightbackground="gray", highlightthickness=1)
frame_material.place(relwidth=0.20, relheight=0.6, relx=0.00, rely=0.40)

lbl_material = tk.Label(frame_material, text="Material Properties")
lbl_material.place(relwidth=1.0, relheight=0.2, relx=0.00, rely=0.00)
lbl_material_matrix = tk.Label(frame_material, text="Matrix")
lbl_material_matrix.place(relwidth=0.4, relheight=0.1, relx=0.10, rely=0.2)
lbl_material_inclusion = tk.Label(frame_material, text="Inclusion")
lbl_material_inclusion.place(relwidth=0.4, relheight=0.1, relx=0.50, rely=0.2)

lbl_material_e = tk.Label(frame_material, text="E")
lbl_material_e.place(relwidth=0.1, relheight=0.1, relx=0.0, rely=0.4)
entry_material_e = tk.Entry(frame_material)
entry_material_e.insert(0, str(1))
entry_material_e.place(relwidth=0.4, relheight=0.1, relx=0.1, rely=0.4)
entry_inclusion_e = tk.Entry(frame_material)
entry_inclusion_e.insert(0, str(10))
entry_inclusion_e.place(relwidth=0.4, relheight=0.1, relx=0.5, rely=0.4)

lbl_material_v = tk.Label(frame_material, text="v")
lbl_material_v.place(relwidth=0.1, relheight=0.1, relx=0.0, rely=0.6)
entry_material_v = tk.Entry(frame_material)
entry_material_v.insert(0, str(0.3))
entry_material_v.place(relwidth=0.4, relheight=0.1, relx=0.1, rely=0.6)
entry_inclusion_v = tk.Entry(frame_material)
entry_inclusion_v.insert(0, str(0.3))
entry_inclusion_v.place(relwidth=0.4, relheight=0.1, relx=0.5, rely=0.6)

# Frame Deformation
frame_deformation = tk.Frame(root, highlightbackground="gray", highlightthickness=1)
frame_deformation.place(relwidth=0.20, relheight=0.6, relx=0.20, rely=0.40)

lbl_deformation = tk.Label(frame_deformation, text="Deformation Type")
lbl_deformation.place(relwidth=1.0, relheight=0.2, relx=0.00, rely=0.00)

deformation_type = tk.IntVar()
deformation_type.set(1)
rb_extension = Radiobutton(frame_deformation, text="Extension", variable=deformation_type, value=1)
rb_extension.place(relwidth=0.30, relheight=0.1, relx=0.0, rely=0.20)
rb_expansion = Radiobutton(frame_deformation, text="Expansion", variable=deformation_type, value=2)
rb_expansion.place(relwidth=0.30, relheight=0.1, relx=0.30, rely=0.20)
rb_shear = Radiobutton(frame_deformation, text="Shear", variable=deformation_type, value=3)
rb_shear.place(relwidth=0.30, relheight=0.1, relx=0.6, rely=0.20)

# lbl_deformation_mag = tk.Label(frame_deformation, text="Deformation Magnitude")
# lbl_deformation_mag.place(relwidth=0.5, relheight=0.1, relx=0.0, rely=0.4)
# entry_deformation_mag = tk.Entry(frame_deformation)
# entry_deformation_mag.insert(0, str(0.1))
# entry_deformation_mag.place(relwidth=0.4, relheight=0.1, relx=0.5, rely=0.4)

var_scale = tk.StringVar()
var_scale.set(f"Scale Factor: {SCALE}")
label_slider = tk.Label(frame_deformation, textvariable=var_scale, width=20)
label_slider.place(relwidth=0.8, relheight=0.2, relx=0.1, rely=0.6)
slider1 = tk.Scale(frame_deformation, from_=1, to=5, orient=tk.HORIZONTAL, width=20, length=300,
                   command=switch_slider)
slider1.place(relwidth=0.8, relheight=0.2, relx=0.1, rely=0.8)

# Frame Run
frame_run = tk.Frame(root, highlightbackground="gray", highlightthickness=1)
frame_run.place(relwidth=0.20, relheight=0.6, relx=0.40, rely=0.40)

lbl_run = ttk.Label(frame_run, text="Run Script")
lbl_run.place(relwidth=1.0, relheight=0.2, relx=0.00, rely=0.00)
btn_run = ttk.Button(frame_run, text="Run", command=run_post_process)
btn_run.place(relwidth=0.8, relheight=0.2, relx=0.10, rely=0.20)

color_map = tk.StringVar()
color_map.set("jet")
rb_copper = Radiobutton(frame_run, text="Copper", variable=color_map, value="copper")
rb_copper.place(relwidth=0.25, relheight=0.1, relx=0.00, rely=0.60)
rb_jet = Radiobutton(frame_run, text="Jet", variable=color_map, value="jet")
rb_jet.place(relwidth=0.25, relheight=0.1, relx=0.25, rely=0.60)
rb_cool = Radiobutton(frame_run, text="Cool", variable=color_map, value="cool")
rb_cool.place(relwidth=0.25, relheight=0.1, relx=0.50, rely=0.60)
rb_bone = Radiobutton(frame_run, text="Bone", variable=color_map, value="bone")
rb_bone.place(relwidth=0.25, relheight=0.1, relx=0.75, rely=0.60)

# Frame Analysis
frame_analysis = tk.Frame(root, highlightbackground="red", highlightthickness=1)
frame_analysis.place(relwidth=0.40, relheight=0.50, relx=0.60, rely=0.50)

lbl_material = tk.Label(root, text="Analysis")
lbl_material.place(relwidth=0.40, relheight=0.05, relx=0.60, rely=0.40)

options = [
    "stress xx",
    "stress xy",
    "stress yx",
    "stress yy",
    # "strain xx",
    # "strain xy",
    # "strain yx",
    # "strain yy",
    "disp x",
    "disp y"
]
graph_type = tk.StringVar()
graph_type.set(options[0])
drop_type = ttk.OptionMenu(root, graph_type, *options, command=call_post_process)
drop_type.place(relwidth=0.40, relheight=0.05, relx=0.60, rely=0.45)

# Initialize functions
draw_node_element()

root.mainloop()
