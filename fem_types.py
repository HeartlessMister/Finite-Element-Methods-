from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    index: int
    x: float
    y: float
    bc_f_x: Optional[float] = None
    bc_f_y: Optional[float] = None
    bc_d_x: Optional[float] = None
    bc_d_y: Optional[float] = None


@dataclass
class Element:
    index: int
    l_node: int
    r_node: int
