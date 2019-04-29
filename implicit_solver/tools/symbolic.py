"""
@author: Vincent Bonnet
@description : usage symbolic python to derive the constraint functions
"""

from sympy.physics.vector import ReferenceFrame
from sympy import symbols, diff, simplify, latex
from sympy.matrices import Matrix
import sympy

def bending_energy_derivation():
    '''
    Symbolic derivation of bending energy
    '''
    # Create three positions (X0, x1, x2)
    stiffness = symbols("stiffness",real=True)
    rest_angle = symbols("rest_angle",real=True)
    px0,px1,px2=symbols("px0 px1 px2",real=True)
    py0,py1,py2=symbols("py0 py1 py2",real=True)
    x0=Matrix([px0, py0])
    x1=Matrix([px1, py1])
    x2=Matrix([px2, py2])

    # Angle formula
    t01 = x1 - x0
    t12 = x2 - x1
    det = t01[0]*t12[1] - t01[1]*t12[0]
    dot = t01[0]*t12[0] + t01[1]*t12[1]
    angle = sympy.atan2(det, dot)

    # Arc Length
    arc_length = (t01.norm() + t12.norm()) * 0.5

    # Bending Energy
    bending_energy = 0.5 * stiffness * ((angle - rest_angle)**2) * arc_length

    # Derivatives
    dEdx0 = [simplify(diff(bending_energy, px0)), simplify(diff(bending_energy, py0))]
    dEdx2 = [simplify(diff(bending_energy, px2)), simplify(diff(bending_energy, py2))]
    #dEdx1 = - dEdx0 - dEdx2

def area_energy_derivation():
    '''
    Symbolic derivation of area energy
    '''
    # Create three positions (X0, x1, x2)
    stiffness = symbols("stiffness",real=True)
    rest_area = symbols("rest_area",real=True)
    px0,px1,px2=symbols("px0 px1 px2",real=True)
    py0,py1,py2=symbols("py0 py1 py2",real=True)
    x0=Matrix([px0, py0])
    x1=Matrix([px1, py1])
    x2=Matrix([px2, py2])

    # Area
    u = x1 - x0
    v = x2 - x0

    det = u[0]*v[1]-v[0]*u[1] # cross product
    area = sympy.Abs(det)  * 0.5

    # Area Energy
    area_energy = 0.5 * stiffness * ((area - rest_area)**2)

    # Derivatives
    dEdx0 = [simplify(diff(area_energy, px0)), simplify(diff(area_energy, py0))]
    dEdx2 = [simplify(diff(area_energy, px2)), simplify(diff(area_energy, py2))]
    #dEdx1 = - dEdx0 - dEdx2
