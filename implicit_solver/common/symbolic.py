"""
@author: Vincent Bonnet
@description : usage symbolic python to derive the constraint functions
"""

from sympy.physics.vector import ReferenceFrame
from sympy import symbols, diff, simplify
from sympy.matrices import Matrix
import math

# 
x0,y0=symbols("x0 y0",real=True)
x1,y1=symbols("x1 y1",real=True)
d0=Matrix([x0, y0])
d1=Matrix([x1, y1])

# Compute the derivative of spring energy
rest = symbols("rest",real=True)
stiffness = symbols("stiffness",real=True)
f = (d0 - d1).norm()
f = 0.5 * stiffness * (f - rest)**2
#print(f)
print(simplify(diff(f, x0, y0)))
#print(diff(v0.norm(v1), x0))

print(math.sqrt(0))


'''
    direction = x1 - x0
    stretch = np.linalg.norm(direction)
    if not np.isclose(stretch, 0.0):
        direction /= stretch
    return direction * ((stretch - rest) * stiffness)
'''

#-1.0*stiffness * (rest - norm(d1 - d0) * (x0 - x1) / norm(d1 - d0))


#x = sympy.Symbol('x')
#f = sympy.sin(x)
#c = sympy.sin(f)**2 + sympy.cos(x)**2
#x, x1, x2, x3 = sympy.symbols('x x1 x2 x3')
#A = sympy.Matrix([[x+x1, x+x2, x+x3]])
#R = ReferenceFrame('R')
#v = 3*R.x + 4*R.y + 5*R.z


#def elasticSpringEnergy(x0, x1, rest, stiffness):
#    stretch = np.linalg.norm(x1 - x0)
#    return 0.5 * stiffness * ((stretch - rest) * (stretch - rest))
#