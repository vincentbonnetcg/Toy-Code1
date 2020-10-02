"""
@author: Vincent Bonnet
@description : solve a 1D spring - aka damped harmonic oscillator
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

SPRING_STIFFNESS = 1.0  # Newtons per meters
SPRING_DAMPING = 0.2   # Netwons per meters per second
INITIAL_POSITION = 2.0 # particle initial position in meters
MASS = 1.0  # particle mass in kg

# time constant
TIME_START = 0.0  # in Second
TIME_END = 30.0   # in Second
DT = 0.1         # in Second
N = round((TIME_END - TIME_START) / DT)

'''
 Helper Classes (State, Derivate, Particle)
'''
@dataclass
class State:
    x : float = INITIAL_POSITION
    v : float = 0.0

@dataclass
class Derivative:
    dx : float = 0.0 # derivative of x - change in position
    dv : float = 0.0 # derivative of v - change in velocity

    def __add__(self, other):
        result = Derivative();
        result.dx = self.dx + other.dx
        result.dv = self.dv + other.dv
        return result

    def __truediv__(self, other):
        result = Derivative();
        result.dx = self.dx / other
        result.dv = self.dv / other
        return result

@dataclass
class Particle:
    mass = 1.0
    state = State()

'''
 Helper Functions
 Derivative Functions
'''
# internal spring force
def acceleration(state, mass):
    attachement = 0.0
    spring_force = -1.0 * (state.x - attachement) * SPRING_STIFFNESS  # spring force
    spring_force += -1.0 * (state.v * SPRING_DAMPING)  # spring damping
    acceleration = spring_force / mass
    return acceleration

# derivative of x - change in position
def dx(state):
    return state.v

# derivative of v - change in velocity
def dv(state, mass):
    return acceleration(state, mass)

# forward integration - return a new state
def integrate(state, derivative, dt):
    result_state = State()
    result_state.x = state.x + derivative.dx * dt
    result_state.v = state.v + derivative.dv * dt
    return result_state

# compute derivate at the state
def derivate(state, mass):
    result_derivate = Derivative()
    result_derivate.dx = dx(state)
    result_derivate.dv = dv(state, mass)
    return result_derivate

'''
 Integration Functions
'''
def forwardEuler(particle, time, dt):
    k = derivate(particle.state, particle.mass)
    particle.state = integrate(particle.state, k, DT)

def RK2(particle, time, dt):
    s1 = particle.state
    k1 = derivate(s1, particle.mass)
    s2 = integrate(s1, k1, DT * 0.5)
    k2 = derivate(s2, particle.mass)
    particle.state = integrate(particle.state, k2, DT)

def RK4(particle, time, dt):
    s1 = particle.state
    k1 = derivate(s1, particle.mass)
    s2 = integrate(s1, k1, DT * 0.5)
    k2 = derivate(s2, particle.mass)
    s3 = integrate(s1, k2, DT * 0.5)
    k3 = derivate(s3, particle.mass)
    s4 = integrate(s1, k3, DT)
    k4 = derivate(s4, particle.mass)
    particle.state = integrate(particle.state, k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6 , DT)

def semiImplicitEulerV1(particle, time, dt):
    particle.state.v += dv(particle.state, particle.mass) * DT
    particle.state.x += dx(particle.state) * DT

def semiImplicitEulerV2(particle, time, dt):
    particle.state.x += dx(particle.state) * DT
    particle.state.v += dv(particle.state, particle.mass) * DT

def leapFrog(particle, time, dt):
    if (time == TIME_START):
        # compute the velocity at half step which will cause the velocity to be half-step ahead of position
        particle.state.v += dv(particle.state, particle.mass) * DT * 0.5
    particle.state.x += particle.state.v * DT
    particle.state.v += dv(particle.state, particle.mass) * DT

def analyticSolution(particle, time, dt):
    if(SPRING_DAMPING==0.0):
        w = np.sqrt(SPRING_STIFFNESS/particle.mass)
        particle.state.x = (INITIAL_POSITION * np.cos(w * time));
    else:
        w0 = np.sqrt(SPRING_STIFFNESS/particle.mass)
        y = SPRING_DAMPING / (2 * particle.mass)
        w = np.sqrt(w0 * w0 - y * y)
        a = np.exp(-1.0 * y * time)
        particle.state.x = (INITIAL_POSITION * a * np.cos(w * time));

def main():
    integrators = [(forwardEuler,"xkcd:aqua"),
                   (RK2,"xkcd:plum"),
                   (RK4,"xkcd:teal"),
                   (semiImplicitEulerV1,"xkcd:chartreuse"),
                   (semiImplicitEulerV2,"xkcd:olive"),
                   (leapFrog,"xkcd:green"),
                   (analyticSolution,"xkcd:red" )]

    plt.title('Single Damped Harmonic Oscillator')
    plt.xlabel('time(t)')
    plt.ylabel('position(x)')

    # integrators Loop
    for integrator in integrators:
        function = integrator[0]
        plot_colour = integrator[1]

        particle = Particle();

        # initialize time and positions samples
        time_samples = np.zeros(N) # TODO : can be computer with linspace(...))
        position_samples = np.zeros(N)

        # simulation Loop
        time = TIME_START
        for i in range(0,N):
            time_samples[i] = time
            position_samples[i] = particle.state.x
            function(particle, time, DT)
            time += DT

        plt.plot(time_samples, position_samples, color=plot_colour, label=function.__name__)

    # display result
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.show()

if __name__ == '__main__':
    main()
