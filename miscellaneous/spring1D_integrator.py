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
 Classes (State, Derivate, Particle)
'''
@dataclass
class State:
    x = INITIAL_POSITION # position
    v = 0.0 # velocity

@dataclass
class Derivative:
    dx = 0.0 # derivative of x
    dv = 0.0 # derivative of v

@dataclass
class Particle:
    mass = 1.0
    state = State()

'''
  Helper Functions
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
    particle.state = integrate(particle.state, k, dt)

def RK2(particle, time, dt):
    s1 = particle.state
    k1 = derivate(s1, particle.mass)
    s2 = integrate(s1, k1, dt * 0.5)
    k2 = derivate(s2, particle.mass)
    particle.state = integrate(particle.state, k2, dt)

def RK4(particle, time, dt):
    s1 = particle.state
    k1 = derivate(s1, particle.mass)
    s2 = integrate(s1, k1, dt * 0.5)
    k2 = derivate(s2, particle.mass)
    s3 = integrate(s1, k2, dt * 0.5)
    k3 = derivate(s3, particle.mass)
    s4 = integrate(s1, k3, dt)
    k4 = derivate(s4, particle.mass)
    k = Derivative()
    k.dx = k1.dx / 6 + k2.dx / 3 + k3.dx / 3 + k4.dx / 6
    k.dv = k1.dv / 6 + k2.dv / 3 + k3.dv / 3 + k4.dv / 6
    particle.state = integrate(particle.state, k, dt)

def semiImplicitEulerV1(particle, time, dt):
    particle.state.v += dv(particle.state, particle.mass) * dt
    particle.state.x += dx(particle.state) * dt

def semiImplicitEulerV2(particle, time, dt):
    particle.state.x += dx(particle.state) * dt
    particle.state.v += dv(particle.state, particle.mass) * dt

def leapFrog(particle, time, dt):
    if (time == TIME_START):
        # compute the velocity at half step which will cause the velocity to be half-step ahead of position
        particle.state.v += dv(particle.state, particle.mass) * dt * 0.5
    particle.state.x += particle.state.v * dt
    particle.state.v += dv(particle.state, particle.mass) * dt

def analyticSolution(particle, time, dt):
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
    time_samples = np.linspace(TIME_START, TIME_END, num=N, endpoint=False)
    position_samples = np.zeros(N)
    for integrator in integrators:
        function = integrator[0]
        plot_colour = integrator[1]
        particle = Particle();

        for i in range(N):
            position_samples[i] = particle.state.x
            function(particle, time_samples[i], DT)

        plt.plot(time_samples, position_samples, color=plot_colour, label=function.__name__)

    # display result
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.show()

if __name__ == '__main__':
    main()
