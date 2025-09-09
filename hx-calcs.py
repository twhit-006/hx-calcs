import coolprop as cp
import numpy as np


class fluid:
    def __init__(self, t=[], p=[], x=[], h=[], s=[], v=[], rho=[]):
        self.temperature = t
        self.pressure = p
        self.quality = x
        self.enthalpy = h
        self.entropy = s
        self.volume = v
        self.density = rho


    def state(self):
