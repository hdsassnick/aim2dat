import numpy as np

from aim2dat.utils.units import constants, energy

_component_data = {
        "water": (647.096, 22.064e6, 0.3449),
        "methane": (190.6, 4e6, 0.008), #TODO doublecheck these values.
    }


class PREOS:
    _omega_a = 0.45724
    _omega_b = 0.07780
    _r = constants.kb * constants.na

    def __init__(self, temperature_c, pressure_c, omega):
        self.temperature_c = temperature_c
        self.pressure_c = pressure_c
        self.omega = omega
        self.attraction_param = self._omega_a * (self._r * temperature_c)**2.0/pressure_c
        self.repulsion_param = self._omega_b * self._r * temperature_c / pressure_c

    @classmethod
    def from_name(cls, name):
        if name in _component_data:
            return cls(*_component_data[name])

    def get_alpha(self, temperature):
        kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2.0 #TODO double-check this correction factor, implement different ones?
        return (1.0 + kappa * (1.0 - np.sqrt(temperature / self.temperature_c)))**2.0

    def get_pressure(self, temperature, volume):
        return (self._r * temperature) / (volume - self.repulsion_param) - (self.attraction_param * self.get_alpha(temperature)) / (volume**2.0 + 2.0 * self.repulsion_param * volume - self.repulsion_param**2.0)

    def get_compressiblity_factors(self, temperature, pressure):
        a = self.get_alpha(temperature) * self.attraction_param * pressure / (self._r * temperature)**2.0
        b = self.repulsion_param * pressure / (self._r * temperature)
        roots = np.roots([1.0, b - 1.0, a -2.0*b -3.0*b**2.0, -a*b + b**2.0 + b**3.0])
        return np.sort([root.real for root in roots if not np.iscomplex(root)])

    def get_fugacity(self, temperature, pressure):
        # TODO check if we can have a common implementation
        a = self.get_alpha(temperature) * self.attraction_param * pressure / (self._r * temperature)**2.0
        b = self.repulsion_param * pressure / (self._r * temperature)
        z = max(self.get_compressiblity_factors(temperature, pressure))
        ln_fc =      z - 1.0 - np.log(z - b) - a / (np.sqrt(8.0) * b) * np.log((z + (np.sqrt(2.0) + 1.0) * b) / (z + (1.0 - np.sqrt(2.0)) * b))
        return np.exp(ln_fc) * pressure











