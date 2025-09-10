import CoolProp.CoolProp as CP
import numpy as np
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Canonical property symbols expected by CoolProp
# T[K], P[Pa], H[J/kg], S[J/kg/K], D[kg/m^3], V[m^3/kg], Q[-], Cp[J/kg/K], viscosity[Pa.s], conductivity[W/m/K]
DIMENSIONS = {
    "T": ureg.kelvin,
    "P": ureg.pascal,
    "H": ureg.joule / ureg.kilogram,
    "S": ureg.joule / (ureg.kilogram * ureg.kelvin),
    "D": ureg.kilogram / ureg.meter**3,
    "V": ureg.meter**3 / ureg.kilogram,
    "Q": ureg.dimensionless,
    "Cp": ureg.joule / (ureg.kilogram * ureg.kelvin),
    "viscosity": ureg.pascal * ureg.second,
    "conductivity": ureg.watt / (ureg.meter * ureg.kelvin)
}

# Commonly used fluids and corresponding CoolProp names
FLUIDS = {
    "1233": "R1233zd(E)",
    "515": "R1234ze(E)[0.127]&R227ea[0.873]",
    "PG25": "INCOMP::APG-25%",
    "water": "Water"
}

class fluid:
    def __init__(self, name):
        if name not in FLUIDS:
            self.name = name
            print(f"User provided custom fluid, ensure valid CoolProp name: {name}")
        else:
            self.name = FLUIDS[name]

    # Convert to SI units from user input
    def to_si(prop, value, units):
        nonsi = Q_(value, units)
        si = nonsi.to(DIMENSIONS[prop])
        return si.magnitude
    
    # Convert from SI units to user input
    def from_si(prop, value, units):
        si = Q_(value, DIMENSIONS[prop])
        nonsi = si.to(units)
        return nonsi.magnitude

    # Convert units, reset input properties in object
    def val_set(self, prop1, value1, units1, prop2, value2, units2):
        if prop1 == prop2:
            raise ValueError("Input properties must be different")
        self.prop1 = prop1
        self.val1 = fluid.to_si(prop1, value1, units1)
        self.prop2 = prop2
        self.val2 = fluid.to_si(prop2, value2, units2)

    # Check if state is valid and input properties are appropriate for the given state
    def state_check(self):
        self.phase = CP.PhaseSI(self.prop1, self.val1, self.prop2, self.val2, self.name)
        if self.phase == "unknown":
            raise ValueError("Input properties do not correspond to a valid state")
        if self.phase == "two_phase" and (self.prop1 in ("T", "P")) and (self.prop2 in ("T", "P")):
            raise ValueError("Two independent intensive properties required for two-phase state")

    # Convert units, check state, and update all properties
    def update_properties(self, prop1, value1, units1, prop2, value2, units2):
        self.val_set(prop1, value1, units1, prop2, value2, units2)
        self.state_check()
        self.prop_array = []

        # Get properties from CoolProp
        for prp in DIMENSIONS.keys():
            setattr(self, prp, CP.PropsSI(prp, self.prop1, self.val1, self.prop2, self.val2, self.name))
            self.prop_array.append(getattr(self, prp))

        
if __name__ == "__main__":
    r515 = fluid("PG25")
    r515.update_properties("T", 25, "degC", "P", 150, "psi")
    print(f"Temperature: {r515.T} {DIMENSIONS['T']}")
    print(f"Pressure: {r515.P} {DIMENSIONS['P']}")
    print(f"Thermodynamic state: {r515.phase}")
    print(f"Property array: {r515.prop_array}")