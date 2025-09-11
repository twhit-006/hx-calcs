import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import numpy as np
import pint
import os

class fluid:
    def __init__(self, name):
        self.ureg = pint.UnitRegistry()
        self.Q_ = self.ureg.Quantity

        # Canonical property symbols expected by CoolProp
        # T[K], P[Pa], H[J/kg], S[J/kg/K], D[kg/m^3], Q[-], C[J/kg/K], viscosity[Pa.s],
        # conductivity[W/m/K]
        self.parameters = {
            "T": self.ureg.kelvin,
            "P": self.ureg.pascal,
            "H": self.ureg.joule / self.ureg.kilogram,
            "S": self.ureg.joule / (self.ureg.kilogram * self.ureg.kelvin),
            "D": self.ureg.kilogram / self.ureg.meter**3,
            "Q": self.ureg.dimensionless,
            "C": self.ureg.joule / (self.ureg.kilogram * self.ureg.kelvin),
            "viscosity": self.ureg.pascal * self.ureg.second,
            "conductivity": self.ureg.watt / (self.ureg.meter * self.ureg.kelvin)
        }

        # Commonly used fluids and corresponding CoolProp names
        self.fluids = {
            "1233": "R1233zd(E)",
            "pg25": "INCOMP::APG-25%",
            "water": "Water"
        }
        
        # R515b needs REFPROP and a custom mixture file to work properly, but approximate
        # properties can be obtained without either. ALWAYS RUN REFPROP FOR ACCURATE RESULTS.
        if name == "515":
            has_refprop = bool(CP.get_global_param_string("REFPROP_version"))
            if has_refprop:
                print("REFPROP found.")
                try:
                    AbstractState("REFPROP", "R515B-TW.MIX")
                except Exception:
                    print("R515B-TW.MIX missing from REFPROP mixtures directory, using custom mixture for R515b, properties may differ.")
                    self.fluids["515"] = "REFPROP::R1234ZEE[0.1271]&R227EA[0.8729]"
                else:
                    print("R515B-TW.MIX found.")
                    self.fluids["515"] = "REFPROP::R515B-TW.MIX"
            else:
                print("No REFPROP installation found, using custom mixture for R515b, properties may differ.")
                self.fluids["515"] = "R1234yf[0.1271]&R227EA[0.8729]"
        
        # Allow user to input custom fluid types.
        if name not in self.fluids:
            self.name = name
            print(f"User provided custom fluid, ensure valid CoolProp name: {name}")
        else:
            self.name = self.fluids[name]

    # Convert to SI units from user input
    def to_si(self, prop, value, units):
        nonsi = self.Q_(value, units)
        si = nonsi.to(self.parameters[prop])
        return si.magnitude
    
    # Convert from SI units to user input
    def from_si(self, prop, value, units):
        si = self.Q_(value, self.parameters[prop])
        nonsi = si.to(units)
        return nonsi.magnitude

    # Convert units, reset input properties in object
    def val_set(self, prop1, value1, units1, prop2, value2, units2):
        if prop1 == prop2:
            raise ValueError("Input properties must be different")
        self.prop1 = prop1
        self.val1 = self.to_si(prop1, value1, units1)
        self.prop2 = prop2
        self.val2 = self.to_si(prop2, value2, units2)

    # Check if state is valid and input properties are appropriate for the given state
    def state_check(self):
        self.phase = CP.PhaseSI(self.prop1, self.val1, self.prop2, self.val2, self.name)
        if self.phase == "unknown":
            raise ValueError("Input properties do not correspond to a valid state")
        if self.phase == "two_phase" and (self.prop1 in ("T", "P")) and (self.prop2 in ("T", "P")):
            raise ValueError("Two independent intensive properties required for two-phase state")

    # Convert units, check state, and update all properties
    def get_properties(self, prop1, value1, units1, prop2, value2, units2):
        self.val_set(prop1, value1, units1, prop2, value2, units2)
        self.state_check()
        self.prop_array = []
        property_dict = {}
        # Get properties from CoolProp
        for prp in self.parameters.keys():
            # Skip quality if not in two-phase region
            if prp == "Q" and self.phase != "two_phase":
                continue
            property_dict[prp] = CP.PropsSI(prp, self.prop1, self.val1, self.prop2, self.val2, self.name)
        return property_dict
        
if __name__ == "__main__":
    fld = fluid("515")
    print(fld.get_properties("T", 40, "degC", "Q", 0, ""))
    print(fld.from_si("P", fld.get_properties("T", 40, "degC", "Q", 0, "")["P"], "psi"))