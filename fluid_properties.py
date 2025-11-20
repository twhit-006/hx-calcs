import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import numpy as np
import pint
import os
from unit_conversion import unit_conversion as unit_conv

ureg = pint.UnitRegistry()

class fluid:
    def __init__(self, name):
        ureg = pint.UnitRegistry()

        # Canonical property symbols expected by CoolProp
        # T[K], P[Pa], H[J/kg], S[J/kg/K], D[kg/m^3], Q[-], C[J/kg/K], viscosity[Pa.s],
        # conductivity[W/m/K]
        self.parameters = {
            "T": ureg.kelvin,
            "P": ureg.pascal,
            "H": ureg.joule / ureg.kilogram,
            "S": ureg.joule / (ureg.kilogram * ureg.kelvin),
            "D": ureg.kilogram / ureg.meter**3,
            "Q": ureg.dimensionless,
            "C": ureg.joule / (ureg.kilogram * ureg.kelvin),
            "viscosity": ureg.pascal * ureg.second,
            "conductivity": ureg.watt / (ureg.meter * ureg.kelvin)
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
            try:
                AbstractState("REFPROP", "Water")
            except Exception:
                print("No REFPROP installation found, using custom CoolProp mixture for R515b, properties may differ.")
                self.fluids["515"] = "R1234yf[0.1271]&R227EA[0.8729]"
            else:
                print("REFPROP found. Version:", CP.get_global_param_string("REFPROP_version"))
                try:
                    AbstractState("REFPROP", "R515B-TW.MIX")
                except Exception:
                    print("R515B-TW.MIX missing from REFPROP mixtures directory, using custom mixture for R515b, properties may differ.")
                    self.fluids["515"] = "REFPROP::R1234ZEE[0.1271]&R227EA[0.8729]"
                else:
                    print("R515B-TW.MIX found.")
                    self.fluids["515"] = "REFPROP::R515B-TW.MIX"                
        
        # Allow user to input custom fluid types.
        if name not in self.fluids:
            self.name = name
            print(f"User provided custom fluid, ensure valid CoolProp name: {name}")
        else:
            self.name = self.fluids[name]

    # Convert units, reset input properties in object
    def val_set(self, prop1, value1, units1, prop2, value2, units2):
        if prop1 == prop2:
            raise ValueError("Input properties must be different")
        self.prop1 = prop1
        self.val1 = unit_conv(value1, units1, self.parameters[prop1]).magnitude
        self.prop2 = prop2
        self.val2 = unit_conv(value2, units2, self.parameters[prop2]).magnitude

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
    print(fld.get_properties("T", 40, ureg.celsius, "Q", 0, ureg.dimensionless)["C"])
    print(unit_conv(fld.get_properties("T", 40, ureg.celsius, "Q", 0, ureg.dimensionless)["P"], fld.parameters["P"], ureg.psi).magnitude)