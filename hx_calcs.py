import fluid_properties as fp
import numpy as np
from unit_conversion import unit_conversion as unit_conv
from effectiveness import effectiveness as eff
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

class HeatExchanger:
    def __init__ (self, cold_fluid, hot_fluid, configuration, knowns, cold_quality=None, hot_quality=None):

        self.cold_fluid = fp.fluid(cold_fluid)
        self.hot_fluid = fp.fluid(hot_fluid)
        self.configuration = configuration
        self.knowns = [
            {"name": p, "quantity": Q_(v, u)} for (p, v, u) in knowns
        ]

        self.cold_quality = cold_quality
        self.hot_quality = hot_quality
        if (self.cold_fluid.name == "1233" or self.cold_fluid.name == "515") and self.cold_quality is None:
            raise ValueError("Cold fluid exit quality must be specified for refrigerants.")
        if (self.hot_fluid.name == "1233" or self.hot_fluid.name == "515") and self.hot_quality is None:
            raise ValueError("Hot fluid inlet quality must be specified for refrigerants.")
        if (self.cold_fluid.name != "1233" and self.cold_fluid.name != "515") and self.cold_quality is not None:
            raise ValueError("Cold fluid quality should only be specified for refrigerants.")
        if (self.hot_fluid.name != "1233" and self.hot_fluid.name != "515") and self.hot_quality is not None:
            raise ValueError("Hot fluid quality should only be specified for refrigerants.")

        self.internal_units = {
            "T": ureg.kelvin,
            "q": ureg.watt,
            "UA": ureg.watt / ureg.kelvin,
            "m_dot": ureg.kilogram / ureg.second,
        }

        self.HX_parameters = {
            "T_H_i": "",  # Hot fluid inlet temperature
            "T_H_o": "",  # Hot fluid outlet temperature
            "T_C_i": "",  # Cold fluid inlet temperature
            "T_C_o": "",  # Cold fluid outlet temperature
            "q": "",      # Hot fluid inlet pressure
            "UA": "",     # Overall heat transfer coefficient times area
            "m_dot_H": "", # Hot fluid mass flow rate
            "m_dot_C": "", # Cold fluid mass flow rate
        }

        # Convert supplied knowns to internal units
        self.internal_knowns = []

        for known in self.knowns:
            
            for param_type in self.internal_units.keys():
                
                if param_type in known["name"]:
                    
                    int_units = self.internal_units[param_type] 
                    int_value = unit_conv(known["quantity"].magnitude, known["quantity"].units, int_units).magnitude
                    int_knowns = {"name": known["name"], "quantity": Q_(int_value, int_units)}
                    break

                elif param_type not in known["name"] and param_type == list(self.internal_units.keys())[-1]:
                    raise ValueError(f"Unknown parameter type in knowns: {known["name"]}")
            
            self.internal_knowns.append(int_knowns)

        self.knowns_list = [k["name"] for k in self.internal_knowns]

        # CHECK #1: Exactly 5 known parameters
        if len(self.internal_knowns) != 5:
            raise ValueError(f"Exactly 5 known parameters required, {len(self.internal_knowns)} provided: {self.knowns_list}")
        
        # CHECK #2: UA known or unknown to determine solving method
        if "UA" in self.knowns_list:
            self.solver = "e-NTU"
        else:
            self.solver = "LMTD"

        if self.solver == "e-NTU":
            self.hot_stream_parameters = ["q","T_H_i", "T_H_o", "m_dot_H"]
            self.cold_stream_parameters = ["q","T_C_i", "T_C_o", "m_dot_C"]
            self.unknowns = [{
                "name": p, "quantity": None} for p in self.HX_parameters.keys() if p not in self.knowns_list
                ]
            self.cold_unknowns = [{
                "name": p, "quantity": None} for p in self.cold_stream_parameters if p in self.unknowns
                ]
            self.hot_unknowns = [{
                "name": p, "quantity": None} for p in self.hot_stream_parameters if p in self.unknowns
                ]

            # CHECK #3: 1 unknown in each stream equation for e-NTU method
            if not any(param in self.unknowns["name"] for param in self.hot_stream_parameters) or not any(param in self.unknowns["name"] for param in self.cold_stream_parameters):
                raise ValueError("Each stream equation must have at least one unknown.")
            
            if "q" in self.unknowns["name"]:
                if not any(param in self.unknowns["name"] for param in self.hot_stream_parameters[1:]) and not any(param in self.unknowns["name"] for param in self.cold_stream_parameters[1:]):
                    raise ValueError("If q is unknown, there must be at least one other unknown in either stream equation.")
            
            # CHECK #4: If m_dot is unknown, it must be the only unknown in its stream equation
            if "m_dot_H" in self.unknowns["name"] and len([p for p in self.unknowns["name"] if p in self.hot_stream_parameters[1:]]) > 1:
                raise ValueError("If m_dot_H is unknown, it must be the only unknown in the hot stream equation.")
            if "m_dot_C" in self.unknowns["name"] and len([p for p in self.unknowns["name"] if p in self.cold_stream_parameters[1:]]) > 1:
                raise ValueError("If m_dot_C is unknown, it must be the only unknown in the cold stream equation.")
            if "m_dot_H" in self.unknowns["name"] and "m_dot_C" in self.unknowns["name"] and "q" in self.unknowns["name"]:
                raise ValueError("Invalid combination of unknowns.")

    def solve(self):
        if self.solver == "e-NTU":
            if self.cold_fluid.name == "1233" or "515":
                if "q" in self.unknowns["name"] and len(self.cold_unknowns["name"]) == 1:
                    self.unknowns["q"] = self.knowns["m_dot_C"] * self.cold_quality * self.cold_hfg
                




            epsilon = eff(self.configuration, self.CR, self.NTU)
        else:

if __name__ == "__main__":
    cold_fluid = "pg25"
    hot_fluid = "pg25"
    configuration = "counterflow"
    
    knowns = [
        ("T_H_i", 50, ureg.degC),
        ("T_C_i", 20, ureg.degC),
        ("T_C_o", 30, ureg.degC),
        ("UA", 2, ureg.watt / ureg.kelvin),
        ("q", 250, ureg.kilowatt)
    ]
    
    hx1 = HeatExchanger(primary_fluid, secondary_fluid, configuration, knowns)
    print(hx1.internal_knowns)
    print(hx1.solver)