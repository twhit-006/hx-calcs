import fluid_properties as fp
import numpy as np
from unit_conversion import unit_conversion as unit_conv
from effectiveness import effectiveness as eff
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

class HeatExchanger:
    def __init__ (self, fluid1, fluid2, configuration, knowns):

        self.fluid1 = fp(fluid1)
        self.fluid2 = fp(fluid2)
        self.configuration = configuration
        self.knowns = [
            {"name": p, "value": v, "units": u} for (p, v, u) in knowns
        ]

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
                    int_value = unit_conv(known["value"], known["units"], int_units).magnitude
                    int_knowns = {"name": known["name"], "value": int_value, "units": int_units}
                    break

                elif param_type not in known["name"] and param_type == list(self.internal_units.keys())[-1]:
                    raise ValueError(f"Unknown parameter type in knowns: {known["name"]}")
            
            self.internal_knowns.append(int_knowns)

        self.knowns_list = [k["name"] for k in self.internal_knowns]

        if len(self.internal_knowns) != 5:
            raise ValueError(f"Exactly 5 known parameters required, {len(self.internal_knowns)} provided: {self.knowns_list}")
        
        if "UA" in self.knowns_list:
            self.solver = "e-NTU"
        else:
            self.solver = "LMTD"

        if self.solver == "e-NTU":
            hot_stream_parameters = ["q","T_H_i", "T_H_o", "m_dot_H"]
            cold_stream_parameters = ["q","T_C_i", "T_C_o", "m_dot_C"]
            unknowns = [p for p in self.HX_parameters.keys() if p not in self.knowns_list]
            if not any(param in unknowns for param in hot_stream_parameters) or not any(param in unknowns for param in cold_stream_parameters):
                raise ValueError("Each stream equation must have at least one unknown.")


    def solve(self):
        if self.solver == "e-NTU":

        else:

if __name__ == "__main__":
    primary_fluid = "pg25"
    secondary_fluid = "pg25"
    configuration = "counterflow"
    
    knowns = [
        ("T_H_i", 50, ureg.degC),
        ("T_C_i", 20, ureg.degC),
        ("T_C_o", 30, ureg.degC),
        ("UA", 2, ureg.watt / ureg.kelvin),
        ("q", 250, ureg.kilowatt)
    ]
    
    hx1 = HeatExchanger(primary_fluid, secondary_fluid, configuration, knowns)
    #print(hx1.internal_knowns)
    #print(hx1.solver)