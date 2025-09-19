import fluid_properties as fp
from property_conversion import property_conversion as unit_conv
import pint

class e_NTU_HX:
    def __init__ (self, fluid1, fluid2, knowns):

        self.ureg = pint.UnitRegistry()
        self.Q_ = self.ureg.Quantity

        self.fluid1 = fluid1
        self.fluid2 = fluid2
        self.knowns = [
            {"name": p, "value": v, "units": u} for (p, v, u) in knowns
        ]

        self.internal_units = {
            "T": "K",
            "q": "W",
            "UA": "W/K",
            "m_dot": "kg/s"
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
        
        self.internal_knowns = []

        for known in self.knowns:
            
            for param_type in self.internal_units.keys():
                
                if param_type in known["name"]:
                    
                    int_units = self.internal_units[param_type]
                    int_value = unit_conv(known["value"], known["units"], int_units).magnitude
                    int_knowns = {"name": known["name"], "value": int_value, "units": int_units}
            
            self.internal_knowns.append(int_knowns)
