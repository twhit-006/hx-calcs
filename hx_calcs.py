import fluid_properties as fp
import numpy as np
from unit_conversion import unit_conversion as unit_conv
from effectiveness import effectiveness as eff
import pint
from sympy import symbols, Eq, solve

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

class HeatExchanger:
    def __init__ (self, cold_fluid, hot_fluid, configuration, knowns, cold_quality=None, hot_quality=None, cold_subcool=None, hot_subcool=None):

        self.cold_fluid = fp.fluid(cold_fluid)
        self.hot_fluid = fp.fluid(hot_fluid)
        self.configuration = configuration
        self.knowns = {p: Q_(v, u) for (p, v, u) in knowns}
        self.quality = {}
        self.hfg = {}
        self.cp = {}
        self.subcool = {}

        if cold_quality is not None and self.cold_fluid.name in ("1233", "515"):
            self.quality["cold"] = Q_(cold_quality,ureg.dimensionless)
            if cold_subcool is not None:
                self.subcool["cold"] = Q_(cold_subcool,ureg.kelvin)
            else:
                self.subcool["hot"] = Q_(0,ureg.kelvin)
                print("No cold fluid subcooling specified, assuming saturated fluid inlet.")
        elif cold_quality is not None and self.cold_fluid.name not in ("1233", "515"):
            raise ValueError("Cold fluid quality should only be specified for refrigerants.")
        elif cold_quality is None and self.cold_fluid.name in ("1233", "515"):
            raise ValueError("Cold fluid quality must be specified for refrigerants.")
        elif cold_subcool is not None and self.cold_fluid.name not in ("1233", "515"):
            raise ValueError("Cold fluid subcooling should only be specified for refrigerants.")
            
        if hot_quality is not None and self.hot_fluid.name in ("1233", "515"):
            self.quality["hot"] = Q_(hot_quality,ureg.dimensionless)
            if hot_subcool is not None:
                self.subcool["hot"] = Q_(hot_subcool,ureg.kelvin)
            else:
                self.subcool["hot"] = Q_(0,ureg.kelvin)
                print("No hot fluid subcooling specified, assuming saturated fluid exit.")
        elif hot_quality is not None and self.hot_fluid.name not in ("1233", "515"):
            raise ValueError("Hot fluid quality should only be specified for refrigerants.")
        elif hot_quality is None and self.hot_fluid.name in ("1233", "515"):
            raise ValueError("Hot fluid quality must be specified for refrigerants.")
        elif hot_subcool is not None and self.hot_fluid.name not in ("1233", "515"):
            raise ValueError("Hot fluid subcooling should only be specified for refrigerants.")

        self.internal_units = {
            "T": ureg.kelvin,
            "q": ureg.watt,
            "UA": ureg.watt / ureg.kelvin,
            "m_dot": ureg.kilogram / ureg.second,
        }

        self.HX_parameters = {
            "q": "",      # Hot fluid inlet pressure
            "T_H_i": "",  # Hot fluid inlet temperature
            "T_H_o": "",  # Hot fluid outlet temperature
            "T_C_i": "",  # Cold fluid inlet temperature
            "T_C_o": "",  # Cold fluid outlet temperature
            "UA": "",     # Overall heat transfer coefficient times area
            "m_dot_H": "", # Hot fluid mass flow rate
            "m_dot_C": "", # Cold fluid mass flow rate
        }

        self.unit_conv_knowns()

        self.hot_stream_parameters = ["q","T_H_i", "T_H_o", "m_dot_H"]
        self.cold_stream_parameters = ["q","T_C_i", "T_C_o", "m_dot_C"]
        self.unknowns = {p: None for p in self.HX_parameters.keys() if p not in self.knowns_list}
        self.cold_unknowns = {p: None for p in self.cold_stream_parameters if p in self.unknowns}
        self.hot_unknowns = {p: None for p in self.hot_stream_parameters if p in self.unknowns}

        # CHECK #1: Exactly 5 known parameters
        if len(self.knowns_list) != 5:
            raise ValueError(f"Exactly 5 known parameters required, {len(self.knowns_list)} provided: {self.knowns_list}")
        
        # CHECK #2: UA known or unknown to determine solving method
        if "UA" in self.knowns_list:
            self.solver = "e-NTU"
        else:
            self.solver = "LMTD"

        if self.solver == "e-NTU":

            # CHECK #3: 1 unknown in each stream equation for e-NTU method
            if not any(param in self.unknowns for param in self.hot_stream_parameters) or not any(param in self.unknowns for param in self.cold_stream_parameters):
                raise ValueError("Each stream equation must have at least one unknown.")
            
            if "q" in self.unknowns:
                if not any(param in self.unknowns for param in self.hot_stream_parameters[1:]) and not any(param in self.unknowns for param in self.cold_stream_parameters[1:]):
                    raise ValueError("If q is unknown, there must be at least one other unknown in either stream equation.")
            
            # CHECK #4: If m_dot is unknown, it must be the only unknown in its stream equation
            if "m_dot_H" in self.unknowns and (len(list(p for p in self.unknowns if p in self.hot_stream_parameters[1:])) > 0):
                raise ValueError("If m_dot_H is unknown, it must be the only unknown in the hot stream equation.")
            if "m_dot_C" in self.unknowns and (len(list(p for p in self.unknowns if p in self.cold_stream_parameters[1:])) > 0):
                raise ValueError("If m_dot_C is unknown, it must be the only unknown in the cold stream equation.")
            if "m_dot_H" in self.unknowns and "m_dot_C" in self.unknowns and "q" in self.unknowns:
                raise ValueError("Invalid combination of unknowns.")

    # INTERNAL METHODS
    # Update knowns and unknown dictionaries  
    def update_knowns(self):
        # Separate solved and unsolved unknowns (keep the keys!)
        solved = {key: qty for key, qty in self.unknowns.items() if qty is not None}
        unsolved = {key: qty for key, qty in self.unknowns.items() if qty is None}

        # Merge solved unknowns into knowns
        self.knowns.update(solved)

        # Keep only the still-unsolved ones
        self.unknowns = unsolved
        
    # Convert supplied knowns to internal units
    def unit_conv_knowns(self):
        self.internal_knowns = {}
        for name, qty in self.knowns.items():
            for param_type, int_units in self.internal_units.items():
                if param_type in name:
                    int_qty = unit_conv(qty.magnitude, qty.units, int_units)
                    self.internal_knowns[name] = int_qty
                    break
            else:
                raise ValueError(f"Unknown parameter type in knowns: {name}")
        self.knowns_list = list(self.internal_knowns.keys())

    def solve(self):
        self.solveorder = 0
        
        if self.solver == "e-NTU":

            # Solve cold stream equation
            if len(self.cold_unknowns) == 1:
                self.solveorder += 1

                # Unknown heat rate
                if "q" in self.cold_unknowns.keys():
                    
                    # Two phase cold stream
                    if self.cold_fluid.name in ("1233", "515"):
                        self.hfg["cold"] = self.cold_fluid.get_properties("T", self.knowns["T_C_o"], ureg.celsius, "Q", 1, ureg.dimensionless)["H"] - self.cold_fluid.get_properties("T", self.knowns["T_C_o"], ureg.celsius, "Q", 0, ureg.dimensionless)["H"]
                        self.cp["cold"] = self.cold_fluid.get_properties("T", self.knowns["T_C_i"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["q"] = self.knowns["m_dot_C"] * (self.quality["cold"] * self.hfg["cold"] + self.cp["cold"] * (self.knowns["T_C_o"] - self.knowns["T_C_i"]))

                    # Single phase cold stream
                    else:
                        self.cp["cold"] = self.cold_fluid.get_properties("T", (self.knowns["T_C_i"]+self.knowns["T_C_o"])/2, ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["q"] = self.knowns["m_dot_C"] * self.cp["cold"] * (self.knowns["T_C_o"] - self.knowns["T_C_i"])
                
                # Unknown mass flow rate
                elif "m_dot_C" in self.cold_unknowns.keys():
                    
                    # Two phase cold stream
                    if self.cold_fluid.name in ("1233", "515"):
                        self.hfg["cold"] = self.cold_fluid.get_properties("T", self.knowns["T_C_o"], ureg.celsius, "Q", 1, ureg.dimensionless)["H"] - self.cold_fluid.get_properties("T", self.knowns["T_C_o"], ureg.celsius, "Q", 0, ureg.dimensionless)["H"]
                        self.cp["cold"] = self.cold_fluid.get_properties("T", self.knowns["T_C_i"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["m_dot_C"] = self.knowns["q"] / (self.quality["cold"] * self.hfg["cold"] + self.cp["cold"] * (self.knowns["T_C_o"] - self.knowns["T_C_i"]))
                    
                    # Single phase cold stream
                    else:
                        self.cp["cold"] = self.cold_fluid.get_properties("T", (self.knowns["T_C_i"]+self.knowns["T_C_o"])/2, ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["m_dot_C"] = self.knowns["q"] / (self.cp["cold"] * (self.knowns["T_C_o"] - self.knowns["T_C_i"]))

                # Unknown outlet temperature
                elif "T_C_o" in self.cold_unknowns.keys():
                    
                    # Two phase cold stream
                    if self.cold_fluid.name in ("1233", "515"):
                        self.unknowns["T_C_o"] = self.knowns["T_C_i"] + self.subcool["cold"]

                    # Single phase cold stream
                    else:
                        self.cp["cold"] = self.cold_fluid.get_properties("T", self.knowns["T_C_i"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["T_C_o"] = self.knowns["T_C_i"] + self.knowns["q"] / (self.knowns["m_dot_C"] * self.cp["cold"])

                # Unknown inlet temperature
                elif "T_C_i" in self.cold_unknowns.keys():
                    
                    # Two phase cold stream
                    if self.cold_fluid.name in ("1233", "515"):
                        self.unknowns["T_C_i"] = self.knowns["T_C_o"] - self.subcool["cold"]

                    # Single phase cold stream
                    else:
                        self.cp["cold"] = self.cold_fluid.get_properties("T", self.knowns["T_C_o"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["T_C_i"] = self.knowns["T_C_o"] - self.knowns["q"] / (self.knowns["m_dot_C"] * self.cold_cp)

            # Solve hot stream equation
            if len([d["name"] for d in self.hot_unknowns]) == 1:
                self.solveorder += 2
                
                # Unknown heat rate
                if self.hot_unknowns[0]["name"] == "q":
                    
                    # Two phase hot stream
                    if self.hot_fluid.name == "1233" or "515":
                        self.hot_hfg = self.hot_fluid.get_properties("T", self.knowns["T_H_o"], ureg.celsius, "Q", 1, ureg.dimensionless)["H"] - self.hot_fluid.get_properties("T", self.knowns["T_H_o"], ureg.celsius, "Q", 0, ureg.dimensionless)["H"]
                        self.hot_cp = self.hot_fluid.get_properties("T", self.knowns["T_H_o"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["q"] = self.knowns["m_dot_H"] * (self.hot_quality * self.hot_hfg + self.hot_cp * (self.knowns["T_H_i"] - self.knowns["T_H_o"]))

                    # Single phase hot stream
                    else:
                        self.hot_cp = self.hot_fluid.get_properties("T", (self.knowns["T_H_i"]+self.knowns["T_H_o"])/2, ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["q"] = self.knowns["m_dot_H"] * self.hot_cp * (self.knowns["T_H_i"] - self.knowns["T_H_o"])
                
                # Unknown mass flow rate
                elif self.hot_unknowns[0]["name"] == "m_dot_H":
                    
                    # Two phase hot stream
                    if self.hot_fluid.name == "1233" or "515":
                        self.hot_hfg = self.hot_fluid.get_properties("T", self.knowns["T_H_o"], ureg.celsius, "Q", 1, ureg.dimensionless)["H"] - self.hot_fluid.get_properties("T", self.knowns["T_H_o"], ureg.celsius, "Q", 0, ureg.dimensionless)["H"]
                        self.hot_cp = self.hot_fluid.get_properties("T", self.knowns["T_H_o"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["m_dot_H"] = self.knowns["q"] / (self.hot_quality * self.hot_hfg + self.hot_cp * (self.knowns["T_H_i"] - self.knowns["T_H_o"]))
                    
                    # Single phase hot stream
                    else:
                        self.hot_cp = self.hot_fluid.get_properties("T", (self.knowns["T_H_i"]+self.knowns["T_H_o"])/2, ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["m_dot_H"] = self.knowns["q"] / (self.hot_cp * (self.knowns["T_H_i"] - self.knowns["T_H_o"]))

                # Unknown outlet temperature
                elif self.hot_unknowns[0]["name"] == "T_H_o":
                    
                    # Two phase hot stream
                    if self.hot_fluid.name == "1233" or "515":
                        self.hot_hfg = self.hot_fluid.get_properties("T", self.knowns["T_H_i"], ureg.celsius, "Q", 1, ureg.dimensionless)["H"] - self.hot_fluid.get_properties("T", self.knowns["T_H_i"], ureg.celsius, "Q", 0, ureg.dimensionless)["H"]
                        self.hot_cp = self.hot_fluid.get_properties("T", self.knowns["T_H_i"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["T_H_o"] = self.knowns["T_H_i"] - (self.knowns["q"] / self.knowns["m_dot_H"] - self.hot_quality * self.hot_hfg) / self.hot_cp

                    # Single phase hot stream
                    else:
                        self.hot_cp = self.hot_fluid.get_properties("T", self.knowns["T_H_i"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["T_H_o"] = self.knowns["T_H_i"] - self.knowns["q"] / (self.knowns["m_dot_H"] * self.hot_cp)

                # Unknown inlet temperature
                elif self.hot_unknowns[0]["name"] == "T_H_i":
                    
                    # Two phase hot stream
                    if self.hot_fluid.name == "1233" or "515":
                        self.hot_hfg = self.hot_fluid.get_properties("T", self.knowns["T_H_o"], ureg.celsius, "Q", 1, ureg.dimensionless)["H"] - self.hot_fluid.get_properties("T", self.knowns["T_H_o"], ureg.celsius, "Q", 0, ureg.dimensionless)["H"]
                        self.hot_cp = self.hot_fluid.get_properties("T", self.knowns["T_H_o"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["T_H_i"] = self.knowns["T_H_o"] + (self.knowns["q"] / self.knowns["m_dot_H"] - self.hot_quality * self.hot_hfg) / self.hot_cp

                    # Single phase hot stream
                    else:
                        self.hot_cp = self.hot_fluid.get_properties("T", self.knowns["T_H_i"], ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                        self.unknowns["T_H_i"] = self.knowns["T_H_o"] + self.knowns["q"] / (self.knowns["m_dot_H"] * self.hot_cp)

            self.update_knowns()
            print(f"Solveorder: {self.solveorder}")

            # Solveorder 1 or 2 indicates one stream was solved and the other has two unknowns remaining. Must use effectiveness equation.
            if self.solveorder in [1,2]:
                self.CMIN = min(self.knowns["m_dot_H"] * self.hot_cp, self.knowns["m_dot_C"] * self.cold_cp)
                self.CR = self.CMIN / max(self.knowns["m_dot_H"] * self.hot_cp, self.knowns["m_dot_C"] * self.cold_cp)
                self.NTU = self.knowns["UA"] / self.CMIN
                self.epsilon = eff(self.configuration, self.CR, self.NTU)

                if "T_H_i" in self.unknowns:
                    self.unknowns["T_H_i"] = self.knowns["T_C_i"] + self.knowns["q"] / (self.CMIN * self.epsilon)
                    self.unknowns["T_H_o"] = self.unknowns["T_H_i"] - self.knowns["q"] / (self.knowns["m_dot_H"] * self.hot_cp)
                    self.update_knowns()
                elif "T_C_i" in self.unknowns:
                    self.unknowns["T_C_i"] = self.knowns["T_H_i"] - self.knowns["q"] / (self.CMIN * self.epsilon)
                    self.unknowns["T_C_o"] = self.unknowns["T_C_i"] + self.knowns["q"] / (self.knowns["m_dot_C"] * self.cold_cp)
                    self.update_knowns()
            else:
                raise ValueError("Solver failed due to invalid solveorder. This error should not be reachable.")
            
            if len(self.unknowns) > 0:
                raise ValueError("Unknowns remain after running e-NTU solver. This error should not be reachable. Remaining unknowns:", self.unknowns)
            
            return self.knowns
            
        else:
            raise NotImplementedError("LMTD method not yet implemented.")

if __name__ == "__main__":
    cold_fluid = "pg25"
    hot_fluid = "pg25"
    configuration = "counterflow"
    
    knowns = [
        ("T_H_i", 50, ureg.degC),
        ("m_dot_C", 0.1, ureg.kilogram / ureg.second),
        ("m_dot_H", 0.1, ureg.kilogram / ureg.second),
        #("T_C_i", 20, ureg.degC),
        #("T_C_o", 30, ureg.degC),
        ("UA", 5, ureg.watt / ureg.kelvin),
        ("q", 250, ureg.kilowatt)
    ]
    
    hx1 = HeatExchanger(cold_fluid, hot_fluid, configuration, knowns)
    #print(hx1.internal_knowns)
    #print(hx1.solver)
    results = hx1.solve()
    print(f"Results: {results}")
