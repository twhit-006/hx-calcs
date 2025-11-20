import fluid_properties as fp
import numpy as np
from unit_conversion import unit_conversion as unit_conv
from effectiveness import effectiveness as eff
import pint
from sympy import symbols, Eq, solve

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

class HeatExchanger:
    def __init__ (self, cold_fluid, hot_fluid, configuration, knowns, cold_quality=None, hot_quality=None):

        self.cold_fluid = fp.fluid(cold_fluid)
        self.hot_fluid = fp.fluid(hot_fluid)
        self.configuration = configuration
        self.knowns = {p: Q_(v, u) for (p, v, u) in knowns}
        self.quality = {}
        self.hfg = {}
        self.cp = {}

        if self.hot_fluid.name in ("1233", "515") and self.cold_fluid.name in ("1233", "515"):
            raise ValueError("Use refrigerant to refrigerant heat exchanger solver.")

        if cold_quality is not None and self.cold_fluid.name in ("1233", "515"):
            self.quality["cold"] = Q_(cold_quality,ureg.dimensionless)
        elif cold_quality is not None and self.cold_fluid.name not in ("1233", "515"):
            raise ValueError("Cold fluid quality should only be specified for refrigerants.")
        elif cold_quality is None and self.cold_fluid.name in ("1233", "515"):
            raise ValueError("Cold fluid quality must be specified for refrigerants.")
            
        if hot_quality is not None and self.hot_fluid.name in ("1233", "515"):
            self.quality["hot"] = Q_(hot_quality,ureg.dimensionless)
        elif hot_quality is not None and self.hot_fluid.name not in ("1233", "515"):
            raise ValueError("Hot fluid quality should only be specified for refrigerants.")
        elif hot_quality is None and self.hot_fluid.name in ("1233", "515"):
            raise ValueError("Hot fluid quality must be specified for refrigerants.")

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
        self.internal_knowns.update(solved)

        # Keep only the still-unsolved ones
        self.unknowns = unsolved
        
    # Convert supplied knowns to internal units
    def unit_conv_knowns(self):
        self.internal_knowns = {}
        for name, qty in self.knowns.items():
            for param_type, int_units in self.internal_units.items():
                if param_type in name:
                    int_qty = unit_conv(qty.magnitude, qty.units, int_units).magnitude
                    self.internal_knowns[name] = Q_(int_qty, int_units)
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
                        self.hfg["cold"] = Q_(self.cold_fluid.get_properties("T", self.internal_knowns["T_C_o"].magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.cold_fluid.get_properties("T", self.internal_knowns["T_C_o"].magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                        self.unknowns["q"] = self.internal_knowns["m_dot_C"] * self.quality["cold"] * self.hfg["cold"]

                    # Single phase cold stream
                    else:
                        self.cp["cold"] = Q_(self.cold_fluid.get_properties("T", (self.internal_knowns["T_C_i"].magnitude+self.internal_knowns["T_C_o"].magnitude)/2, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        self.unknowns["q"] = self.internal_knowns["m_dot_C"] * self.cp["cold"] * (self.internal_knowns["T_C_o"] - self.internal_knowns["T_C_i"])
                
                # Unknown mass flow rate
                elif "m_dot_C" in self.cold_unknowns.keys():
                    
                    # Two phase cold stream
                    if self.cold_fluid.name in ("1233", "515"):
                        self.hfg["cold"] = Q_(self.cold_fluid.get_properties("T", self.internal_knowns["T_C_o"].magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.cold_fluid.get_properties("T", self.internal_knowns["T_C_o"].magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                        self.unknowns["m_dot_C"] = self.internal_knowns["q"] / (self.quality["cold"] * self.hfg["cold"])
                    
                    # Single phase cold stream
                    else:
                        self.cp["cold"] = Q_(self.cold_fluid.get_properties("T", (self.internal_knowns["T_C_i"].magnitude+self.internal_knowns["T_C_o"].magnitude)/2, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        self.unknowns["m_dot_C"] = self.internal_knowns["q"] / (self.cp["cold"] * (self.internal_knowns["T_C_o"] - self.internal_knowns["T_C_i"]))

                # Unknown outlet temperature
                elif "T_C_o" in self.cold_unknowns.keys():
                    
                    # Two phase cold stream
                    if self.cold_fluid.name in ("1233", "515"):
                        self.unknowns["T_C_o"] = self.internal_knowns["T_C_i"]

                    # Single phase cold stream
                    else:
                        self.cp["cold"] = Q_(self.cold_fluid.get_properties("T", self.internal_knowns["T_C_i"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        self.unknowns["T_C_o"] = self.internal_knowns["T_C_i"] + self.internal_knowns["q"] / (self.internal_knowns["m_dot_C"] * self.cp["cold"])
                        cpcheck = Q_(self.cold_fluid.get_properties("T", self.internal_knowns["T_C_o"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        if abs((cpcheck - self.cp["cold"])/self.cp["cold"]) > 0.05:
                            print(f"Warning: Significant change in cold stream specific heat capacity detected. Relative change: {100*abs((cpcheck - self.cp["cold"])/self.cp["cold"]):.2f}%")

                # Unknown inlet temperature
                elif "T_C_i" in self.cold_unknowns.keys():
                    
                    # Two phase cold stream
                    if self.cold_fluid.name in ("1233", "515"):
                        self.unknowns["T_C_i"] = self.internal_knowns["T_C_o"]

                    # Single phase cold stream
                    else:
                        self.cp["cold"] = Q_(self.cold_fluid.get_properties("T", self.internal_knowns["T_C_o"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        self.unknowns["T_C_i"] = self.internal_knowns["T_C_o"] - self.internal_knowns["q"] / (self.internal_knowns["m_dot_C"] * self.cp["cold"])
                        cpcheck = Q_(self.cold_fluid.get_properties("T", self.internal_knowns["T_C_i"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        if abs((cpcheck - self.cp["cold"])/self.cp["cold"]) > 0.05:
                            print(f"Warning: Significant change in cold stream specific heat capacity detected. Relative change: {100*abs((cpcheck - self.cp["cold"])/self.cp["cold"]):.2f}%")

            # Solve hot stream equation
            if len(self.hot_unknowns) == 1:
                self.solveorder += 2
                
                # Unknown heat rate
                if "q" in self.hot_unknowns.keys():
                    
                    # Two phase hot stream
                    if self.hot_fluid.name in ("1233", "515"):
                        self.hfg["hot"] = Q_(self.hot_fluid.get_properties("T", self.internal_knowns["T_H_i"].magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.hot_fluid.get_properties("T", self.internal_knowns["T_H_i"].magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                        self.unknowns["q"] = self.internal_knowns["m_dot_H"] * self.quality["hot"] * self.hfg["hot"]

                    # Single phase hot stream
                    else:
                        self.cp["hot"] = Q_(self.hot_fluid.get_properties("T", (self.internal_knowns["T_H_i"].magnitude+self.internal_knowns["T_H_o"].magnitude)/2, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        self.unknowns["q"] = self.internal_knowns["m_dot_H"] * self.cp["hot"] * (self.internal_knowns["T_H_i"] - self.internal_knowns["T_H_o"])
                
                # Unknown mass flow rate
                elif "m_dot_H" in self.hot_unknowns.keys():
                    
                    # Two phase hot stream
                    if self.hot_fluid.name in ("1233", "515"):
                        self.hfg["hot"] = Q_(self.hot_fluid.get_properties("T", self.internal_knowns["T_H_i"].magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.hot_fluid.get_properties("T", self.internal_knowns["T_H_i"].magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                        self.unknowns["m_dot_H"] = self.internal_knowns["q"] / (self.quality["hot"] * self.hfg["hot"])
                    
                    # Single phase hot stream
                    else:
                        self.cp["hot"] = Q_(self.hot_fluid.get_properties("T", (self.internal_knowns["T_H_i"].magnitude+self.internal_knowns["T_H_o"].magnitude)/2, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        self.unknowns["m_dot_H"] = self.internal_knowns["q"] / (self.cp["hot"] * (self.internal_knowns["T_H_i"] - self.internal_knowns["T_H_o"]))

                # Unknown outlet temperature
                elif "T_H_o" in self.hot_unknowns.keys():
                    
                    # Two phase hot stream
                    if self.hot_fluid.name in ("1233", "515"):
                        self.unknowns["T_H_o"] = self.internal_knowns["T_H_i"]

                    # Single phase hot stream
                    else:
                        self.cp["hot"] = Q_(self.hot_fluid.get_properties("T", self.internal_knowns["T_H_i"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        self.unknowns["T_H_o"] = self.internal_knowns["T_H_i"] - self.internal_knowns["q"] / (self.internal_knowns["m_dot_H"] * self.cp["hot"])
                        cpcheck = Q_(self.hot_fluid.get_properties("T", self.unknowns["T_H_o"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        if abs((cpcheck - self.cp["hot"])/self.cp["hot"]) > 0.05:
                            print(f"Warning: Significant change in hot stream specific heat capacity detected. Relative change: {100*abs((cpcheck - self.cp["hot"])/self.cp["hot"]):.2f}%")

                # Unknown inlet temperature
                elif "T_H_i" in self.hot_unknowns.keys():
                    
                    # Two phase hot stream
                    if self.hot_fluid.name in ("1233", "515"):
                        self.unknowns["T_H_i"] = self.internal_knowns["T_H_o"]

                    # Single phase hot stream
                    else:
                        self.cp["hot"] = Q_(self.hot_fluid.get_properties("T", self.internal_knowns["T_H_o"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        self.unknowns["T_H_i"] = self.internal_knowns["T_H_o"] + self.internal_knowns["q"] / (self.internal_knowns["m_dot_H"] * self.cp["hot"])
                        cpcheck = Q_(self.hot_fluid.get_properties("T", self.internal_knowns["T_H_i"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                        if abs((cpcheck - self.cp["hot"])/self.cp["hot"]) > 0.05:
                            print(f"Warning: Significant change in hot stream specific heat capacity detected. Relative change: {100*abs((cpcheck - self.cp["hot"])/self.cp["hot"]):.2f}%")

            self.update_knowns()
            print(f"Solveorder: {self.solveorder}")

            # Solveorder 1 or 2 indicates one stream was solved and the other has two unknowns remaining. Must use effectiveness equation.                
            if self.solveorder in [1,2]:
                self.CHOT = Q_(self.internal_knowns["m_dot_H"] * self.cp["hot"] if self.cold_fluid.name not in ("1233", "515") else 9.999e99, ureg.watt / ureg.kelvin)
                self.CCOLD = Q_(self.internal_knowns["m_dot_C"] * self.cp["cold"] if self.hot_fluid.name not in ("1233", "515") else 9.999e99, ureg.watt / ureg.kelvin)
                self.CMIN = min(self.CHOT, self.CCOLD)
                self.CR = self.CMIN / max(self.CHOT, self.CCOLD)
                self.NTU = self.internal_knowns["UA"] / self.CMIN
                self.epsilon = eff(self.configuration, self.CR, self.NTU)

                if "T_H_i" in self.unknowns:
                    self.unknowns["T_H_i"] = self.internal_knowns["T_C_i"] + self.internal_knowns["q"] / (self.CMIN * self.epsilon)
                    self.cp["hot"] = Q_(self.hot_fluid.get_properties("T", self.unknowns["T_H_i"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin)) if self.hot_fluid.name not in ("1233", "515") else None
                    self.unknowns["T_H_o"] = self.unknowns["T_H_i"] - self.internal_knowns["q"] / (self.internal_knowns["m_dot_H"] * self.cp["hot"]) if self.hot_fluid.name not in ("1233", "515") else self.internal_knowns["T_H_i"]
                elif "T_C_i" in self.unknowns:
                    self.unknowns["T_C_i"] = self.internal_knowns["T_H_i"] - self.internal_knowns["q"] / (self.CMIN * self.epsilon)
                    print(self.unknowns["T_C_i"], self.internal_knowns["T_H_i"], self.internal_knowns["q"], self.CMIN, self.epsilon)
                    self.cp["cold"] = Q_(self.cold_fluid.get_properties("T", self.unknowns["T_C_i"].magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin)) if self.cold_fluid.name not in ("1233", "515") else None
                    self.unknowns["T_C_o"] = self.unknowns["T_C_i"] + self.internal_knowns["q"] / (self.internal_knowns["m_dot_C"] * self.cp["cold"]) if self.cold_fluid.name not in ("1233", "515") else self.internal_knowns["T_C_i"]
                self.update_knowns()
            else:
                raise ValueError("Solver failed due to invalid solveorder. This error should not be reachable.")
            
            if len(self.unknowns) > 0:
                raise ValueError("Unknowns remain after running e-NTU solver. This error should not be reachable. Remaining unknowns:", self.unknowns)
            
            return self.internal_knowns
            
        else:
            raise NotImplementedError("LMTD method not yet implemented.")

if __name__ == "__main__":
    cold_fluid = "pg25"
    hot_fluid = "pg25"
    configuration = "counterflow"
    
    knowns = [
        ("T_H_i", 50, ureg.degC),
        ("m_dot_C", 10, ureg.kilogram / ureg.second),
        ("m_dot_H", 10, ureg.kilogram / ureg.second),
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
