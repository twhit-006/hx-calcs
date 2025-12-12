import fluid_properties as fp
import numpy as np
from unit_conversion import unit_conversion as unit_conv
from effectiveness import effectiveness as eff
import pint
from sympy import symbols, Eq, solve

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

class HeatExchanger:
    def __init__ (self, sys_input, fac_input, it_input):
        self.hot_name = it_input[0][1]
        self.cold_name = fac_input[0][1]        
        self.hot_fluid = fp.fluid(it_input[0][1])
        self.cold_fluid = fp.fluid(fac_input[0][1])
        self.configuration = "counterflow"
        self.knowns = {
            p: Q_(v, u)
            for (p, v, u) in (fac_input + it_input + sys_input)
            if v is not None and not p.startswith("fluid_")
        }

        self.internal_units = {
            "T": ureg.kelvin,
            "q": ureg.watt,
            "UA": ureg.watt / ureg.kelvin,
            "m_dot": ureg.kilogram / ureg.second,
            "x": ureg.dimensionless
        }

        self.HX_parameters = {
            "q": "",      # Hot fluid inlet pressure
            "T_i_H": "",  # Hot fluid inlet temperature
            "T_o_H": "",  # Hot fluid outlet temperature
            "T_i_C": "",  # Cold fluid inlet temperature
            "T_o_C": "",  # Cold fluid outlet temperature
            "UA": "",     # Overall heat transfer coefficient times area
            "m_dot_H": "", # Hot fluid mass flow rate
            "m_dot_C": "", # Cold fluid mass flow rate
            "x_H": "",    # Hot fluid quality
            "x_C": ""     # Cold fluid quality
        }

        self.unit_conv_knowns()

        self.hot_stream_parameters = ["q","T_i_H", "T_o_H", "m_dot_H", "x_H"]
        self.cold_stream_parameters = ["q","T_i_C", "T_o_C", "m_dot_C", "x_C"]
        self.unknowns = {p: None for p in self.HX_parameters.keys() if p not in self.knowns_list}
        self.cold_unknowns = {p: None for p in self.cold_stream_parameters if p in self.unknowns}
        self.hot_unknowns = {p: None for p in self.hot_stream_parameters if p in self.unknowns}

        # Assign heat exchanger mode based on fluid types and check for appropriate number of knowns
        fluids_2ph = ['515', '1233']
        fluids_1ph = ['water', 'pg25', 'air']

        if self.hot_name in fluids_2ph and self.cold_name in fluids_2ph:
            self.mode = '2P-2P'
        elif self.hot_name in fluids_1ph and self.cold_name in fluids_1ph:
            self.mode = '1P-1P'
            del self.unknowns['x_H']
            del self.unknowns['x_C']
            del self.hot_unknowns['x_H']
            del self.cold_unknowns['x_C']
        elif self.hot_name in fluids_1ph and self.cold_name in fluids_2ph:
            self.mode = '2P-1P'
            del self.hot_unknowns['x_H']
            del self.unknowns['x_H']
        elif self.hot_name in fluids_2ph and self.cold_name in fluids_1ph:
            self.mode = '1P-2P'
            del self.cold_unknowns['x_C']
            del self.unknowns['x_C']
        else:
            raise ValueError("One or both fluids are not recognized. Ensure fluids are among supported types.")
        print(f"Heat exchanger mode set to: {self.mode}")

        # CHECK #1: Exactly 3 unknown parameters
        if len(self.unknowns) != 3:
            raise ValueError(f"Exactly 3 unknown parameters required, {len(self.unknowns)} unkowns determined: {self.unknowns.keys()}")
        
        # CHECK #2: UA known or unknown to determine solving method
        if "UA" in self.knowns_list:
            self.solver = "e-NTU"
        else:
            self.solver = "LMTD"
            raise NotImplementedError("LMTD method not yet implemented. Please provide UA as a known parameter.")

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
            
            # CHECK #5: If x is unknown, it must be the only unknown in its stream equation
            if "x_H" in self.unknowns and (len(list(p for p in self.unknowns if p in self.hot_stream_parameters[1:])) > 0):
                raise ValueError("If x_H is unknown, it must be the only unknown in the hot stream equation.")
            if "x_C" in self.unknowns and (len(list(p for p in self.unknowns if p in self.cold_stream_parameters[1:])) > 0):
                raise ValueError("If x_C is unknown, it must be the only unknown in the cold stream equation.")
            if "x_H" in self.unknowns and "x_C" in self.unknowns and "q" in self.unknowns:
                raise ValueError("Invalid combination of unknowns.")
        
        self.first_stream_solver()
        self.phase_contributions()
        self.stream_discretizer()
        self.heatexchanger_funcs()
        self.second_stream_solver()

    ###############################################
    # Helper Functions
    ###############################################    
    # Update knowns and unknown dictionaries  
    def update_knowns(self):
        # Separate solved and unsolved unknowns (keep the keys!)
        solved = {key: qty for key, qty in self.unknowns.items() if qty is not None}
        unsolved = {key: qty for key, qty in self.unknowns.items() if qty is None}
        unsolved_cold = {key: qty for key, qty in self.cold_unknowns.items() if qty is None}
        unsolved_hot = {key: qty for key, qty in self.hot_unknowns.items() if qty is None}

        # Merge solved unknowns into knowns
        self.internal_knowns.update(solved)

        # Keep only the still-unsolved ones
        self.unknowns = unsolved
        self.cold_unknowns = unsolved_cold
        self.hot_unknowns = unsolved_hot

    # Convert user inputs to internal units
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

    # Stream equation functions
    def stream_funcs(self, fluid, knowns, unknowns):
        # Figure out if this is hot or cold based on keys
        if any(k.endswith("_C") for k in knowns.keys()):
            suffix = "_C"
        else:
            suffix = "_H"

        T_i_key   = f"T_i{suffix}"
        T_o_key   = f"T_o{suffix}"
        m_dot_key = f"m_dot{suffix}"
        x_key     = f"x{suffix}"

        # Grab known values if present
        T_i   = knowns.get(T_i_key)        # Quantity or None
        T_o   = knowns.get(T_o_key)
        m_dot = knowns.get(m_dot_key)
        x_val = knowns.get(x_key, Q_(0, ureg.dimensionless))  # default x=0 for 1P

        # Representative temperature & ΔT
        if T_i is not None and T_o is not None:
            T_props = max(T_i, T_o)
            T_dif   = abs(T_o - T_i)
        else:
            # only one T known → just use it for cp/hfg, ΔT not needed in those branches
            T_props = T_i or T_o
            T_dif   = Q_(0, ureg.kelvin)

        # cp and hfg (note the .magnitude)
        cp_val = fluid.get_properties("T", T_props.magnitude, ureg.kelvin,"Q", 0, ureg.dimensionless)["C"]
        cp = Q_(cp_val, ureg.joule / (ureg.kilogram * ureg.kelvin))

        if any(tag in fluid.name for tag in ("1233", "515", "1234")):
            h_liq = fluid.get_properties("T", T_props.magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"]
            h_vap = fluid.get_properties("T", T_props.magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"]
            hfg   = Q_(h_vap - h_liq, ureg.joule / ureg.kilogram)
        else:
            hfg = Q_(0, ureg.joule / ureg.kilogram)  # single-phase

        # Now branch based on which variable is unknown
        if "q" in unknowns:
            q = m_dot * (cp*T_dif + x_val*hfg)
            unknowns["q"] = q

        elif m_dot_key in unknowns:
            q = knowns["q"]
            unknowns[m_dot_key] = q / (cp*T_dif + x_val*hfg)

        elif x_key in unknowns:
            q = knowns["q"]
            unknowns[x_key] = (q/m_dot - cp*T_dif) / hfg

        elif T_o_key in unknowns:
            q = knowns["q"]
            if suffix == "_C":
                unknowns[T_o_key] = T_i + (q/m_dot - x_val*hfg) / cp
            else:
                unknowns[T_o_key] = T_i - (q/m_dot - x_val*hfg) / cp

        elif T_i_key in unknowns:
            q = knowns["q"]
            if suffix == "_C":
                unknowns[T_i_key] = T_o - (q/m_dot - x_val*hfg) / cp
            else:
                unknowns[T_i_key] = T_o + (q/m_dot - x_val*hfg) / cp

        return unknowns


    # This function solves the initially solveable stream equation
    def first_stream_solver(self):
        if self.solver == "e-NTU":

            # Solve cold stream equation
            if len(self.cold_unknowns) == 1:
                fluid = self.cold_fluid
                knowns = {p for p in self.internal_knowns if ("_C" in p or "q" in p or "UA" in p)}
                unknowns = self.cold_unknowns
            
            # Solve hot stream equation
            elif len(self.hot_unknowns) == 1:
                fluid = self.hot_fluid
                knowns = {p for p in self.internal_knowns if ("_H" in p or "q" in p or "UA" in p)}
                unknowns = self.hot_unknowns

            else:
                raise ValueError("No solveable stream equation found for e-NTU method.")
            
            solved_unknown = self.stream_funcs(fluid, knowns, unknowns)
            self.unknowns.update(solved_unknown)
            self.update_knowns()
                
    def phase_contributions(self):        
        if self.solver == "e-NTU":
            # Determine temperature for property calculations
            if "T_" not in self.cold_unknowns.keys():
                T_props_C = max(self.internal_knowns["T_o_C"], self.internal_knowns["T_i_C"])
            else:
                T_props_C = {T for T in self.internal_knowns if (T.startswith("T_") and T.endswith("_C"))}

            if "T_" not in self.hot_unknowns.keys():
                T_props_H = max(self.internal_knowns["T_o_H"], self.internal_knowns["T_i_H"])
            else:
                T_props_H = {T for T in self.internal_knowns if (T.startswith("T_") and T.endswith("_H"))}
            
            # Fluid specific calculations
            cp_C = Q_(self.cold_fluid.get_properties("T", T_props_C.magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
            cp_H = Q_(self.hot_fluid.get_properties("T", T_props_H.magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
            
            self.CHOT = self.internal_knowns["m_dot_H"] * cp_H
            self.CCOLD = self.internal_knowns["m_dot_C"] * cp_C

            # Determine single phase and two phase contributions            
            if self.mode == "2P-2P":
                hfg_C = Q_(self.cold_fluid.get_properties("T", T_props_C.magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.cold_fluid.get_properties("T", T_props_C.magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                hfg_H = Q_(self.hot_fluid.get_properties("T", T_props_H.magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.hot_fluid.get_properties("T", T_props_H.magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                self.q_2p_C = self.internal_knowns["m_dot_C"] * self.internal_knowns["x_C"] * hfg_C
                self.q_2p_H = self.internal_knowns["m_dot_H"] * self.internal_knowns["x_H"] * hfg_H
            elif self.mode == "2P-1P":
                hfg_C = Q_(self.cold_fluid.get_properties("T", T_props_C.magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.cold_fluid.get_properties("T", T_props_C.magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                self.q_2p_C = self.internal_knowns["m_dot_C"] * self.internal_knowns["x_C"] * hfg_C
                self.q_1p_C = self.internal_knowns["q"] - self.q_2p_C
                self.q_2p_H = Q_(0, ureg.watt)
                self.q_1p_H = self.internal_knowns["q"]
            elif self.mode == "1P-2P":
                hfg_H = Q_(self.hot_fluid.get_properties("T", T_props_H.magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.hot_fluid.get_properties("T", T_props_H.magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                self.q_2p_H = self.internal_knowns["m_dot_H"] * self.internal_knowns["x_H"] * hfg_H
                self.q_1p_H = self.internal_knowns["q"] - self.q_2p_H
                self.q_2p_C = Q_(0, ureg.watt)
                self.q_1p_C = self.internal_knowns["q"]
            else:
                self.q_2p_C = Q_(0, ureg.watt)
                self.q_2p_H = Q_(0, ureg.watt)

            self.q_1p_C = self.internal_knowns["q"] - self.q_2p_C
            self.q_1p_H = self.internal_knowns["q"] - self.q_2p_H

            self.q_1p = min(self.q_1p_C, self.q_1p_H)
            self.q_1p2p = abs(self.q_1p_C - self.q_1p_H)
            self.q_2p = self.internal_knowns["q"] - self.q_1p2p - self.q_1p
                
    def stream_discretizer(self):
        if self.solver == "e-NTU":
            if len(self.cold_unknowns) == 0:
                knowns_1p = self.internal_knowns.copy()
                knowns_1p["q"] = self.q_1p
                knowns_1p["x_C"] = Q_(0, ureg.dimensionless)
                del knowns_1p["T_o_C"]
                unknowns_1p = {"T_o_C_1p": None}
                self.internal_knowns.update(self.stream_funcs(self.cold_fluid,knowns_1p, unknowns_1p))
                self.internal_knowns.update({"T_i_C_1p2p": self.internal_knowns["T_o_C_1p"]})

                if self.q_1p_C < self.q_1p_H:
                    self.internal_knowns.update({"T_o_C_1p2p": self.internal_knowns["T_o_C"]})
                    self.internal_knowns.update({"T_i_C_2p": self.internal_knowns["T_o_C"]})
                else:
                    knowns_1p2p = self.internal_knowns.copy()
                    knowns_1p2p["q"] = self.q_1p2p
                    knowns_1p2p["x_C"] = Q_(0, ureg.dimensionless)
                    knowns_1p2p["T_i_C"] = self.internal_knowns["T_i_C_1p2p"]
                    del knowns_1p2p["T_o_C"]
                    unknowns_1p2p = {"T_o_C_1p2p": None}
                    self.internal_knowns.update(self.stream_funcs(self.cold_fluid,knowns_1p2p, unknowns_1p2p))
                    self.internal_knowns.update({"T_i_C_2p": self.internal_knowns["T_o_C"]})
                    print(f'check for heat rate mismatch in cold stream discretization: T_i_C_2p = {self.internal_knowns["T_i_C_2p"]}, T_o_C_1p2p = {self.internal_knowns["T_o_C_1p2p"]}')

                self.internal_knowns.update({"T_o_C_2p": self.internal_knowns["T_o_C"]})
                self.internal_knowns.update({"T_i_C_1p": self.internal_knowns["T_i_C"]})

            elif len(self.hot_unknowns) == 0:
                knowns_1p = self.internal_knowns.copy()
                knowns_1p["q"] = self.q_1p
                knowns_1p["x_H"] = Q_(0, ureg.dimensionless)
                del knowns_1p["T_i_H"]
                unknowns_1p = {"T_i_H_1p": None}
                self.internal_knowns.update(self.stream_funcs(self.hot_fluid,knowns_1p, unknowns_1p))
                self.internal_knowns.update({"T_o_H_1p2p": self.internal_knowns["T_i_H_1p"]})

                if self.q_1p_H < self.q_1p_C:
                    self.internal_knowns.update({"T_i_H_1p2p": self.internal_knowns["T_i_H"]})
                    self.internal_knowns.update({"T_o_H_2p": self.internal_knowns["T_i_H"]})
                else:
                    knowns_1p2p = self.internal_knowns.copy()
                    knowns_1p2p["q"] = self.q_1p2p
                    knowns_1p2p["x_H"] = Q_(0, ureg.dimensionless)
                    knowns_1p2p["T_o_H"] = self.internal_knowns["T_o_H_1p2p"]
                    del knowns_1p2p["T_i_H"]
                    unknowns_1p2p = {"T_i_H_1p2p": None}
                    self.internal_knowns.update(self.stream_funcs(self.hot_fluid,knowns_1p2p, unknowns_1p2p))
                    self.internal_knowns.update({"T_o_H_2p": self.internal_knowns["T_i_H"]})
                    print(f'check for heat rate mismatch in cold stream discretization: T_i_H_2p = {self.internal_knowns["T_i_H_2p"]}, T_i_H_1p2p = {self.internal_knowns["T_i_H_1p2p"]}')

                self.internal_knowns.update({"T_i_H_2p": self.internal_knowns["T_i_H"]})
                self.internal_knowns.update({"T_o_H_1p": self.internal_knowns["T_o_H"]})

            else:
                raise ValueError("Stream equation must be solved before discretization.")

    def heatexchanger_funcs(self):        
        if self.solver == "e-NTU":
            
            if "T_i_C" in self.unknowns:
                # Single phase portion
                CMIN_1p = min(self.CHOT, self.CCOLD)
                CR_1p = CMIN_1p / max(self.CHOT, self.CCOLD)
                NTU_1p = self.internal_knowns["UA"] / CMIN_1p
                epsilon_1p = eff(self.configuration, CR_1p, NTU_1p)
                T_i_C = self.internal_knowns["T_i_H_1p"] - self.q_1p/(epsilon_1p * CMIN_1p)
            
                # Mixed portion
                if self.mode == "1P-2P":
                    CMIN_1p2p = self.CCOLD
                    CR_1p2p = 0
                    NTU_1p2p = self.internal_knowns["UA"] / CMIN_1p2p
                    epsilon_1p2p = eff(self.configuration, CR_1p2p, NTU_1p2p)
                    T_i_C_1p2p = self.internal_knowns["T_i_H_1p2p"] - self.q_1p2p/(epsilon_1p2p * CMIN_1p2p)
                elif self.mode == "2P-1P":
                    CMIN_1p2p = self.CHOT
                    CR_1p2p = 0
                    NTU_1p2p = self.internal_knowns["UA"] / CMIN_1p2p
                    epsilon_1p2p = eff(self.configuration, CR_1p2p, NTU_1p2p)
                    T_i_C_1p2p = self.internal_knowns["T_i_H_1p2p"] - self.q_1p2p/(epsilon_1p2p * CMIN_1p2p)

                # Two phase portion
                T_i_C_2p = self.internal_knowns["T_i_H"] - self.q_2p/self.internal_knowns["UA"]

                self.internal_knowns.update({"T_i_C": T_i_C})
                self.internal_knowns.update({"T_i_C_1p2p": T_i_C_1p2p})
                self.internal_knowns.update({"T_i_C_2p": T_i_C_2p})
                self.internal_knowns.update({"T_o_C_1p2p": T_i_C_2p})
                self.internal_knowns.update({"T_o_C_1p": T_i_C_1p2p})
                
            elif "T_i_H" in self.unknowns:
                # Single phase portion
                CMIN_1p = min(self.CHOT, self.CCOLD)
                CR_1p = CMIN_1p / max(self.CHOT, self.CCOLD)
                NTU_1p = self.internal_knowns["UA"] / CMIN_1p
                epsilon_1p = eff(self.configuration, CR_1p, NTU_1p)
                T_i_H_1p = self.internal_knowns["T_i_C"] + self.q_1p/(epsilon_1p * CMIN_1p)
            
                # Mixed portion
                if self.mode == "1P-2P":
                    CMIN_1p2p = self.CCOLD
                    CR_1p2p = 0
                    NTU_1p2p = self.internal_knowns["UA"] / CMIN_1p2p
                    epsilon_1p2p = eff(self.configuration, CR_1p2p, NTU_1p2p)
                    T_i_H_1p2p = self.internal_knowns["T_i_C_1p2p"] + self.q_1p2p/(epsilon_1p2p * CMIN_1p2p)
                elif self.mode == "2P-1P":
                    CMIN_1p2p = self.CHOT
                    CR_1p2p = 0
                    NTU_1p2p = self.internal_knowns["UA"] / CMIN_1p2p
                    epsilon_1p2p = eff(self.configuration, CR_1p2p, NTU_1p2p)
                    T_i_H_1p2p = self.internal_knowns["T_i_C_1p2p"] - self.q_1p2p/(epsilon_1p2p * CMIN_1p2p)

                # Two phase portion
                T_i_H = self.internal_knowns["T_i_C_2p"] - self.q_2p/self.internal_knowns["UA"]

                self.internal_knowns.update({"T_i_H": T_i_H})
                self.internal_knowns.update({"T_i_H_1p2p": T_i_H_1p2p})
                self.internal_knowns.update({"T_i_H_1p": T_i_H_1p})
                self.internal_knowns.update({"T_o_H_1p2p": T_i_H_1p})
                self.internal_knowns.update({"T_o_H_2p": T_i_H_1p2p})

            self.update_knowns()

    def second_stream_solver(self):
        if self.solver == "e-NTU":            
            if "T_o_C" in self.unknowns:
                cp = Q_(self.cold_fluid.get_properties("T", self.internal_knowns["T_i_C"].magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                
                if self.cold_name in ("1233", "515"):
                    hfg = Q_(self.cold_fluid.get_properties("T", self.internal_knowns["T_i_C"].magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.cold_fluid.get_properties("T", self.internal_knowns["T_i_C"].magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                else:
                    hfg = Q_(0, ureg.joule / ureg.kilogram)
                    self.internal_knowns["x_C"] = Q_(0, ureg.dimensionless)

                self.unknowns["T_o_C"] = self.internal_knowns["T_i_C"] + (self.internal_knowns["q"]/self.internal_knowns["m_dot_C"] - self.internal_knowns["x_C"] * hfg) / cp
            
            elif "T_o_H" in self.unknowns:
                cp = Q_(self.hot_fluid.get_properties("T", self.internal_knowns["T_i_H"].magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["C"], ureg.joule / (ureg.kilogram * ureg.kelvin))
                
                if self.hot_name in ("1233", "515"):
                    hfg = Q_(self.hot_fluid.get_properties("T", self.internal_knowns["T_i_H"].magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"] - self.hot_fluid.get_properties("T", self.internal_knowns["T_i_H"].magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"], ureg.joule / ureg.kilogram)
                else:
                    hfg = Q_(0, ureg.joule / ureg.kilogram)
                    self.internal_knowns["x_H"] = Q_(0, ureg.dimensionless)

                self.unknowns["T_o_H"] = self.internal_knowns["T_i_H"] - (self.internal_knowns["q"]/self.internal_knowns["m_dot_H"] - self.internal_knowns["x_H"] * hfg) / cp

            self.update_knowns()

    def print_results(self):
        print("Results:")
        print("---------")
        print(f"Heat Rate [kW]: ", self.internal_knowns["q"].to(ureg.kilowatt).magnitude)
        print(f"UA [W/K]: ", self.internal_knowns["UA"].to(ureg.watt / ureg.kelvin).magnitude)
        print(f"Hot Inlet Temp [°C]: ", self.internal_knowns["T_i_H"].to(ureg.degC).magnitude)
        print(f"Hot Outlet Temp [°C]: ", self.internal_knowns["T_o_H"].to(ureg.degC).magnitude)
        print(f"Cold Inlet Temp [°C]: ", self.internal_knowns["T_i_C"].to(ureg.degC).magnitude)
        print(f"Cold Outlet Temp [°C]: ", self.internal_knowns["T_o_C"].to(ureg.degC).magnitude)
        print(f"Hot Mass Flow Rate [kg/s]: ", self.internal_knowns["m_dot_H"].to(ureg.kilogram / ureg.second).magnitude)
        print(f"Cold Mass Flow Rate [kg/s]: ", self.internal_knowns["m_dot_C"].to(ureg.kilogram / ureg.second).magnitude)
        if self.cold_name in ("1233", "515"):
            print(f"Cold Quality: ", self.internal_knowns["x_C"].magnitude)
        if self.hot_name in ("1233", "515"):
            print(f"Hot Quality: ", self.internal_knowns["x_H"].magnitude)
        print("---------")




if __name__ == "__main__":
    ################################################################################
    # User Inputs
    ################################################################################
    # System Input Parameters
    sys_input = [("UA",  5,  ureg.watt / ureg.kelvin),
        ("q",  2,  ureg.megawatt)]

    # Facility Side Input Parameters
    fac_input = [("fluid_facility",  "515",  None),
        ("T_i_C",  None,  ureg.degC),
        ("T_o_C",  None,  ureg.degC),
        ("m_dot_C",  None,  ureg.kilogram / ureg.second),
        ("x_C",  None,  None)]

    # IT Side Input Parameters
    it_input = [("fluid_it",  "515",  None),
        ("T_i_H",  None,  ureg.degC),
        ("T_o_H",  None,  ureg.degC),
        ("m_dot_H",  None,  ureg.kilogram / ureg.second),
        ("x_H",  None,  ureg.dimensionless)]
    
    hx = HeatExchanger(sys_input, fac_input, it_input)
    print(hx.knowns)