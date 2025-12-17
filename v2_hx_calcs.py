"""Heat exchanger sizing utility leveraging e-NTU relationships."""

import fluid_properties as fp
import numpy as np
from unit_conversion import unit_conversion as unit_conv
from effectiveness import effectiveness as eff
import pint
from sympy import symbols, Eq, solve

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

TWO_PHASE_FLUIDS = {"1233", "515", "1234"}


def is_two_phase_label(label: str) -> bool:
    """Return True when the provided fluid label represents a two-phase fluid."""
    return label in TWO_PHASE_FLUIDS


class HeatExchanger:
    """Encapsulates heat exchanger calculations for both 1P and 2P streams."""

    def __init__(self, sys_input, fac_input, it_input):
        """Normalize user inputs, categorize phases, and kick off solving routine."""
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
        self.errors = []
        self.step = "Initial"
        self.knowntemps_C = True if "T_i_C" and "T_o_C" in self.knowns.keys() else False
        self.knowntemps_H = True if "T_i_H" and "T_o_H" in self.knowns.keys() else False

        self.internal_units = {
            "T": ureg.kelvin,
            "q": ureg.watt,
            "UA": ureg.watt / ureg.kelvin,
            "m_dot": ureg.kilogram / ureg.second,
            "x": ureg.dimensionless
        }

        self.HX_parameters = {
            "q": "",       # Heat rate
            "T_i_H": "",   # Hot fluid inlet temperature
            "T_o_H": "",   # Hot fluid outlet temperature
            "T_i_C": "",   # Cold fluid inlet temperature
            "T_o_C": "",   # Cold fluid outlet temperature
            "UA": "",      # Overall heat transfer coefficient times area
            "m_dot_H": "", # Hot fluid mass flow rate
            "m_dot_C": "", # Cold fluid mass flow rate
            "x_H": "",     # Hot fluid quality
            "x_C": ""      # Cold fluid quality
        }

        self.unit_conv_knowns()

        self.hot_stream_parameters = ["q", "T_i_H", "T_o_H", "m_dot_H", "x_H"]
        self.cold_stream_parameters = ["q", "T_i_C", "T_o_C", "m_dot_C", "x_C"]
        self.unknowns = {p: None for p in self.HX_parameters.keys() if p not in self.knowns_list}
        self.cold_unknowns = {p: None for p in self.cold_stream_parameters if p in self.unknowns}
        self.hot_unknowns = {p: None for p in self.hot_stream_parameters if p in self.unknowns}

        # Assign heat exchanger mode based on fluid types and check for appropriate number of knowns.
        fluids_2ph = ['515', '1233']  # reference list maintained for documentation parity
        fluids_1ph = ['water', 'pg25', 'air']

        self.two_phase_hot = is_two_phase_label(self.hot_name)
        self.two_phase_cold = is_two_phase_label(self.cold_name)

        if self.two_phase_cold and self.two_phase_hot:
            self.mode = '2P-2P'
        elif not self.two_phase_cold and not self.two_phase_hot:
            self.mode = '1P-1P'
            del self.unknowns['x_H']
            del self.unknowns['x_C']
            del self.hot_unknowns['x_H']
            del self.cold_unknowns['x_C']
        elif self.two_phase_cold and not self.two_phase_hot:
            self.mode = '2P-1P'
            del self.hot_unknowns['x_H']
            del self.unknowns['x_H']
        elif not self.two_phase_cold and self.two_phase_hot:
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
            if "m_dot_H" in self.unknowns and (len(list(p for p in self.unknowns if p in self.hot_stream_parameters[1:])) > 1):
                raise ValueError(f"If m_dot_H is unknown, it must be the only unknown in the hot stream equation: {self.unknowns}")
            if "m_dot_C" in self.unknowns and (len(list(p for p in self.unknowns if p in self.cold_stream_parameters[1:])) > 1):
                raise ValueError(f"If m_dot_C is unknown, it must be the only unknown in the cold stream equation: {self.unknowns}")
            if "m_dot_H" in self.unknowns and "m_dot_C" in self.unknowns and "q" in self.unknowns:
                raise ValueError("Invalid combination of unknowns.")
            
            # CHECK #5: If x is unknown, it must be the only unknown in its stream equation
            if "x_H" in self.unknowns and (len(list(p for p in self.unknowns if p in self.hot_stream_parameters[1:])) > 1):
                raise ValueError(f"If x_H is unknown, it must be the only unknown in the hot stream equation: {self.unknowns}")
            if "x_C" in self.unknowns and (len(list(p for p in self.unknowns if p in self.cold_stream_parameters[1:])) > 1):
                raise ValueError(f"If x_C is unknown, it must be the only unknown in the cold stream equation: {self.unknowns}")
            if "x_H" in self.unknowns and "x_C" in self.unknowns and "q" in self.unknowns:
                raise ValueError("Invalid combination of unknowns.")
            
            # CHECK #6: m_dot and UA must be > 0
            if "m_dot_H" in self.knowns_list and self.internal_knowns["m_dot_H"].magnitude <= 0:
                raise ValueError("Hot mass flow rate must be greater than zero.")
            if "m_dot_C" in self.knowns_list and self.internal_knowns["m_dot_C"].magnitude <= 0:
                raise ValueError("Cold mass flow rate must be greater than zero.")
            if self.internal_knowns["UA"].magnitude <= 0:
                raise ValueError("UA must be greater than zero.")
        
        # Run the solution workflow in stages so each routine has the data it needs.
        self.step = "First Stream"
        self.first_stream_solver()
        self.step = "Phase Contributions"
        self.phase_contributions()
        self.step = "Stream Discretization"
        self.stream_discretizer()
        self.step = "Heat Exchanger Equations"
        self.heatexchanger_funcs()
        self.step = "Second Stream"
        self.second_stream_solver()
        self.step = "Print"
        self.print_errors()
        self.print_results()

    ###############################################
    # Helper Functions
    ###############################################
    def update_knowns(self):
        """Update known/unknown dictionaries after solving part of a stream."""
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

    def unit_conv_knowns(self):
        """Convert user provided knowns to the internal base units."""
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

    def prop_calcs(self, T_props, fluid):
        # cp and hfg (note the .magnitude)
        if fluid is self.hot_fluid:
            two_phase = self.two_phase_hot
        else:
            two_phase = self.two_phase_cold

        if two_phase:
            cp_val = fluid.get_properties("T", T_props.magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["C"]
            cp = Q_(cp_val, ureg.joule / (ureg.kilogram * ureg.kelvin))
            h_liq = fluid.get_properties("T", T_props.magnitude, ureg.kelvin, "Q", 0, ureg.dimensionless)["H"]
            h_vap = fluid.get_properties("T", T_props.magnitude, ureg.kelvin, "Q", 1, ureg.dimensionless)["H"]
            hfg   = Q_(h_vap - h_liq, ureg.joule / ureg.kilogram)
        else:
            cp_val = fluid.get_properties("T", T_props.magnitude, ureg.kelvin, "P", 101.3, ureg.kilopascal)["C"]
            cp = Q_(cp_val, ureg.joule / (ureg.kilogram * ureg.kelvin))
            hfg = Q_(0, ureg.joule / ureg.kilogram)  # single-phase

        return cp, hfg

    def stream_funcs(self, fluid, knowns, unknowns):
        """Solve a single stream equation for the requested unknown quantity."""
        if len(unknowns) != 1:
            raise TypeError(f"stream_funcs was given an incorrect number of unknowns. unknowns list: {unknowns}")
        
        unknown_keystr = list(unknowns.keys())[0]
        matches = [k for k in knowns if "T_" in k]
        known_keystr = matches[0]

        if "_C" in unknown_keystr:
            if "T_i" in unknown_keystr:
                T_i_key = f"T_i_C{unknown_keystr[5:]}"
                T_o_key = f"T_o_C{known_keystr[5:]}"
            elif "T_o" in unknown_keystr:
                T_i_key = f"T_i_C{known_keystr[5:]}"
                T_o_key = f"T_o_C{unknown_keystr[5:]}"
            else:
                T_i_key = "T_i_C" if "T_i_C_" not in known_keystr else ValueError(f"Solving for variable other than temperature at step: {self.step}. Should not be reachable.")
                T_o_key = "T_o_C" if "T_o_C_" not in known_keystr else ValueError(f"Solving for variable other than temperature at step: {self.step}. Should not be reachable.")
                
            suffix = "_C"
            assert len(matches) == 1 or (self.knowntemps_C if T_i_key == "T_i_C" else False), f"Expected 1 T_ key, found {matches}"
            
        elif "_H" in unknown_keystr:
            if "T_i" in unknown_keystr:
                T_i_key = f"T_i_H{unknown_keystr[5:]}"
                T_o_key = f"T_o_H{known_keystr[5:]}"
            elif "T_o" in unknown_keystr:
                T_i_key = f"T_i_H{known_keystr[5:]}"
                T_o_key = f"T_o_H{unknown_keystr[5:]}"
            else:
                T_i_key = "T_i_H" if "T_i_H_" not in known_keystr else ValueError(f"Solving for variable other than temperature at step: {self.step}. Should not be reachable.")
                T_o_key = "T_o_H" if "T_o_H_" not in known_keystr else ValueError(f"Solving for variable other than temperature at step: {self.step}. Should not be reachable.")
                
            suffix = "_H"
            assert len(matches) == 1 or (self.knowntemps_H if T_i_key == "T_i_H" else False), f"Expected 1 T_ key, found {matches}"
        else:
            raise ValueError(f"Unrecognized key passed to stream_funcs at step: {self.step}. key: {unknown_keystr}")

        m_dot_key = f"m_dot{suffix}"
        x_key = f"x{suffix}"

        # Grab known values if present
        T_i = knowns.get(T_i_key)  # Quantity or None
        T_o = knowns.get(T_o_key)
        m_dot = knowns.get(m_dot_key)
        x_val = knowns.get(x_key, Q_(0, ureg.dimensionless))  # default x=0 for 1P

        # Representative temperature & delta-T for property lookups
        if T_i is not None and T_o is not None:
            T_props = Q_(np.mean([T_i.magnitude, T_o.magnitude]),ureg.kelvin)
            T_dif = abs(T_o - T_i)
        else:
            # only one temperature known -> use it for cp/hfg lookups
            T_props = Q_(T_i or T_o,ureg.kelvin)
            T_dif = Q_(0, ureg.kelvin)

        # Get fluid properties
        cp, hfg = self.prop_calcs(T_props, fluid)

        # Now branch based on which variable is unknown
        if "q" in unknowns:
            q = m_dot * (cp * T_dif + x_val * hfg)
            unknowns["q"] = q

        elif m_dot_key in unknowns:
            q = knowns["q"]
            unknowns[m_dot_key] = q / (cp * T_dif + x_val * hfg)

        elif x_key in unknowns:
            q = knowns["q"]
            unknowns[x_key] = (q / m_dot - cp * T_dif) / hfg

        elif T_o_key in unknowns:
            q = knowns["q"]
            if suffix == "_C":
                unknowns[T_o_key] = T_i + (q / m_dot - x_val * hfg) / cp
            else:
                unknowns[T_o_key] = T_i - (q / m_dot - x_val * hfg) / cp

        elif T_i_key in unknowns:
            q = knowns["q"]
            if suffix == "_C":
                unknowns[T_i_key] = T_o - (q / m_dot - x_val * hfg) / cp
            else:
                unknowns[T_i_key] = T_o + (q / m_dot - x_val * hfg) / cp

        return unknowns


    # This function solves the initially solveable stream equation
    def first_stream_solver(self):
        """Solve whichever stream equation has a single unknown."""
        if self.solver == "e-NTU":

            # Solve cold stream equation
            if len(self.cold_unknowns) == 1:
                fluid = self.cold_fluid
                knowns = {
                    k: v
                    for k, v in self.internal_knowns.items()
                    if ("_C" in k) or (k in ("q", "UA"))
                }
                unknowns = self.cold_unknowns

            elif len(self.hot_unknowns) == 1:
                fluid = self.hot_fluid
                knowns = {
                    k: v
                    for k, v in self.internal_knowns.items()
                    if ("_H" in k) or (k in ("q", "UA"))
                }
                unknowns = self.hot_unknowns

            else:
                raise ValueError("No solveable stream equation found for e-NTU method.")
            
            solved_unknown = self.stream_funcs(fluid, knowns, unknowns)
            self.unknowns.update(solved_unknown)
            self.update_knowns()

    def phase_contributions(self):
        """Break total duty into single-phase and two-phase components."""
        if self.solver == "e-NTU":
            # Determine temperature for property calculations
            if "T_o_C" not in self.cold_unknowns.keys() and "T_i_C" not in self.cold_unknowns.keys():
                T_props_C = Q_(np.mean([self.internal_knowns["T_o_C"].magnitude, self.internal_knowns["T_i_C"].magnitude]),ureg.kelvin)
            elif "T_o_C" in self.cold_unknowns.keys() and "T_i_C" in self.cold_unknowns.keys():
                T_props_C = Q_(np.mean([self.internal_knowns["T_o_H"].magnitude, self.internal_knowns["T_i_H"].magnitude]),ureg.kelvin)
                self.errors.append("phasecont_error_cold")
            else:
                T_props_C = self.internal_knowns["T_i_C"] if "T_i_C" in self.internal_knowns else self.internal_knowns["T_o_C"]

            if "T_o_H" not in self.hot_unknowns.keys() and "T_i_H" not in self.hot_unknowns.keys():
                T_props_H = Q_(np.mean([self.internal_knowns["T_o_H"].magnitude, self.internal_knowns["T_i_H"].magnitude]),ureg.kelvin)
            elif "T_o_H" in self.hot_unknowns.keys() and "T_i_H" in self.hot_unknowns.keys():
                T_props_H = Q_(np.mean([self.internal_knowns["T_o_C"].magnitude, self.internal_knowns["T_i_C"].magnitude]),ureg.kelvin)
                self.errors.append("phasecont_error_hot")
            else:
                T_props_H = self.internal_knowns["T_i_H"] if "T_i_H" in self.internal_knowns else self.internal_knowns["T_o_H"]
            
            # Fluid specific calculations
            cp_C, hfg_C = self.prop_calcs(T_props_C, self.cold_fluid)
            cp_H, hfg_H = self.prop_calcs(T_props_H, self.hot_fluid)
            
            self.CHOT = self.internal_knowns["m_dot_H"] * cp_H
            self.CCOLD = self.internal_knowns["m_dot_C"] * cp_C

            # Determine single phase and two phase contributions            
            if self.mode == "2P-2P":
                self.q_2p_C = self.internal_knowns["m_dot_C"] * self.internal_knowns["x_C"] * hfg_C
                self.q_2p_H = self.internal_knowns["m_dot_H"] * self.internal_knowns["x_H"] * hfg_H
            elif self.mode == "2P-1P":
                self.q_2p_C = self.internal_knowns["m_dot_C"] * self.internal_knowns["x_C"] * hfg_C
                self.q_2p_H = Q_(0, ureg.watt)
            elif self.mode == "1P-2P":
                self.q_2p_H = self.internal_knowns["m_dot_H"] * self.internal_knowns["x_H"] * hfg_H
                self.q_2p_C = Q_(0, ureg.watt)
            else:
                self.q_2p_C = Q_(0, ureg.watt)
                self.q_2p_H = Q_(0, ureg.watt)

            self.q_1p_C = self.internal_knowns["q"] - self.q_2p_C
            self.q_1p_H = self.internal_knowns["q"] - self.q_2p_H

            self.q_1p = Q_(min(self.q_1p_C.magnitude, self.q_1p_H.magnitude), ureg.watt)
            self.q_1p2p = Q_(abs(self.q_1p_C.magnitude - self.q_1p_H.magnitude), ureg.watt)
            self.q_2p = Q_(max(self.q_2p_C.magnitude, self.q_2p_H.magnitude), ureg.watt)

            if self.q_1p < 0:
                raise ValueError(f"Two-phase contribution larger than total heat load. q_2p: {self.q_2p:.0} q: {self.internal_knowns["q"]:.0}")
                
    def stream_discretizer(self):
        """Create intra-stream sections so that each region satisfies e-NTU."""
        if self.solver == "e-NTU":
            # Determine whether the cold or hot side already has a closed energy balance.
            if len(self.cold_unknowns) == 0:
                # Discretizing known cold stream, build knowns array for single phase portion of cold stream.
                knowns_1p = {
                    k: v
                    for k, v in self.internal_knowns.items()
                    if ("T_" not in k) and (("_C" in k) or (k in ("q", "UA")))
                }
                knowns_1p["q"] = self.q_1p
                knowns_1p["x_C"] = Q_(0, ureg.dimensionless)
                knowns_1p["T_i_C"] = self.internal_knowns["T_i_C"]
                unknowns_1p = {"T_o_C_1p": None}
                # Solve the stream equation over the single phase to single phase section.
                self.internal_knowns.update(self.stream_funcs(self.cold_fluid, knowns_1p, unknowns_1p))
                self.internal_knowns.update({"T_i_C_1p2p": self.internal_knowns["T_o_C_1p"]})
                self.internal_knowns.update({"T_o_C_2p": self.internal_knowns["T_o_C"]})
                self.internal_knowns.update({"T_i_C_1p": self.internal_knowns["T_i_C"]})

                if self.mode == "1P-1P":
                    # For a purely single-phase heat exchanger, only 1 heat exchanger segment is necessary.
                    del self.internal_knowns["T_o_C_2p"]
                    del self.internal_knowns["T_i_C_1p2p"]
                    # Calculate absolute error between calculated exit temp and known exit temp.
                    if abs(self.internal_knowns["T_o_C_1p"].magnitude - self.internal_knowns["T_o_C"].magnitude) / self.internal_knowns["T_o_C"].magnitude > 0.05:
                            self.errors.append("streamdisc_error_cold")
                    return

                if self.q_1p_C < self.q_1p_H:
                    # In this condition, cold stream boils in mixed-phase section. Isothermal from T_i_C_1p2p to T_o_C.
                    self.internal_knowns.update({"T_o_C_1p2p": self.internal_knowns["T_o_C"]})
                    # Calculate absolute error between calculated inlet temp and known outlet temp for isothermal section.
                    if abs(self.internal_knowns["T_o_C_1p2p"].magnitude - self.internal_knowns["T_i_C_1p2p"].magnitude) / self.internal_knowns["T_o_C_1p2p"].magnitude > 0.05:
                            self.errors.append("streamdisc_error_cold")
                    
                    if self.mode == "2P-2P":
                        # For a 2P-2P heat exchanger, just add an additional isothermal temperature node.
                        self.internal_knowns.update({"T_i_C_2p": self.internal_knowns["T_o_C"]})

                else:
                    # In this condition, hot stream condenses in mixed-phase section. Solve single phase stream equation for cold stream.
                    knowns_1p2p = {
                        k: v
                        for k, v in self.internal_knowns.items()
                        if ("T_" not in k) and (("_C" in k) or (k in ("q", "UA")))
                    }
                    knowns_1p2p["q"] = self.q_1p2p
                    knowns_1p2p["x_C"] = Q_(0, ureg.dimensionless)
                    knowns_1p2p["T_i_C"] = self.internal_knowns["T_i_C_1p2p"]
                    unknowns_1p2p = {"T_o_C_1p2p": None}
                    self.internal_knowns.update(self.stream_funcs(self.cold_fluid, knowns_1p2p, unknowns_1p2p))
                    
                    if self.mode == "2P-2P":
                        # For a 2P-2P heat exchanger, just add an additional isothermal temperature node.
                        self.internal_knowns.update({"T_i_C_2p": self.internal_knowns["T_o_C"]})
                        # Calculate absolute error between calculated exit temp and known exit temp.
                        if abs(self.internal_knowns["T_o_C"].magnitude - self.internal_knowns["T_o_C_1p2p"].magnitude) / self.internal_knowns["T_o_C"].magnitude > 0.05:
                            self.errors.append("streamdisc_error_cold")

            elif len(self.hot_unknowns) == 0:
                # Mirror the same discretization logic for the hot stream.
                # Discretizing known hot stream, build knowns array for single phase portion of hot stream.
                knowns_1p = {
                    k: v
                    for k, v in self.internal_knowns.items()
                    if ("T_" not in k) and (("_H" in k) or (k in ("q", "UA")))
                }
                knowns_1p["q"] = self.q_1p
                knowns_1p["x_H"] = Q_(0, ureg.dimensionless)
                knowns_1p["T_o_H"] = self.internal_knowns["T_o_H"]
                unknowns_1p = {"T_i_H_1p": None}
                self.internal_knowns.update(self.stream_funcs(self.hot_fluid, knowns_1p, unknowns_1p))
                self.internal_knowns.update({"T_o_H_1p2p": self.internal_knowns["T_i_H_1p"]})
                self.internal_knowns.update({"T_i_H_2p": self.internal_knowns["T_i_H"]})
                self.internal_knowns.update({"T_o_H_1p": self.internal_knowns["T_o_H"]})

                if self.mode == "1P-1P":
                    # For a purely single-phase heat exchanger, only 1 heat exchanger segment is necessary.
                    del self.internal_knowns["T_i_H_2p"]
                    del self.internal_knowns["T_o_H_1p2p"]
                    # Calculate absolute error between calculated exit temp and known exit temp.
                    if abs(self.internal_knowns["T_i_H_1p"].magnitude - self.internal_knowns["T_i_H"].magnitude) / self.internal_knowns["T_i_H"].magnitude > 0.05:
                            self.errors.append("streamdisc_error_hot")
                    return

                if self.q_1p_H < self.q_1p_C:
                    # In this condition, hot stream condenses in mixed-phase section. Isothermal from T_o_H_1p2p to T_i_H.
                    self.internal_knowns.update({"T_i_H_1p2p": self.internal_knowns["T_i_H"]})
                    # Calculate absolute error between known inlet temp and calculated outlet temp for isothermal section.
                    if abs(self.internal_knowns["T_i_H_1p2p"].magnitude - self.internal_knowns["T_o_H_1p2p"].magnitude) / self.internal_knowns["T_i_H_1p2p"].magnitude > 0.05:
                            self.errors.append("streamdisc_error_hot")

                    if self.mode == "2P-2P":
                        # For a 2P-2P heat exchanger, just add an additional isothermal temperature node.
                        self.internal_knowns.update({"T_o_H_2p": self.internal_knowns["T_i_H"]})

                else:
                    # In this condition, cold stream boils in mixed-phase section. Solve single phase stream equation for hot stream.
                    knowns_1p2p = {
                        k: v
                        for k, v in self.internal_knowns.items()
                        if ("T_" not in k) and (("_H" in k) or (k in ("q", "UA")))
                    }
                    knowns_1p2p["q"] = self.q_1p2p
                    knowns_1p2p["x_H"] = Q_(0, ureg.dimensionless)
                    knowns_1p2p["T_o_H"] = self.internal_knowns["T_o_H_1p2p"]
                    unknowns_1p2p = {"T_i_H_1p2p": None}
                    self.internal_knowns.update(self.stream_funcs(self.hot_fluid, knowns_1p2p, unknowns_1p2p))
                    
                    if self.mode == "2P-2P":
                        # For a 2P-2P heat exchanger, just add an additional isothermal temperature node.
                        self.internal_knowns.update({"T_o_H_2p": self.internal_knowns["T_i_H"]})
                        # Calculate absolute error between calculated inlet temp and known inlet temp.
                        if abs(self.internal_knowns["T_i_H"].magnitude - self.internal_knowns["T_i_H_1p2p"].magnitude) / self.internal_knowns["T_i_H"].magnitude > 0.05:
                            self.errors.append("streamdisc_error_hot")

            else:
                raise ValueError("Stream equation must be solved before discretization.")

    def heatexchanger_funcs(self):
        """Apply effectiveness-NTU relationships to solve inlet temperatures."""
        if self.solver == "e-NTU":
            
            if "T_i_C" in self.unknowns:
                if self.mode == "1P-1P":
                    required = ["T_i_H_1p"]
                elif self.mode == "2P-2P":
                    required = ["T_i_H_2p", "T_i_H_1p2p", "T_i_H_1p"]
                else:
                    required = ["T_i_H_1p2p", "T_i_H_1p"]                

                assert all(k in self.internal_knowns.keys() for k in required), \
                    f"Hot inlet temperatures for all sections must be known before calculating cold inlet temperature: {self.internal_knowns[required]}"
                # Single phase portion
                CMIN_1p = min(self.CHOT, self.CCOLD)
                CR_1p = (CMIN_1p / max(self.CHOT, self.CCOLD)).magnitude
                NTU_1p = (self.internal_knowns["UA"] / CMIN_1p).magnitude
                epsilon_1p = eff(self.configuration, CR_1p, NTU_1p)
                T_i_C_1p = self.internal_knowns["T_i_H_1p"] - self.q_1p / (epsilon_1p * CMIN_1p)
            
                # Mixed portion
                if self.q_1p_C > self.q_1p_H:
                    CMIN_1p2p = self.CCOLD
                    CR_1p2p = 0
                    NTU_1p2p = (self.internal_knowns["UA"] / CMIN_1p2p).magnitude
                    epsilon_1p2p = eff(self.configuration, CR_1p2p, NTU_1p2p)
                    T_i_C_1p2p = self.internal_knowns["T_i_H_1p2p"] - self.q_1p2p / (epsilon_1p2p * CMIN_1p2p)
                elif self.q_1p_H > self.q_1p_C:
                    CMIN_1p2p = self.CHOT
                    CR_1p2p = 0
                    NTU_1p2p = (self.internal_knowns["UA"] / CMIN_1p2p).magnitude
                    epsilon_1p2p = eff(self.configuration, CR_1p2p, NTU_1p2p)
                    T_i_C_1p2p = self.internal_knowns["T_i_H_1p2p"] - self.q_1p2p / (epsilon_1p2p * CMIN_1p2p)
                elif self.q_1p2p == 0:
                    T_i_C_1p2p = self.internal_knowns["T_i_H_1p2p"]
                else:
                    raise ValueError(f"Heating load mismatch in mixed segment of HX. q_1p2p: {self.q_1p2p}, q_1p_H:{self.q_1p_H}, q_1p_C:{self.q_1p_C}")

                # Two phase portion
                if self.mode == "2P-2P":
                    T_i_C_2p = self.internal_knowns["T_i_H"] - self.q_2p / self.internal_knowns["UA"]
                    if T_i_C_2p > self.internal_knowns["T_i_H"]:
                        raise ValueError("PINCH ERROR: Calculated cold inlet temperature in two-phase section exceeds hot inlet temperature.")

                self.internal_knowns.update({"T_i_C_1p": T_i_C_1p})
                self.internal_knowns.update({"T_i_C": T_i_C_1p})
                
                if self.mode in ["1P-2P", "2P-1P"]:
                    self.internal_knowns.update({"T_i_C_1p2p": T_i_C_1p2p})
                    self.internal_knowns.update({"T_o_C_1p": T_i_C_1p2p})
                if self.mode == "2P-2P":
                    self.internal_knowns.update({"T_i_C_1p2p": T_i_C_1p2p})
                    self.internal_knowns.update({"T_o_C_1p": T_i_C_1p2p})
                    self.internal_knowns.update({"T_i_C_2p": T_i_C_2p})
                    self.internal_knowns.update({"T_o_C_1p2p": T_i_C_2p})
                
                
            elif "T_i_H" in self.unknowns:
                if self.mode == "1P-1P":
                    required = ["T_i_C_1p"]
                elif self.mode == "2P-2P":
                    required = ["T_i_C_2p", "T_i_C_1p2p", "T_i_C_1p"]
                else:
                    required = ["T_i_C_1p2p", "T_i_C_1p" ]              

                assert all(k in self.internal_knowns.keys() for k in required), \
                    f"Cold inlet temperatures for all sections must be known before calculating hot inlet temperature: {self.internal_knowns[required]}"
                # Single phase portion
                CMIN_1p = min(self.CHOT, self.CCOLD)
                CR_1p = (CMIN_1p / max(self.CHOT, self.CCOLD)).magnitude
                NTU_1p = (self.internal_knowns["UA"] / CMIN_1p).magnitude
                epsilon_1p = eff(self.configuration, CR_1p, NTU_1p)
                T_i_H_1p = self.internal_knowns["T_i_C"] + self.q_1p / (epsilon_1p * CMIN_1p)
            
                # Mixed portion
                if self.q_1p_C > self.q_1p_H:
                    CMIN_1p2p = self.CCOLD
                    CR_1p2p = 0
                    NTU_1p2p = (self.internal_knowns["UA"] / CMIN_1p2p).magnitude
                    epsilon_1p2p = eff(self.configuration, CR_1p2p, NTU_1p2p)
                    T_i_H_1p2p = self.internal_knowns["T_i_C_1p2p"] + self.q_1p2p / (epsilon_1p2p * CMIN_1p2p)
                elif self.q_1p_H > self.q_1p_C:
                    CMIN_1p2p = self.CHOT
                    CR_1p2p = 0
                    NTU_1p2p = (self.internal_knowns["UA"] / CMIN_1p2p).magnitude
                    epsilon_1p2p = eff(self.configuration, CR_1p2p, NTU_1p2p)
                    T_i_H_1p2p = self.internal_knowns["T_i_C_1p2p"] - self.q_1p2p / (epsilon_1p2p * CMIN_1p2p)
                elif self.q_1p2p == 0:
                    T_i_H_1p2p = self.internal_knowns["T_i_C_1p2p"]
                else:
                    raise ValueError(f"Heating load mismatch in mixed segment of HX. q_1p2p: {self.q_1p2p}, q_1p_H:{self.q_1p_H}, q_1p_C:{self.q_1p_C}")

                # Two phase portion
                if self.mode == "2P-2P":
                    T_i_H_2p = self.internal_knowns["T_i_C_2p"] - self.q_2p / self.internal_knowns["UA"]
                    if T_i_H_2p > self.internal_knowns["T_i_C_2p"]:
                        raise ValueError("PINCH ERROR: Calculated hot inlet temperature in two-phase section exceeds cold inlet temperature.")

                self.internal_knowns.update({"T_i_H_1p": T_i_H_1p})

                if self.mode == "1P-1P":
                    self.internal_knowns.update({"T_i_H": T_i_H_1p})
                if self.mode in ["1P-2P", "2P-1P"]:
                    self.internal_knowns.update({"T_i_H_1p2p": T_i_H_1p2p})
                    self.internal_knowns.update({"T_o_H_1p2p": T_i_H_1p})
                    self.internal_knowns.update({"T_i_H": T_i_H_1p2p})                    
                if self.mode == "2P-2P":
                    self.internal_knowns.update({"T_i_H_1p2p": T_i_H_1p2p})
                    self.internal_knowns.update({"T_o_H_1p2p": T_i_H_1p})
                    self.internal_knowns.update({"T_o_H_2p": T_i_H_1p2p})
                    self.internal_knowns.update({"T_o_H_2p": T_i_H_1p2p})
                    self.internal_knowns.update({"T_i_H": T_i_H_2p}) 

            self.update_knowns()

    def second_stream_solver(self):
        """Complete the second stream by back-substituting new temperatures."""
        if self.solver == "e-NTU":
            if "T_o_C" in self.unknowns:
                cp, hfg = self.prop_calcs(self.internal_knowns["T_i_C"], self.cold_fluid)

                if not self.two_phase_cold:
                    hfg = Q_(0, ureg.joule / ureg.kilogram)
                    self.internal_knowns["x_C"] = Q_(0, ureg.dimensionless)

                self.unknowns["T_o_C"] = self.internal_knowns["T_i_C"] + (
                    self.internal_knowns["q"] / self.internal_knowns["m_dot_C"] - self.internal_knowns["x_C"] * hfg
                ) / cp
            
            elif "T_o_H" in self.unknowns:
                cp, hfg = self.prop_calcs(self.internal_knowns["T_i_H"], self.hot_fluid)
                       
                if not self.two_phase_hot:
                    hfg = Q_(0, ureg.joule / ureg.kilogram)
                    self.internal_knowns["x_H"] = Q_(0, ureg.dimensionless)

                self.unknowns["T_o_H"] = self.internal_knowns["T_i_H"] - (
                    self.internal_knowns["q"] / self.internal_knowns["m_dot_H"] - self.internal_knowns["x_H"] * hfg
                ) / cp

            self.update_knowns()

    def print_errors(self):
        if not self.errors:
            print("No errors detected in heat exchanger calculation.")
            return
        
        if "phasecont_error_cold" in self.errors:
            print("WARNING: Unknown temperatures in cold stream during phase contribution calculations. Hot stream temps used for property evals.")
            T_err = Q_(np.mean([self.internal_knowns["T_o_H"].magnitude, self.internal_knowns["T_i_H"].magnitude]),ureg.kelvin)
            T_calc = Q_(np.mean([self.internal_knowns["T_o_C"].magnitude, self.internal_knowns["T_i_C"].magnitude]),ureg.kelvin)
            
            cp_err, hfg_err = self.prop_calcs(T_err, self.cold_fluid)
            cp_calc, hfg_calc = self.prop_calcs(T_calc, self.cold_fluid)

            if self.two_phase_cold:
                hfg_rel_error = abs((hfg_err.magnitude - hfg_calc.magnitude)/hfg_calc.magnitude)
                print(f"Cold stream hfg error: {hfg_rel_error*100:.2f}%")

            cp_rel_error = abs((cp_err.magnitude - cp_calc.magnitude)/cp_calc.magnitude)
            print(f"Cold stream cp error: {cp_rel_error*100:.2f}%")

        if "phasecont_error_hot" in self.errors:
            print("WARNING: Unknown temperatures in hot stream during phase contribution calculations. Cold stream temps used for property evals.")
            T_err = Q_(np.mean([self.internal_knowns["T_o_C"].magnitude, self.internal_knowns["T_i_C"].magnitude]),ureg.kelvin)
            T_calc = Q_(np.mean([self.internal_knowns["T_o_H"].magnitude, self.internal_knowns["T_i_H"].magnitude]),ureg.kelvin)

        cp_err, hfg_err  = self.prop_calcs(T_err, self.hot_fluid)
        cp_calc, hfg_calc  = self.prop_calcs(T_calc, self.hot_fluid)

        if self.two_phase_hot:
            hfg_rel_error = abs((hfg_err.magnitude - hfg_calc.magnitude)/hfg_calc.magnitude)
            print(f"Hot stream hfg error: {hfg_rel_error*100:.2f}%")

        cp_rel_error = abs((cp_err.magnitude - cp_calc.magnitude)/cp_calc.magnitude)
        print(f"Hot stream cp error: {cp_rel_error*100:.2f}%")

        if "streamdisc_error_cold" in self.errors:
            print(f"WARNING: Large error in calculated temperatures between discretized sections of cold stream. T_i_C_2p: {self.internal_knowns['T_i_C_2p']}, T_o_C_1p2p: {self.internal_knowns['T_o_C_1p2p']}, T_i_C_1p2p: {self.internal_knowns['T_i_C_1p2p']}")
        
        if "streamdisc_error_hot" in self.errors:
            print(f"WARNING: Large error in calculated temperatures between discretized sections of hot stream. T_o_H_2p: {self.internal_knowns['T_o_H_2p']}, T_i_H_1p2p: {self.internal_knowns['T_i_H_1p2p']}, T_o_H_1p2p: {self.internal_knowns['T_o_H_1p2p']}")

    def print_results(self):
        """Pretty print the solved state for quick inspection."""
        print("Results:")
        print("---------")
        print(f"Heat Rate [kW]: {self.internal_knowns['q'].to(ureg.kilowatt).magnitude:.2f}")
        print(f"UA [W/K]: {self.internal_knowns['UA'].to(ureg.watt / ureg.kelvin).magnitude:.2f}")
        print(f"Hot Inlet Temp [°C]: {self.internal_knowns['T_i_H'].to(ureg.degC).magnitude:.2f}")
        print(f"Hot Outlet Temp [°C]: {self.internal_knowns['T_o_H'].to(ureg.degC).magnitude:.2f}")
        print(f"Cold Inlet Temp [°C]: {self.internal_knowns['T_i_C'].to(ureg.degC).magnitude:.2f}")
        print(f"Cold Outlet Temp [°C]: {self.internal_knowns['T_o_C'].to(ureg.degC).magnitude:.2f}")
        print(f"Hot Mass Flow Rate [kg/s]: {self.internal_knowns['m_dot_H'].to(ureg.kilogram / ureg.second).magnitude:.2f}")
        print(f"Cold Mass Flow Rate [kg/s]: {self.internal_knowns['m_dot_C'].to(ureg.kilogram / ureg.second).magnitude:.2f}")
        if self.two_phase_cold:
            print(f"Cold Quality: {self.internal_knowns['x_C'].magnitude:.2f}")
        if self.two_phase_hot:
            print(f"Hot Quality: {self.internal_knowns['x_H'].magnitude:.2f}")
        print("---------")


if __name__ == "__main__":
    ################################################################################
    # User Inputs
    ################################################################################
    # System Input Parameters
    sys_input = [
        ("UA", 55000, ureg.watt / ureg.kelvin),
        ("q", 250, ureg.kilowatt),
    ]

    # Facility Side Input Parameters
    fac_input = [
        ("fluid_facility", "pg25", None),
        ("T_i_C", None, ureg.degC),
        ("T_o_C", None, ureg.degC),
        ("m_dot_C", 6.25, ureg.kilogram / ureg.second),
        ("x_C", None, ureg.dimensionless),
    ]

    # IT Side Input Parameters
    it_input = [
        ("fluid_it", "515", None),
        ("T_i_H", 35, ureg.degC),
        ("T_o_H", 30, ureg.degC),
        ("m_dot_H", None, ureg.kilogram / ureg.second),
        ("x_H", 0.7, ureg.dimensionless),
    ]
    
    # Instantiate and display knowns for quick manual verification.
    hx = HeatExchanger(sys_input, fac_input, it_input)
