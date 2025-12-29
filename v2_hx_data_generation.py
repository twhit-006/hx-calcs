from v2_hx_calcs import HeatExchanger, ureg
import fluid_properties as fp
import csv
import os
import numpy as np

Q_ = ureg.Quantity

################################################################################
# User Inputs
################################################################################

# Fluid options and operating points to sweep.
#FAC_FLUIDS = ("air", "pg25", "515")
#IT_FLUIDS = ("pg25", "515")
#POWER = (100, 250, 500, 1000)
#T_INLET = (10, 20, 30, 40, 50)

# Optional single input test
FAC_FLUIDS = ("air",)
IT_FLUIDS = ("515",)
POWER = (127.2,)
T_INLET = (31.4,)

TWO_PHASE_NAMES = {"515"}
COLD_QUALITY = 0.7
HOT_QUALITY = 0.7
T_APPROACH = 1.2 # C. Desired approach temperature between IT and Facility.

# Flow rate control. SET EITHER DT OR MDOT. Defaults to mdot if both are set.
DT_1P = 21.2 # C. Allowed temperature rise of a single phase stream.
DT_2P = 0 # C. Allowed temperature rise of a two phase stream. Equal to inlet subcooling.
#m_dot_fac = Q_(357.6 * (1025/60000), ureg.kilogram / ureg.second) # kg/s. Faility mass flow rate
#m_dot_it = Q_(132 * (1180/60000), ureg.kilogram / ureg.second) # kg/s. Faility mass flow rate

# When True, skip the prompt and always save results to CSV.
AUTO_SAVE = False

################################################################################

def two_phase_check(label: str) -> bool:
    """Return True when the provided fluid label represents a two-phase fluid."""
    return label in TWO_PHASE_NAMES

# System Input Parameters
sys_input = [
    ("UA", 10000, ureg.watt / ureg.kelvin),
    ("q", None, ureg.kilowatt),
]

# Facility Side Input Parameters
fac_input = [
    ("fluid_facility", None, None),
    ("T_i_C", None, ureg.degC),
    ("T_o_C", None, ureg.degC),
    ("m_dot_C", None, ureg.kilogram / ureg.second),
    ("x_C", COLD_QUALITY, ureg.dimensionless),
]

# IT Side Input Parameters
it_input = [
    ("fluid_it", None, None),
    ("T_i_H", None, ureg.degC),
    ("T_o_H", None, ureg.degC),
    ("m_dot_H", None, ureg.kilogram / ureg.second),
    ("x_H", HOT_QUALITY, ureg.dimensionless),
]

# Sweep all combinations of facility/IT fluids, loads, and inlet temperatures.
for fac_f in FAC_FLUIDS:
    for it_f in IT_FLUIDS:
        for heat_rate in POWER:
            for T_in in T_INLET:
                sys_input[0] = ("UA", 50000, ureg.watt / ureg.kelvin)
                sys_input[1] = ("q", heat_rate, ureg.kilowatt)
                fac_input[0] = ("fluid_facility", fac_f, None)
                fac_input[1] = ("T_i_C", T_in, ureg.degC)
                fac_input[4] = ("x_C", COLD_QUALITY, ureg.dimensionless)
                it_input[0] = ("fluid_it", it_f, None)
                it_input[4] = ("x_H", HOT_QUALITY, ureg.dimensionless)

                # Build fluid objects and pick representative temperature for property lookups.
                fluid_name_fac = fac_input[0][1]
                fluid_name_it = it_input[0][1]
                fluid_fac = fp.fluid(fluid_name_fac)
                fluid_it = fp.fluid(fluid_name_it)
                two_phase_fac = two_phase_check(fluid_name_fac)
                two_phase_it = two_phase_check(fluid_name_it)
                T_props = fac_input[1][1] or fac_input[2][1] or it_input[1][1] or fac_input[2][1]
                dT_fac = Q_(DT_2P,ureg.kelvin) if two_phase_fac else Q_(DT_1P, ureg.kelvin)
                dT_it = Q_(DT_2P,ureg.kelvin) if two_phase_it else Q_(DT_1P, ureg.kelvin)

                # Look up approximate properties from single known temperature to calculate a mass flow rate.
                if two_phase_fac:
                    cp_val_fac = fluid_fac.get_properties("T", T_props, ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                    cp_fac = Q_(cp_val_fac, ureg.joule / (ureg.kilogram * ureg.kelvin))
                    h_liq_fac = fluid_fac.get_properties("T", T_props, ureg.celsius, "Q", 0, ureg.dimensionless)["H"]
                    h_vap_fac = fluid_fac.get_properties("T", T_props, ureg.celsius, "Q", 1, ureg.dimensionless)["H"]
                    hfg_fac = Q_(h_vap_fac - h_liq_fac, ureg.joule / ureg.kilogram)
                else:
                    cp_val_fac = fluid_fac.get_properties("T", T_props, ureg.celsius, "P", 101.3, ureg.kilopascal)["C"]
                    cp_fac = Q_(cp_val_fac, ureg.joule / (ureg.kilogram * ureg.kelvin))
                    hfg_fac = Q_(0, ureg.joule / ureg.kilogram)
                if two_phase_it:
                    cp_val_it = fluid_it.get_properties("T", T_props, ureg.celsius, "Q", 0, ureg.dimensionless)["C"]
                    cp_it = Q_(cp_val_it, ureg.joule / (ureg.kilogram * ureg.kelvin))
                    h_liq_it = fluid_it.get_properties("T", T_props, ureg.celsius, "Q", 0, ureg.dimensionless)["H"]
                    h_vap_it = fluid_it.get_properties("T", T_props, ureg.celsius, "Q", 1, ureg.dimensionless)["H"]
                    hfg_it = Q_(h_vap_it - h_liq_it, ureg.joule / ureg.kilogram)
                else:
                    cp_val_it = fluid_it.get_properties("T", T_props, ureg.celsius, "P", 101.3, ureg.kilopascal)["C"]
                    cp_it = Q_(cp_val_it, ureg.joule / (ureg.kilogram * ureg.kelvin))
                    hfg_it = Q_(0, ureg.joule / ureg.kilogram)

                # Bring in other parameters necessary to calculate mass flow rate in the correct units.
                q = Q_(sys_input[1][1]*1000,ureg.watt)
                x_C = Q_(fac_input[4][1],ureg.dimensionless)
                x_H = Q_(it_input[4][1],ureg.dimensionless)

                # Estimate required mass flow rates based on target load and chosen temperature lifts.
                if 'm_dot_fac' not in locals():
                    m_dot_fac = q/(cp_fac*dT_fac + x_C * hfg_fac)
                if 'm_dot_it' not in locals():
                    m_dot_it = q/(cp_it*dT_it + x_H * hfg_it)

                fac_input[3] = ("m_dot_C", m_dot_fac.magnitude, ureg.kilogram / ureg.second)
                it_input[3] = ("m_dot_H", m_dot_it.magnitude, ureg.kilogram / ureg.second)

                # Remove single phase qualities as expected by HeatExchanger class.
                if not two_phase_fac:
                    fac_input[4] = ("x_C", None, ureg.dimensionless)
                if not two_phase_it:
                    it_input[4] = ("x_H", None, ureg.dimensionless)

                # Instantiate and display knowns for quick manual verification.
                hx = HeatExchanger(sys_input, fac_input, it_input, out_print=True)

                # Set up iteration variables.
                max_iter = 100
                i = 0
                abs_approach = 1
                UA_updated = sys_input[0][1]

                # Iterate on UA until the approach temperature converges to the target.
                while abs_approach > 0.02:
                    sys_input[0] = ("UA", UA_updated, ureg.watt / ureg.kelvin)
                    hx = HeatExchanger(sys_input, fac_input, it_input, out_print=True)

                    cold_approach = (hx.internal_knowns["T_o_H"] - hx.internal_knowns["T_i_C"]).magnitude
                    hot_approach = (hx.internal_knowns["T_i_H"] - hx.internal_knowns["T_o_C"]).magnitude
                    if hx.mode == "1P-1P":
                        approach = min(cold_approach,hot_approach)
                    else:
                        int_approach_1p = (hx.internal_knowns["T_i_H_1p"] - hx.internal_knowns["T_o_C_1p"]).magnitude
                        int_approach_1p2p = (hx.internal_knowns["T_i_H_1p2p"] - hx.internal_knowns["T_o_C_1p2p"]).magnitude
                        print(cold_approach, hot_approach, int_approach_1p, int_approach_1p2p)
                        approach = min(cold_approach,hot_approach,int_approach_1p,int_approach_1p2p)
                    if approach > T_APPROACH:
                        UA_updated = sys_input[0][1]*(2-((i+1)/(101)))
                    else:
                        UA_updated = sys_input[0][1]*(0.5+((i+1)/(201)))
                    
                    abs_approach = abs(approach-T_APPROACH)/T_APPROACH
                    i += 1
                    if i >= max_iter:
                        raise ValueError(f"Max iterations reached searching for UA. UA:{UA_updated}, approach ratio: {approach/T_APPROACH}")

                print(f"Approach: {approach:.3} °C, Iterations: {i}")

                ################################################################################
                # Optional save to CSV for quick Excel import
                ################################################################################
                resp = "yes" if AUTO_SAVE else input("Save results to HX_Data.csv? [y/N]: ").strip().lower()
                if resp in ("y", "yes"):
                    filepath = "HX_Data.csv"
                    fieldnames = [
                        "Facility Fluid",
                        "IT Fluid",
                        "Heat Rate (kW)",
                        "UA (W/K)",
                        "Hot Inlet Temp (C)",
                        "Hot Outlet Temp (C)",
                        "Cold Inlet Temp (C)",
                        "Cold Outlet Temp (C)",
                        "Hot Mass Flow (kg/s)",
                        "Cold Mass Flow (kg/s)",
                    ]

                    row = {
                        "Facility Fluid": fluid_name_fac,
                        "IT Fluid": fluid_name_it,
                        "Heat Rate (kW)": hx.internal_knowns["q"].to(ureg.kilowatt).magnitude,
                        "UA (W/K)": hx.internal_knowns["UA"].to(ureg.watt / ureg.kelvin).magnitude,
                        "Hot Inlet Temp (C)": hx.internal_knowns["T_i_H"].to(ureg.degC).magnitude,
                        "Hot Outlet Temp (C)": hx.internal_knowns["T_o_H"].to(ureg.degC).magnitude,
                        "Cold Inlet Temp (C)": hx.internal_knowns["T_i_C"].to(ureg.degC).magnitude,
                        "Cold Outlet Temp (C)": hx.internal_knowns["T_o_C"].to(ureg.degC).magnitude,
                        "Hot Mass Flow (kg/s)": hx.internal_knowns["m_dot_H"].to(ureg.kilogram / ureg.second).magnitude,
                        "Cold Mass Flow (kg/s)": hx.internal_knowns["m_dot_C"].to(ureg.kilogram / ureg.second).magnitude,
                    }

                    # Write header only once, then append rows per user run.
                    need_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
                    with open(filepath, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if need_header:
                            writer.writeheader()
                        writer.writerow(row)
                    print(f"Saved results to {filepath}")
                else:
                    print("Data not saved.")
