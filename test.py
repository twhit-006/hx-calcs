import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
DIMENSIONS = {
    "T": ureg.kelvin,
    "P": ureg.pascal,
    "H": ureg.joule / ureg.kilogram,
    "S": ureg.joule / (ureg.kilogram * ureg.kelvin),
    "D": ureg.kilogram / ureg.meter**3,
    "V": ureg.meter**3 / ureg.kilogram,
    "Q": ureg.dimensionless,
}

def to_si(p, v, u):
    nonsi = Q_(v, u)
    si = nonsi.to(DIMENSIONS[p])
    return si
    
val = to_si("D", 100, "lb/ft**3")
print(val)