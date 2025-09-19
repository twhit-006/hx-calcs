# Simple unit conversion function using Pint. Must use appropriate unit strings.
def property_conversion(quantity, from_units, to_units):
    import pint
    
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity
    
    quantity = quantity
    from_units = from_units
    to_units = to_units

    property_in = Q_(quantity, from_units)
    property_out = property_in.to(to_units)
    return property_out