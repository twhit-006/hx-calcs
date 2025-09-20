import pint
    
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Simple unit conversion function using Pint. Must use appropriate unit strings.
def unit_conversion(quantity, from_units, to_units):
    
    
    quantity = quantity
    from_units = from_units
    to_units = to_units

    property_in = Q_(quantity, from_units)
    property_out = property_in.to(to_units)
    return property_out

if __name__ == "__main__":
    # Example usage
    result = unit_conversion(100, ureg.celsius, ureg.kelvin)
    print(result)  # Output: 373.15 kelvin