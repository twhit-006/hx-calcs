import numpy as np

def effectiveness(configuration, CR, NTU):
    if configuration == "counterflow":
        if CR == 0:
            epsilon = 1 - np.exp(-NTU)
        elif CR == 1:
            epsilon = NTU / (1 + NTU)
        else:
            epsilon = (1 - np.exp(-NTU * (1 - CR))) / (1 - CR * np.exp(-NTU * (1 - CR)))
    elif configuration == "parallel":
        epsilon = (1 - np.exp(-NTU * (1 + CR))) / (1 + CR)
    elif configuration == "crossflow":
        print("Crossflow effectiveness approximation used: Cmin stream mixed, Cmax stream unmixed. Verify acceptability.")
        if CR == 0:
            epsilon = 1 - np.exp(-NTU)
        else: 
            epsilon = 1 - np.exp(-1*(1 - np.exp(-1*NTU*CR))/CR)
    else:
        raise ValueError("Invalid configuration. Choose from 'counterflow', 'parallel'.")
    
    return epsilon

if __name__ == "__main__":
    epsilon = effectiveness("counterflow", 0.5, 2)
    print(f"Effectiveness: {epsilon}")