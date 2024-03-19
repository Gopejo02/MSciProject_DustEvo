import numpy as np

# density
def density_radial(R, R_c, Sigma_c, exp_density):
    amplitude = Sigma_c*(R/R_c)**(-exp_density) 
    exponent = -(R/R_c)**(2-exp_density)
    return amplitude * np.exp(exponent)

def scale_height(R, R_c, h_c, exp_scale):
    return h_c*(R/R_c)**exp_scale

def density_vertical_gas(density_radial_gas, zr, h, R):
    amplitude = (density_radial_gas)/(np.sqrt(2*np.pi)*h*R)
    return amplitude*np.exp(-0.5*(zr/h)**2)

def density_vertical_big(f, density_radial_dust, zr, h, chi, R):
    amplitude = (f*density_radial_dust)/(np.sqrt(2*np.pi)*R*h * chi)
    return amplitude* np.exp(-0.5*(zr/(chi* h))**2)

def temperature_radial(T_0, R, q):
    return T_0 * (R) ** (-q)


#temperature
def planck_function(nu, T):
    '''
    Planck function in cgs units (erg/s/cm²/sr/Hz)
    '''
    h   = 6.62607015e-27       # erg s Hz^-1
    c   = 2.997924562e10       # cm/s
    k   = 1.380649e-16         # erg/K

    term1 = (2 * h * nu**3) / c**2
    term2 = 1 / (np.exp((h * nu) / (k * T)) - 1)

    return term1 * term2
def planck_mean_opacity(opac_nu, opac_kappa, T):
    '''
    Calculate the Planck mean opacity using the trapezoidal rule for numerical integration.
    '''
    # Calculate the integrands: kappa_nu * B_nu(T) and B_nu(T)
    integrand_num = opac_kappa * planck_function(opac_nu, T)
    integrand_denom = planck_function(opac_nu, T)

    # Perform numerical integration using the trapezoidal rule
    integral_num = np.trapz(integrand_num, opac_nu)
    integral_denom = np.trapz(integrand_denom, opac_nu)

    # Calculate the Planck mean opacity
    kappa_Pl = integral_num / integral_denom

    return kappa_Pl

# number density 
def distribute_grains(rho_total, sizes):
    rho_material = 4

    # Size distribution 
    n = sizes**-3.5    

    # Mass
    mass_grain = (4/3)*np.pi*((sizes*1e-4)**3) * rho_material
    mass_total = np.sum(n * mass_grain)    

    # Rescale number density so rho_total is correct
    n = n * (rho_total / mass_total)  

    check = (n * mass_grain  - rho_total)
    

    #print(f'Diff Mass Density = {check} g cm^-3')  # check it worked

    return n

###############
# Grain Growth
##############
def v_thermal(T):
    """provides themal motion of dust grains"""
    
    k = 1.380649e-16 #erg⋅cm−2⋅s−1⋅K−4
    mH = 1.673534e-24
    m = 1.68e-14 #gram
    
    return np.sqrt(8 * k * T /(np.pi * 2.3 * mH))
    
def grow_grain(a0, n_i, v, S, mH, dt, rho_material):
    """provides new grain radius (in cm) after dt interval of collissiqns"""
    
    A = 4 * np.pi * (a0 * 1e-4)**2
    top = n_i * v * S * A * mH * dt
    bottom = 4 * rho_material
    return top/bottom

###############
# Grain cooling
###############

def cooling_timescale(a, C, dT, Tdust, rho_material):
    """Time to cool down dust in seconds"""
    SBconst = 5.670374419e-5 #erg⋅cm−2⋅s−1⋅K−4
    m = rho_material * (4/3) * np.pi * (a * 1e-4) ** 3 # mass of the grain
    A = 4 * np.pi * (a * 1e-4) **2
    P = A * SBconst * Tdust ** 4 # erg s-1
    if Tdust <= 250:
        Qrad = 1.25e-5 * (Tdust ** 2) * a # a in microns
    else:
        Qrad = 1.25e-3 * a # a in microns

    tcool =( m * C * dT )/(Qrad * P)
    return tcool

####################
# Settling velocity
####################

def Omega(Mstar, r):
    """keplerian angular velocity"""
    G = 6.67e-8 #dyn⋅cm2⋅g−2
    return np.sqrt((G * Mstar) / (r ** 3))

def settle_velocity(rho_material, rho_gas, a, v_th, r, z):
    """settling velocity"""
    G = 6.67e-8 #dyn⋅cm2⋅g−2
    Mstar = 2e33
    return (rho_material/rho_gas) * ((a*1e-4)/v_th) * (G*Mstar/(r**3)) * z
