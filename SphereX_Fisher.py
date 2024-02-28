# SPHEREx with wb,wcdm,h,ns,H(z),Da(z),Psn(z),b(z),f(z) as parameters

# this code is intended to work in Jupyter Notebook
# I'm using hashtags to indicate the separation between cells
# in the initial cells I am defining all the functions
# the cell which actually starts the Fisher is that indicated by three rows of X

from classy import Class
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import math
from math import pi
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as planck


##########################################################################
##########################################################################
##########################################################################


#flat LCDM
cosmo={
        "h":0.67,
        "wb":0.022445,
        "wcdm":0.12,
        "Wl":0.683,
        "ns":0.96,
        "tau":0.058,
        "H0":67.0,
        "s8":0.816,
        "Wb":0.05,
        "Wcdm":0.267,
        "z_eq":3196,
        "z_reion":10.5,
        "Wm":0.317,
        "Wk":0.,
        "wm":0.142,
        "Mv":0.06
        }

print()
print("REFERENCE COSMOLOGY")
print()
print(cosmo)


##########################################################################
##########################################################################
##########################################################################



def E(z,cosmo):
    wm=cosmo["wb"]+cosmo["wcdm"]
    h=cosmo["h"]
    Wm=wm/(h**2)
    return np.sqrt(Wm*(1+z)**3+(1.-Wm))

# H in km/(s*Mpc)
def H(z,cosmo):
    return cosmo["h"]*100*E(z,cosmo)

#comoving distance [Mpc]
def r(z,cosmo): 
    c=3e5
    com=integrate.quad(func,0,z,args=(cosmo))[0]*c
    return com

def func(z,cosmo):
    return 1/H(z,cosmo)


#Vsurvey in (Mpc/h)^3
def compute_Vsurvey_single(zmin,zmax,fsky,cosmo):
    Area_of_sphere=41252.96125 #deg^2        
    Survey_Area=Area_of_sphere*fsky
    # Survey_Area=1464 for PFS
    conversion=(180/pi)**2
    solid_angle=Survey_Area/conversion
    rmin=r(zmin,cosmo)
    rmax=r(zmax,cosmo)
    Volume=solid_angle/3*((rmax*cosmo["h"])**3-(rmin*cosmo["h"])**3)
    return Volume

def compute_Vsurvey_array(zmin_vec,zmax_vec,fsky,cosmo):
    v=[]
    for i in range(len(zmin_vec)):
        v.append(compute_Vsurvey_single(zmin_vec[i],zmax_vec[i],fsky,cosmo))
    return v



def Omega_m(z,cosmo):
    wm=cosmo["wb"]+cosmo["wcdm"]
    h=cosmo["h"]
    Wm=wm/(h**2)
    return Wm*(1+z)**3/(E(z,cosmo))**2
    
def f(z,cosmo):
    gamma=0.55
    return Omega_m(z,cosmo)**gamma

def f_array(cosmo):
    ff=[]
    for z in cosmo["z"]:
        ff.append(f(z,cosmo))
    return ff


def func2(z,cosmo):
    return f(z,cosmo)/(1+z)

def D(z,cosmo):
    int=integrate.quad(func2,0,z,args=(cosmo))[0]
    return math.e**(-int)


def D_array(cosmo):
    DD=[]
    for z in cosmo["z"]:
        DD.append(D(z,cosmo))
    return DD

def H_array(cosmo):
    Hs=[]
    for z in cosmo["z"]:
        Hs.append(H(z,cosmo))
    return Hs

# in [Mpc]
def Da(z,cosmo): 
    c=3e5
    da=integrate.quad(func,0,z,args=(cosmo))[0]*c/(1+z)
    return da

def Da_array(cosmo):
    Das=[]
    for z in cosmo["z"]:
        Das.append(Da(z,cosmo))
    return Das



def D_from_f_array(cosmo):
    DD=[]
    zero=((1+cosmo["z"][0])/(1+zmin[0]))**(-cosmo["f"][0])
    DD.append(zero)
    for i in range(len(cosmo["z"])-1):
        first=1.
        for j in range(i+1):
            first*=((1+cosmo["zmax"][j])/(1+cosmo["zmin"][j]))**(-cosmo["f"][j])
        DD.append(first*((1+cosmo["z"][j+1])/(1+zmin[j+1]))**(-cosmo["f"][j+1])) 
    return DD




def compute_kmin(cosmo):
    kmin_vec=[]
    for i in range(len(cosmo["Vsurvey"])):
        kmin_vec.append(2*math.pi/((cosmo["Vsurvey"][i])**(1/3)))
        #kmin_vec.append(5e-4)
    return np.array(kmin_vec)


def generate_k_matrix(k_specs):
    kmax=k_specs[1]
    nks=k_specs[2]
    k_matrix=np.zeros((len(k_specs[0]),nks))
    for i in range(len(k_specs[0])):
        kmin=k_specs[0][i]
        k_matrix[i]=np.geomspace(kmin,kmax,nks)
    return k_matrix

def generate_mu_array(mu_specs):
    mu_min=mu_specs[0]
    mu_max=mu_specs[1]
    nmus=mu_specs[2]
    return np.linspace(mu_min,mu_max,nmus)


def generate_mu_array_from_angles(nmus):
    min_angle=math.pi/2
    max_angle=0
    angles_array=np.linspace(min_angle,max_angle,nmus)
    print(angles_array)
    muarray=np.cos(angles_array)
    print(muarray)
    #.plot(muarray)
    #plt.show()
    return muarray

# -----------------------------------------------------------------------------


def fill_cosmo(zmin_vec,zmax_vec,cosmo,z_vec,n,bias,fsky,z_error):

    cosmo["z_error"]=z_error
    cosmo["z"]=z_vec
    cosmo["zmax"]=zmax_vec
    cosmo["zmin"]=zmin_vec
    
    V_survey_array=compute_Vsurvey_array(zmin_vec,zmax_vec,fsky,cosmo)
    fs=f_array(cosmo)
    
    cosmo["f"]=fs
    Ds=D_from_f_array(cosmo)
    Hs=H_array(cosmo)
    Das=Da_array(cosmo)
    Pshot=np.zeros(len(z_vec))
    
    cosmo["bias"]=bias
    cosmo["n"]=n
    cosmo["D"]=Ds
    cosmo["Vsurvey"]=V_survey_array
    cosmo["H"]=Hs
    cosmo["Da"]=Das
    cosmo["Psn"]=Pshot
    
    

print(cosmo)



############################################################################
############################################################################
#############################################################################




def wiggle_EH98(k,cosmo,cosmo_ref):         # in absence of neutrinos
    h_ref=cosmo_ref["h"]
    h=cosmo["h"]
    wb = cosmo ["wb"]
    wcdm = cosmo["wcdm"]
    
    Wcdm = wcdm/h**2
    Wb = wb/h**2
    wm = wb+wcdm
    Wm = wm/h**2
    
    return compute_Tw(k,wb,wm,Wb,Wm,Wcdm,h)


def compute_Tw(k,wb,wm,Wb,Wm,Wcdm,h):
    
    Tcmb0=planck.Tcmb0.value
    f_baryon=Wb/Wm
    theta_cmb=Tcmb0/2.7
    
    z_eq=2.5e4*wm*theta_cmb**(-4)
    k_eq=0.0746*wm*theta_cmb**(-2)  # in 1/Mpc


    #soundhorizon
    b1=0.313*wm**-0.419*(1+0.607*wm**0.674)
    b2=0.238*wm**0.223
    z_drag=1291*wm**0.251/(1.+0.659*wm**0.828)*(1.+b1*wb**b2)

    r_drag=31.5*wb*theta_cmb**(-4)*(1000./z_drag) 
    r_eq=31.5*wb*theta_cmb**(-4)*(1000./z_eq)

    sound_horizon=2./(3.*k_eq)*np.sqrt(6./r_eq)*np.log((np.sqrt(1+r_drag)+np.sqrt(r_drag+r_eq))/(1+np.sqrt(r_eq)))
    k_silk=1.6*wb**0.52*wm**0.73*(1+(10.4*wm)**(-0.95))  # in 1/Mpc

    #alpha_c
    aca1=(46.9*wm)**0.670*(1+(32.1*wm)**(-0.532))
    aca2=(12.0*wm)**0.424*(1+(45.0*wm)**(-0.582))
    alpha_c=aca1**-f_baryon*aca2**(-f_baryon**3)

    #beta_c
    bcb1=0.944/(1.+(458.*wm)**-0.708)
    bcb2=(0.395*wm)**(-0.0266)
    beta_c=1./(1.+bcb1*((Wcdm/Wm)**bcb2-1.))

    y=(1.+z_eq)/(1.+z_drag)
    G=y*(-6.*np.sqrt(1.+y)+(2.+3.*y)*np.log((np.sqrt(1+y)+1)/(np.sqrt(1.+y)-1.)))
    alpha_b=2.07*k_eq*sound_horizon*(1+r_drag)**-0.75*G

    beta_node=8.41*wm**0.435
    beta_b=0.5+f_baryon+(3.-2.*f_baryon)*np.sqrt((17.2*wm)**2+1.)

    k=k*h #1/Mpc
    q=k/(13.41*k_eq)
    ks=k*sound_horizon

    ln_beta=np.log(np.e+1.8*beta_c*q)
    ln_1=np.log(np.e+1.8*q)
    C_a=14.2/alpha_c+386./(1.+69.9*q**1.08)
    C_1=14.2+386./(1.+69.9*q**1.08)

    f=1./(1.+(ks/5.4)**4.)
    T0_tilde=lambda a,b:a/(a+b*q**2)
    T_c=f*T0_tilde(ln_beta,C_1)+(1-f)*T0_tilde(ln_beta,C_a)

    s_tilde=sound_horizon*(1+(beta_node/ks)**3.)**(-1./3.)
    ks_tilde=k*s_tilde

    T_b_1=T0_tilde(ln_1,C_1)/(1.+(ks/5.2)**2.)
    T_b_2=alpha_b/(1+(beta_b/ks)**3.)*np.exp(-(k/k_silk)**1.4)
    T_b=np.sinc(ks_tilde/np.pi)*(T_b_1+T_b_2)

    T=f_baryon*T_b+(1-f_baryon)*T_c
    return T



def PkEH98_w(k,cosmo,cosmo_ref):   ### we employ the Euclid VII noramlization

    from scipy.interpolate import CubicSpline as spline
    import mcfit
 

    def sigma_r(r,cosmo,cosmo_ref,k):
        ns=cosmo["ns"]
        kkk=np.geomspace(1e-5,10,1000)
        ppk=((kkk)**ns)*wiggle_EH98(kkk,cosmo,cosmo_ref)**2
        R, sigmasq = mcfit.TophatVar(kkk, lowring=True)(ppk, extrap=True)
        return spline(R, sigmasq)(r)**0.5
        
    #print()
    #print(cosmo["s8"]**2/sigma_r(8/h_ref,cosmo,cosmo_ref,k*h_ref)**2)
    #print()
    return cosmo["s8"]**2/sigma_r(8,cosmo,cosmo_ref,k)**2*wiggle_EH98(k,cosmo,cosmo_ref)**2*(k)**cosmo["ns"]
    


def sigma_r(i,cosmo,cosmo_ref):       # smearing of the galaxy density field along the l.o.s
    return (3e5/cosmo["H"][i])*cosmo_ref["h"]*(1+cosmo["z"][i])*cosmo["z_error"]
   
def smearing(k,mu,i,cosmo,cosmo_ref):
    #return 1.
    return math.e**(-k**2*mu**2*sigma_r(i,cosmo,cosmo_ref)**2)


def Pobs_interp(k,mu,i,cosmo,cosmo_ref):
    q_par=cosmo_ref["H"][i]/cosmo["H"][i]
    q_per=cosmo["Da"][i]/cosmo_ref["Da"][i]
    mu_mod=mu*q_per/q_par*(1.+mu**2*((q_per/q_par)**2-1.))**(-0.5)
    k_mod=k/q_per*(1.+mu**2*((q_per/q_par)**2-1.))**(0.5)
    
    return 1./(q_per**2*q_par)*(cosmo["bias"][i]+cosmo["f"][i]*mu_mod**2)**2*(cosmo["D"][i])**2*PkEH98_w(k_mod,cosmo,cosmo_ref)*smearing(k_mod,mu_mod,i,cosmo,cosmo_ref)+cosmo["Psn"][i]
  
  
  
  
  

##########################################################################
############################################################################
#############################################################################



def generate_cosmo_plusminus_der(cosmo,og_step,p,z_vec,i,nonz_params,z_params,k): 

    flag=0
    step=np.copy(og_step)
    
    cosmo_plus=cosmo.copy()
    cosmo_minus=cosmo.copy()
    cosmo_plusplus=cosmo.copy()
    cosmo_minusminus=cosmo.copy()

    
    if p in nonz_params:

        if p=="wb":
            step=0.04
        if p=="wcdm":
            step=0.02
        if p=="h":
            step=0.02
        cosmo_plus[p]=(1+step)*cosmo_plus[p]
        cosmo_minus[p]=(1-step)*cosmo_minus[p]
        cosmo_plusplus[p]=(1+2*step)*cosmo_plusplus[p]
        cosmo_minusminus[p]=(1-2*step)*cosmo_minusminus[p] 
        
        delta=cosmo[p]*step
        

        
        return cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag
            
    else:

        red_bins=[]

        for index,z in enumerate(z_vec):
            red_bins.append(index)

        for red in red_bins:
            if p.endswith(str(red))==True:
                position=red
        if position!=i:
            flag=2
            delta=0
            return cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag

        else:
            for fl in z_params: 
                if p.startswith(fl)==True:
                    name=fl
        
        
            if name=="Psn":
                flag=1
                delta=0

            
                return cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag

            else:
                #if name=="H":
                    #step=1e-5
                plus=np.copy(cosmo_plus[name])
                plus[i]=(1+step)*plus[i]
                cosmo_plus[name]=plus

                minus=np.copy(cosmo_minus[name])
                minus[i]=(1-step)*minus[i]
                cosmo_minus[name]=minus
            
                plusplus=np.copy(cosmo_plusplus[name])
                plusplus[i]=(1+2*step)*plusplus[i]
                cosmo_plusplus[name]=plusplus
            
                minusminus=np.copy(cosmo_minusminus[name])
                minusminus[i]=(1-2*step)*minusminus[i]
                cosmo_minusminus[name]=minusminus

                delta=cosmo[name][i]*step

                
                cosmo_plus["D"]=D_from_f_array(cosmo_plus)
                cosmo_minus["D"]=D_from_f_array(cosmo_minus)
                cosmo_plusplus["D"]=D_from_f_array(cosmo_plusplus)
                cosmo_minusminus["D"]=D_from_f_array(cosmo_minusminus)

 
                return cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag
    



##########################################################################
############################################################################
#############################################################################




def Veff(k,mu,i,cosmo):
    dd=cosmo["n"][i]*Pobs_interp(k,mu,i,cosmo,cosmo)
    #print(dd)
    return (dd/(dd+1.))**2*cosmo["Vsurvey"][i]


def compute_Veff_matrix(z_vec,mu_array,k_matrix,cosmo):
        Veff_array=[]
        for i in range(len(z_vec)):
            Veff_single=np.zeros((len(k_matrix[i]),len(mu_array)))
            for key,k in enumerate(k_matrix[i]):
                Veff_single[key]=Veff(k,mu_array,i,cosmo)
            Veff_array.append(Veff_single)
        Veff_array=np.array(Veff_array)
        return Veff_array




##########################################################################
############################################################################
#############################################################################



def der_interp(p,k,mu,i,cosmo,step,z_vec,nonz_params,z_params,pref,pmin,pplus,deltas,k_h,k_ns,k_wb,k_wcdm):
    
    cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag=generate_cosmo_plusminus_der(cosmo,step,p,z_vec,i,nonz_params,z_params,k)

    if flag==0:
        dPobs_dp=8*(Pobs_interp(k,mu,i,cosmo_plus,cosmo)-Pobs_interp(k,mu,i,cosmo_minus,cosmo))/(12*delta)-(Pobs_interp(k,mu,i,cosmo_plusplus,cosmo)-Pobs_interp(k,mu,i,cosmo_minusminus,cosmo))/(12*delta)
        #dPobs_dp=(Pobs_interp(k,mu,i,Pk0[0,x],cosmo_plus,cosmo,k_vec)-Pobs_interp(k,mu,i,Pk0[1,x],cosmo_minus,cosmo,k_vec))/(2*delta)
        derivative=dPobs_dp/Pobs_interp(k,mu,i,cosmo,cosmo)
    elif flag==1:
        derivative=(Pobs_interp(k,mu,i,cosmo,cosmo))**(-1)
    elif flag==2:
        derivative=np.zeros(len(mu))

    
    if p=="h":
        k_h.append(derivative[0])

    if p=="ns":
        k_ns.append(derivative[0])  

    if p=="wb":
        k_wb.append(derivative[0])

    if p=="wcdm":
        k_wcdm.append(derivative[0])
        
    #    pref.append(PkEH98_w(k,cosmo,cosmo))
    #    pplus.append(PkEH98_w(k,cosmo_plus,cosmo))
    #    pmin.append(PkEH98_w(k,cosmo_minus,cosmo))
    #    deltas.append(delta)
    
    return derivative 




def compute_derivative_matrix(param_names,nonz_params,z_params,k_specs,mu_array,cosmo,step,z_vec,k_matrix):
    deriv_matrix=[]
  

    for i in range(len(z_vec)):
        print()
        print("### ",i)
        print()
        deriv_single=np.zeros((len(param_names),len(k_matrix[i]),len(mu_array)))
        pref=[]
        pmin=[]
        pplus=[]
        deltas=[]
        k_h=[]
        k_ns=[]
        k_wb=[]
        k_wcdm=[]
        for x,p in enumerate(param_names):
            print(p)
            for key,k in enumerate(k_matrix[i]):
                deriv_single[x,key]=der_interp(p,k,mu_array,i,cosmo,step,z_vec,nonz_params,z_params,pref,pmin,pplus,deltas,k_h,k_ns,k_wb,k_wcdm)
        #print(deriv_single)
        #print()
        deriv_matrix.append(deriv_single)
        """
        plt.figure(figsize=(12,5))
        pref=np.array(pref)
        pmin=np.array(pmin)
        pplus=np.array(pplus)
        deltas=np.array(deltas)
        add=(pplus-pmin)/(2*deltas)
        add/=pref
        plt.semilogx(k_matrix[i],add/add[0])
        plt.ylim(-0.6,1.1)
        plt.show()
        """
        plt.plot(k_h,label="h")
        plt.plot(k_ns,label="ns")
        plt.plot(k_wcdm,label="wcdm")
        plt.plot(k_wb,label="wb")
        plt.title("Derivata del logPobs rispetto ai parametri")
        plt.ylabel("dlog(Pobs)/dtheta")
        plt.xlabel("k-bin")
        plt.legend()
        plt.show()
    return np.array(deriv_matrix)


##########################################################################
############################################################################
#############################################################################



def F(pi,pj,z_vec,Veff_matrix,deriv_matrix,k_matrix,mu_array,cosmo):
    f=0.
    for x in range(len(z_vec)):
        k_mu_matrix,y=np.meshgrid(k_matrix[x],mu_array)
        k_mu_matrix=k_mu_matrix.transpose()
        funcz=Veff_matrix[x]*deriv_matrix[x,pi,:,:]*deriv_matrix[x,pj,:,:]*k_mu_matrix**2
        one=np.trapz(funcz,k_matrix[x],axis=0)
        two=np.trapz(one,mu_array,axis=0)
        l=2*two/(8*math.pi**2)
        f+=l
        print(x,l)
    print(f)
    print()
    return f

def makeFisher(param_names,nonz_params,z_params,z_vec,k_specs,mu_specs,cosmo,step):
    ll=len(param_names)
    Fisher=np.zeros((ll,ll))
    k_matrix=generate_k_matrix(k_specs)
    mu_array=generate_mu_array(mu_specs)
    #mu_array=generate_mu_array_from_angles(mu_specs[2])
    print("______ Computing Veff_matrix ...")
    Veff_matrix=compute_Veff_matrix(z_vec,mu_array,k_matrix,cosmo)
    print("______ Done")
    print()
    print("______ Computing deriv_matrix ...")
    deriv_matrix=compute_derivative_matrix(param_names,nonz_params,z_params,k_specs,mu_array,cosmo,step,z_vec,k_matrix)
    print("______ Done")
    print()
    print()
    print("Start Fisher")
    print()
    square=len(nonz_params)
    bins=len(z_params)
    bins_vec=np.arange(square,ll,bins)
    counter=0
    for pi in range(ll):
        if pi==(bins_vec[counter]+bins):
            counter+=1
        for pj in range(ll):
            if pi>=square and pj>=square:
                if pi>=bins_vec[counter] and pi<(bins_vec[counter]+bins):
                    if pj<bins_vec[counter] or pj>=(bins_vec[counter]+bins):
                        continue
            print(pi,pj)            
            Fisher[pi,pj]+=F(pi,pj,z_vec,Veff_matrix,deriv_matrix,k_matrix,mu_array,cosmo)
    return Fisher



##########################################################################
############################################################################
#############################################################################


def create_parameters_array(param_names,nonz_params,z_params,z_vec,cosmo):
    red_bins=[]
    par_values=[]
    for i,z in enumerate(redshift):
        red_bins.append(i)
    
    for p in param_names:
        if p in nonz_params:
            par_values.append(cosmo[p])
        else:
            for fl in z_params: 
                if p.startswith(fl)==True:
                    name=fl
            for red in red_bins:
                if p.endswith(str(red))==True:
                    position=red
            par_values.append(cosmo[name][position])
    return par_values



def print_reference_params(param_names,params):
    print()
    print("REFERENCE PARAMETERS")
    print()
    d=[]
    for i in range(len(params)):
        a=[]
        a.append(param_names[i])
        a.append(params[i])
        d.append(a)

    from tabulate import tabulate
    print (tabulate(d, headers=["Parameter", "Reference value"]))
    
    

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



# -----------------------------------------------------------------
# IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII   
# ---------------------------------------------------------------------

#Spherex
zmin=np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.6,2.2,2.8,3.4,4.0])
zmax=np.array([0.2,0.4,0.6,0.8,1.0,1.6,2.2,2.8,3.4,4.0,4.6])
redshift=(zmin+zmax)*0.5

fsky=0.75

n=np.array([0.00997, 0.00411, 0.000501, 7.05e-05, 3.16e-05, 1.64e-05, 3.59e-06, 8.07e-07, 1.84e-06, 1.5e-06, 1.13e-06])

bias=np.array([1.3, 1.5, 1.8, 2.3, 2.1, 2.7, 3.6, 2.3, 3.2, 2.7, 3.8])

z_error=0.003

"""

#PFS #update Vsurvey function as well
zmin=np.array([0.6,0.8,1.0,1.2,1.4,1.6,2.0])
zmax=np.array([0.8,1.0,1.2,1.4,1.6,2.0,2.4])
redshift=(zmin+zmax)*0.5

n=np.array([1.9,6.0,5.8,7.8,5.5,3.1,2.7])*1e-4

bias=np.array([1.18,1.26,1.34,1.42,1.50,1.62,1.78])
"""
fill_cosmo(zmin,zmax,cosmo,redshift,n,bias,fsky,z_error)

print(np.array(cosmo["Vsurvey"])*1e-9*1.1)


mu_min=0.
mu_max=1.
nmus=50

mu_specs=[mu_min,mu_max,nmus]

kmax=0.2
nks=200

k_specs=[compute_kmin(cosmo),kmax,nks]

# fill z-dependente and z-independent parameters arrays - they must have the same names in cosmo

#nonz_params=["wcdm","h","ns","wb"]
nonz_params=["wb","wcdm","ns","h"]

z_params=["H","Da","bias","f","Psn"]   # they must start with different letters
#z_params=[]



param_names=["wb","wcdm","ns","h",
             "H_0","Da_0","bias_0","f_0","Psn_0",
             "H_1","Da_1","bias_1","f_1","Psn_1",
             "H_2","Da_2","bias_2","f_2","Psn_2",
             "H_3","Da_3","bias_3","f_3","Psn_3",
             "H_4","Da_4","bias_4","f_4","Psn_4",
             "H_5","Da_5","bias_5","f_5","Psn_5",
             "H_6","Da_6","bias_6","f_6","Psn_6",
             "H_7","Da_7","bias_7","f_7","Psn_7",
             "H_8","Da_8","bias_8","f_8","Psn_8",
             "H_9","Da_9","bias_9","f_9","Psn_9",
             "H_10","Da_10","bias_10","f_10","Psn_10"]



#param_names=["wb","wcdm","ns","h"]
params=create_parameters_array(param_names,nonz_params,z_params,redshift,cosmo)



print_reference_params(param_names,params)

# -----------------------------------------------------------------
# IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII   
# ---------------------------------------------------------------------

step=0.001
Fisher=makeFisher(param_names,nonz_params,z_params,redshift,k_specs,mu_specs,cosmo,step)






#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX





ll=len(params)
newpar=np.copy(params)
fin=len(Fisher[0])
NewFisher=np.copy(Fisher)
Cov=np.linalg.inv(NewFisher)
print()
print("SSSSSSS")
print(np.diag(Cov))

aa=np.sqrt(np.diag(Cov))
print()
print("Uncertainties:  ",aa)
print()
print("parameters:   ",newpar)
print()
print("relative uncertainties:   ",aa/newpar)
print()

nomi = ["Parameter", "Reference value", "Absolute uncertainty", "Relative uncertainty"]

lll=len(NewFisher[0])
data=np.zeros(4*lll).reshape(lll,4)


print(lll)
d=[]
for i in range(lll):
    #print(i)
    a=[]
    a.append(param_names[i])
    a.append(newpar[i])
    a.append(aa[i])
    a.append(aa[i]/newpar[i])
    d.append(a)


print()

print("FISHER MATRIX")
print()
print(Fisher)
print()
print()
print("COVARIANCE MATRIX")
print()
print(Cov)
print()
print("kmax=",kmax)
print()
from tabulate import tabulate
print (tabulate(d, headers=["Parameter", "Reference value", "Absolute uncertainty", "Relative uncertainty"]))


plt.figure()
plt.imshow(np.log10(np.abs(Cov)),aspect="auto")
plt.colorbar()
plt.title("Covariance")
plt.show()


plt.figure()
plt.imshow(np.log10(np.abs(NewFisher)),aspect="auto")
plt.colorbar()
plt.title("Fisher")
plt.show()





##########################################################################
############################################################################
#############################################################################



cosmo_Cov=Cov[0:4,0:4]

import getdist
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
gauss=GaussianND(newpar[0:4], cosmo_Cov,names=param_names[0:4])
g = plots.get_subplot_plotter()
#g.settings.param_names_for_labels="dd.paramnames"
g.triangle_plot(gauss,filled=True)





##########################################################################
############################################################################
#############################################################################





######################################################### 
# Plotting uncertainties
#########################################################

rel_err_H=[]
rel_err_Da=[]
rel_err_bias=[]
rel_err_f=[]
for pos,name in enumerate(param_names):
    if name.startswith("H"):
        rel_err_H.append(aa[pos]/newpar[pos])
    if name.startswith("Da"):
        rel_err_Da.append(aa[pos]/newpar[pos])
    if name.startswith("bias"):
        rel_err_bias.append(aa[pos]/newpar[pos])
    if name.startswith("f"):
        rel_err_f.append(aa[pos]/newpar[pos])


plt.plot(redshift,rel_err_H,label="H")
plt.plot(redshift,rel_err_Da,label="Da")
plt.plot(redshift,rel_err_bias,label="bias")
plt.plot(redshift,rel_err_f,label="f")
plt.ylabel("Relative uncertainties")
plt.xlabel("z-bin")
plt.tight_layout()
plt.xlim(0,2.5)
plt.ylim(0.008,5)
plt.yscale("log")
plt.legend()
plt.show()

#print("paramaters:   ",par)
#par=[cosmo["wb"],cosmo["wm"],0.67,0.96]   #cosmological parameters
#print("relative uncertainties:   ",aa/par)




##########################################################################
############################################################################
#############################################################################



# check drivative




def generate_cosmo_plusminus_derw(cosmo,og_step,p,z_vec,i,nonz_params,z_params,k): 

    flag=0
    step=np.copy(og_step)
    
    cosmo_plus=cosmo.copy()
    cosmo_minus=cosmo.copy()
    cosmo_plusplus=cosmo.copy()
    cosmo_minusminus=cosmo.copy()

    
    if p in nonz_params:


        cosmo_plus[p]=(1+step)*cosmo_plus[p]
        cosmo_minus[p]=(1-step)*cosmo_minus[p]
        cosmo_plusplus[p]=(1+2*step)*cosmo_plusplus[p]
        cosmo_minusminus[p]=(1-2*step)*cosmo_minusminus[p] 
        
        delta=cosmo[p]*step

        #cosmo_plus["D"]=D_from_f_array(cosmo_plus)
        #cosmo_minus["D"]=D_from_f_array(cosmo_minus)
        #cosmo_plusplus["D"]=D_from_f_array(cosmo_plusplus)
        #cosmo_minusminus["D"]=D_from_f_array(cosmo_minusminus)

            
   
        return cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag
            
    else:

        red_bins=[]

        for index,z in enumerate(z_vec):
            red_bins.append(index)


        for red in red_bins:
            if p.endswith(str(red))==True:
                position=red
        if position!=i:
            flag=2
            delta=0
            return cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag

        else:
            for fl in z_params: 
                if p.startswith(fl)==True:
                    name=fl
        
        
            if name=="Psn":
                flag=1
                delta=0

            
                return cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag

            else:
        
                plus=np.copy(cosmo_plus[name])
                plus[i]=(1+step)*plus[i]
                cosmo_plus[name]=plus

                minus=np.copy(cosmo_minus[name])
                minus[i]=(1-step)*minus[i]
                cosmo_minus[name]=minus
            
                plusplus=np.copy(cosmo_plusplus[name])
                plusplus[i]=(1+2*step)*plusplus[i]
                cosmo_plusplus[name]=plusplus
            
                minusminus=np.copy(cosmo_minusminus[name])
                minusminus[i]=(1-2*step)*minusminus[i]
                cosmo_minusminus[name]=minusminus

                delta=cosmo[name][i]*step

                
                cosmo_plus["D"]=D_from_f_array(cosmo_plus)
                cosmo_minus["D"]=D_from_f_array(cosmo_minus)
                cosmo_plusplus["D"]=D_from_f_array(cosmo_plusplus)
                cosmo_minusminus["D"]=D_from_f_array(cosmo_minusminus)
 
                return cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag
    
    
    
    
# ------------------------------------------------------------


def derw(p,k,mu,i,cosmo,step,step_pos,z_vec,nonz_params,z_params):

    #if step_pos%10==0:
    #    print(step_pos)
    cosmo_plus,cosmo_minus,cosmo_plusplus,cosmo_minusminus,delta,flag=generate_cosmo_plusminus_derw(cosmo,step,p,z_vec,i,nonz_params,z_params,k)

    if flag==0:
        dPobs_dp=8*(Pobs_interp(k,mu,i,cosmo_plus,cosmo)-Pobs_interp(k,mu,i,cosmo_minus,cosmo))/(12*delta)-(Pobs_interp(k,mu,i,cosmo_plusplus,cosmo)-Pobs_interp(k,mu,i,cosmo_minusminus,cosmo))/(12*delta)
        #dPobs_dp=(Pobs_interp(k,mu,i,Pk0[0,x],cosmo_plus,cosmo,k_vec)-Pobs_interp(k,mu,i,Pk0[1,x],cosmo_minus,cosmo,k_vec))/(2*delta)
        derivative=dPobs_dp/Pobs_interp(k,mu,i,cosmo,cosmo)
    elif flag==1:
        derivative=(Pobs_interp(k,mu,i,cosmo,cosmo))**(-1)
    elif flag==2:
        derivative=np.zeros(len(mu))

    return derivative 




def check_derivative(p,cosmo,k_specs,redshift):
    steps=np.geomspace(1e-4,0.1,100)
    
    
    fig,axs=plt.subplots(11,2,figsize=(22,22))
    for i in range(len(redshift)):
        #print(p,i)
        d=[]
        for x,j in enumerate(steps):
            d.append(derw(p,0.2,1.,i,cosmo,j,x,redshift,nonz_params,z_params))
        axs[i,0].semilogx(steps,d,label=f"{i}")
        axs[i,0].axvline(x=0.001,c="black")
        axs[i,0].set_title(f"{p}     k=0.2,mu=1.")
        axs[i,0].legend()
        #axs[i,0].tight_layout()
        
        d=[]
        for x,j in enumerate(steps):
            d.append(derw(p,0.2,0.,i,cosmo,j,x,redshift,nonz_params,z_params))
        axs[i,1].semilogx(steps,d,label=f"{i}")
        axs[i,1].axvline(x=0.001,c="black")
        axs[i,1].set_title(f"{p}     k=0.2,mu=0.")
        axs[i,1].legend()
        #axs[i,1].tight_layout()
        

    print("Done")
    
    plt.title(f"{p},k=0.2")
    plt.tight_layout()
    plt.show()

    print()
    
    fig,axs=plt.subplots(11,2,figsize=(22,22))
    for i in range(len(redshift)):
        #print(p,i)
        
        d=[]
        for x,j in enumerate(steps):
            d.append(derw(p,0.01,1.,i,cosmo,j,x,redshift,nonz_params,z_params))
        axs[i,0].semilogx(steps,d,label=f"{i}")
        axs[i,0].axvline(x=0.001,c="black")
        axs[i,0].set_title(f"{p}     k=0.01,mu=1.")
        axs[i,0].legend()
        #axs[i,2].tight_layout()

        d=[]
        for x,j in enumerate(steps):
            d.append(derw(p,0.01,0.,i,cosmo,j,x,redshift,nonz_params,z_params))
        axs[i,1].semilogx(steps,d,label=f"{i}")
        axs[i,1].axvline(x=0.001,c="black")
        axs[i,1].set_title(f"{p}     k=0.01,mu=0.")
        axs[i,1].legend()
        #axs[i,3].tight_layout()
    
    plt.title(f"{p},k=0.01")
    plt.tight_layout()
    plt.show()

mu_values=np.linspace(0.,1.,50)



##########################################################################################################
nonz_params=["wb","wcdm","h","ns"]

k_matrix=generate_k_matrix(k_specs)


check_derivative("wb",cosmo,k_specs,redshift)
print()
check_derivative("wcdm",cosmo,k_specs,redshift)
print()
check_derivative("h",cosmo,k_specs,redshift)
print()
check_derivative("ns",cosmo,k_specs,redshift)
print()


