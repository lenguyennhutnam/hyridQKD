import numpy as np
import sys,os,json


def hbin(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def delta(n, eps):
    return np.sqrt(n/2 * np.log(1/eps)) if n > 0 else 0.0


### for simulation only ###
def compute_pap(A, tau, R, dead_time_us, mu, eta, Pdc, coeff):
    T=80/R
    Pc0 = 1-np.exp(-mu*eta) + Pdc
    Pnc = 1 - coeff*Pc0
    dt1=T*dead_time_us # Deadtime in R*pulse_distance us

    R1=R*(1-dt1)

    # Sum over however many afterpulse exponential components (A[i], tau[i]) are
    # configured. Previously hardcoded to exactly 3 components, which crashed with
    # IndexError on 2-component configs such as params_aurea.json.
    pap = 0.0
    for Ai, taui in zip(A, tau):
        ui = 1/ (1/taui - R*np.log(Pnc))
        pap += Pnc**R1 * Ai * np.exp(-(dt1+1)/ui) / (1 - np.exp(-1/ui))

    return pap

def compute_qber(dt, mu, eta, A, tau, R, pdc, e0, F):
    pap = compute_pap(A, tau, R, dt, mu, eta, pdc, F)
    Pdt = F*(1 - np.exp(-mu*eta) + pdc) * (1+F*pap)
    return (F*(1-np.exp(-mu*eta))*(e0 + F*pap*0.5) + F*pdc*0.5*(1 + F*pap)) / Pdt

PE_COEFF = 70
def compute_nsifted(nPP):
    return int((2*nPP + PE_COEFF**2 + np.sqrt((2*nPP+PE_COEFF**2)**2 - 4*nPP**2))/2)

def min_non_neg_skr(x):
    if x < 1.0: return 0.0
    else: return x

### end for simulation only ###


def compute_final_length(nZ, nZ1, nZ2, mZ1, mZ2, nX, nX1, nX2, mX1, mX2, mu1, mu2, p1, lEC, esec, ecor, K):
    """
    nZ : size of the raw key (in bits)
    nZ1 : size of the subset of the raw key from intensity mu1
    nZ2 : size of the subset of the raw key from intensity mu2
    mZ1 : number of errors detected in the subset of the raw key and of intensity mu1 (after error correction)
    mZ2 : number of errors detected in the subset of the raw key and of intensity mu2 (after error correction)
    nX : size of the key used for error estimation
    nX1 : size of the subset of key used for error estimation from intensity mu1
    nX2 : size of the subset of key used for error estimation from intensity mu2
    mX1 : number of errors detected in the subset of the key used for qber estimation and of intensity mu1
    mX2 : number of errors detected in the subset of the key used for qber estimation and of intensity mu2
    mu1 : first intenisty
    mu2 : second intensity, mu2 < mu1
    p1 : probability of choosing the intensity mu1
    lEC : leakage / length of the syndrome sent by Alice to correct the errors
    esec : security parameter (e.g esec = 10^-10)
    ecor : correctness parameter (e.g 10^-10)
    K: security parmater, K = 19 for 1 decoy state protocol

    Alice   Bob
          gc
        <---- 
      basis reconciliation  
        <---->
    nZ1,nZ2 
    nX1,nX2
         Xkey
        <----
    mX1
    mX2
    q_measured
       q_measured, syndrome(Zkey), nZ1,nZ2,mX1,mX2
        ---->
                (a) mZ, compute_final_length
                (b) needs Z1 Z2 to compute mZ1, mZ2, compute_final_length 
        <----

    """


    
    p2 = 1 - p1
    eps1 = esec / K

    t0_l = p1*np.exp(-mu1) + p2*np.exp(-mu2)
    t1_l = p1*np.exp(-mu1)*mu1 + p2*np.exp(-mu2)*mu2

    dnZ = delta(nZ, eps1); dnX = delta(nX, eps1)
    if dnX >= nX2 or dnZ >= nZ2:  return None
    
    #mX1 = nX1*E1; mX2 = nX2*E2; mX = mX1+mX2

    mX = mX1  + mX2
    mZ = mZ1 + mZ2
    dmX = delta(mX, eps1); dmZ = delta(mZ, eps1)
    if dmX >= mX2:  return None

    # Weighted Hoeffding counts
    nZ1pw = (np.exp(mu1)/p1)*(nZ1+dnZ)
    nZ2mw = (np.exp(mu2)/p2)*(nZ2-dnZ)
    nX1pw = (np.exp(mu1)/p1)*(nX1+dnX)
    nX2mw = (np.exp(mu2)/p2)*(nX2-dnX)
    mX1pw = (np.exp(mu1)/p1)*(mX1+dmX)
    mX2mw = (np.exp(mu2)/p2)*(mX2-dmX)
    
    # Rusca bounds
    # Solution (a)
    sz0u1 = 2*((t0_l*np.exp(mu1)/p1)*(mZ1+dmZ) + dnZ)
    sz0u2 = 2*((t0_l*np.exp(mu2)/p2)*(mZ2+dmZ) + dnZ)
    print(f"sz0u1 {sz0u1} sz0u2 {sz0u2}")
    sz0u = min(sz0u1, sz0u2)
    # Solution (b) (A.14)
    #sz0u = 2*(mZ + dnZ)

    sz0l_raw = (t0_l/(mu1-mu2))*(mu1*nZ2mw - mu2*nZ1pw)
    sz0l = max(sz0l_raw, 0.0)
    
    pref = t1_l*mu1/mu2*1/(mu1-mu2)
    term_d = nZ2mw
    term_s = -(mu2/mu1)**2 * nZ1pw
    term_v = -(mu1**2-mu2**2)/mu1**2 * (sz0u/t0_l)
    sz1l = max(pref*(term_d+term_s+term_v), 0.0)
    
    sz0uX1 = 2*((t0_l*np.exp(mu1)/p1)*(mX1+dmX) + dnX)
    sz0uX2 = 2*((t0_l*np.exp(mu2)/p2)*(mX2+dmX) + dnX)
    sz0uX=min(sz0uX1,sz0uX2)
    sx1l = max(pref * (nX2mw - (mu2/mu1)**2 * nX1pw
                      -(mu1**2-mu2**2)/mu1**2 * (sz0uX/t0_l)), 0.0)
    vx1u = max((t1_l/(mu1-mu2))*(mX1pw-mX2mw), 0.0)
    
    phi_raw = min(vx1u/sx1l, 0.5) if sx1l > 0 else 0.5
    
    # Rusca gamma smoothing
    if 0 < phi_raw < 0.5 and sz1l > 0 and sx1l > 0:
        arg = max(((sz1l+sx1l)/(sz1l*sx1l*(1-phi_raw)*phi_raw))
                  * (K**2/esec**2), 1.0)
        gam = np.sqrt((sz1l+sx1l)*(1-phi_raw)*phi_raw
                      /(sz1l*sx1l*np.log(2))*np.log2(arg))
    else:
        gam = 0.0
    phi = min(phi_raw+gam, 0.5)
    
    # Key length
    overhead = 6*np.log2(K/esec) + np.log2(2/ecor)
    ell = max(sz0l + sz1l*(1-hbin(phi)) - lEC - overhead, 0.0)
    
   
    return int(ell)


if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params_aurea.json'
    config_path = os.path.join(os.path.dirname(__file__), config_file)

    with open(config_path) as f:
        cfg = json.load(f)

    print(f"cfg : {json.dumps(cfg, indent=4)}")
    label    = cfg['label']

    # parameters of the protocol
    mu1      = cfg['mu1']
    mu2      = cfg['mu2']
    p1       = cfg['p1']
    pZ       = cfg['pZ']
    nZ       = cfg['nZ']
    esec     = cfg['esec']
    ecor     = cfg['ecor']
    fEC      = cfg['fEC']
    K        = cfg['K']


    # parameters for running estimation

    d_km = cfg['d_operating_km']
    coeff    = cfg.get('coeff', 1.0)  # dead-time correction factor (default 1.0)
    edet       = cfg['edet']
    dead_us    = cfg['dead_us']
    
    Protocol_symmetric = cfg['Protocol_symmetric']
    afterpulse_cfg = cfg.get('afterpulse', {})
    if afterpulse_cfg :
        A = afterpulse_cfg.get('A') 
        tau = afterpulse_cfg.get('tau')
        R = afterpulse_cfg.get('R')
    else: 
        A=[0,0,0]
        tau=[1,1,1]
        R=1  
    disable_pap = cfg['disable_pap']
    if disable_pap:
        # Honour the config flag: zero-amplitude afterpulse => pap contribution = 0.
        A = [0.0, 0.0, 0.0]
        tau = [1.0, 1.0, 1.0]


    eta_bob    = cfg['eta_bob']
    pdc        = cfg['pdc']
    alpha      = cfg['alpha']
    f_rep      = cfg['f_rep']
    odr_losses = cfg['odr_losses']

    eta = 10**(-(alpha*d_km+odr_losses)/10) * eta_bob

    # compute simulation
    
    if Protocol_symmetric:
        nS = compute_nsifted(nZ)
        nX = nS - nZ
    else:
        pX = 1.0 - pZ
        nX = nZ * (pX/pZ)**2

    # simple approximation
    #nZ1 = p1 * nZ
    #nZ2 = nZ - nZ1
    #nX1 = p1*nX 
    #nX2 = nX - nX1


    # precise simulation
    p2 = 1 - p1
    pap1 = compute_pap(A, tau, R, dead_us, mu1, eta, pdc, coeff)
    pap2 = compute_pap(A, tau, R, dead_us, mu2, eta, pdc, coeff)

    R1 =  (1 - np.exp(-mu1*eta) + pdc) * (1 + pap1)
    R2 =  (1 - np.exp(-mu2*eta) + pdc) * (1 + pap2)
    
    nZ1 = int(nZ*p1*R1/(p1*R1+p2*R2))
    nZ2 = int(nZ*p2*R2/(p1*R1+p2*R2))
    nX1 = int(nX*p1*R1/(p1*R1+p2*R2))
    nX2 = int(nX*p2*R2/(p1*R1+p2*R2))

    E1 = compute_qber(dead_us,mu1,eta,A,tau,R,pdc,edet,coeff)
    E2 = compute_qber(dead_us,mu2,eta,A,tau,R,pdc,edet,coeff)

    mX1 = E1*nX1
    mX2 = E2*nX2 
    mZ1 = E1*nZ1
    mZ2 = E2*nZ2  

    eobs = (mZ1+mZ2)/nZ
    lEC = fEC * hbin(eobs) * nZ

    ell = compute_final_length(nZ, nZ1, nZ2, mZ1, mZ2, nX, nX1, nX2, mX1, mX2, mu1, mu2, p1, lEC, esec, ecor, K)

    pdt = p1*R1+p2*R2
    cdt = coeff/(1 + f_rep*pdt*dead_us*1e-6)
    if Protocol_symmetric:
        Ntot = nS / (cdt * 0.5 * pdt)
    else:
        denom = cdt * pZ**2 * pdt
        Ntot = nZ / denom

    skr = min_non_neg_skr(ell * f_rep / Ntot) if Ntot > 0 else 0.0

    print(f"final length : {ell} skr = {skr}")
