import numpy as np
import scipy as sp
import math as mt
from joblib import Parallel, delayed
import psutil
import warnings
from time import time

def frf_local_model(references,responses,abscissa,f_out=None,bandwidth=None,
                    transient=True,modelset=[[1,1,0],[1,1,1],[1,1,2]],
                    export_ratio=.1,max_parallel=512,print_time=True):
    """
    Local modeling FRF estimator for MIMO systems.
    This function can be called directly or using
    frf.timedata2frf(...,method='LRM') or frf.fft2frf(...,method='LRM').
    
    Parameters
    ----------
    references : ndarray (N_in, N_freqs) or (N_in, 1, N_freqs)
        REFERENCES FREQUENCY-DOMAIN DATA.
    responses : ndarray, shape (N_out, N_freqs) or (N_out, 1, N_freqs)
        RESPONSES FREQUENCY-DOMAIN DATA.
    abcissa : ndarray (N_freqs,)
        INPUT FREQUENCY VECTOR.
        
    f_out: ndarray (N_freqs_out,), optional
        Output frequency vector. Finer resolution increases computational cost,
        but only to a finite limit. Compuatational cost will not increase beyond
        f_out = f_in, but large interpolation may result in a large export_ratio.
        The default is f_in
    bandwidth: float, optional
        Local model estimation bandwidth in units of f_in. Larger values yield
        better noise filtering but risk underfitting and take longer to run.
    transient : boolean, optional
        whetheter to include transient estimation in local models. Recommend False
        for impact testing or burst random, True otherwise. The default is True.
    modelset : iterable of numpy (3,) arrays, optional
        Arrays contain [num. order, trans. order, denom. order].
        Recommend increasing numerator order for nonlinear structures
        and complicated FRFs. The default is [[1,1,0],[1,1,1],[1,1,2]].
    export_ratio : f;pat, optional
        (centered) proportion of local bands that can be stored and exported.
        Larger values risk capturing volatile end behavior but yield faster
        computation times. The default is 0.1.
    max_parallel: int, optional
        for very large systems with high frequency resolution and large bandwidths,
        system RAM can be a constraint. The default number of parallel processes
        is the number of CPU threads. If this causes freezing, max_parallel can
        be set to reduce the allowed number of parallel threads. Also, smaller
        pools may initialize faster. The default is 512.
    print_time: boolean, optional
        print remaining time if more than 10 seconds

    Returns
    -------
    f_out: ndarray (N_freqs_out,)
        estimated FRM frequency vector
    H: ndarray (N_freqs_out, N_out, N_in)
        frequency response matrix
    model_data: dict
        candidate model info and selected model at each frequency in f_out
       
        
    Notes
    -------
    To improve noise and transient removal, recommend increasing bandwidth
    
    To reduce computation time, recommend decreasing f_out resolution,
    increasing export_ratio, or decreasing bandwidth. For very large systems,
    if freezing occurs, reduce max_parallel.
    
    For more information, see "A practitioner's guide to local FRF estimation",
    K. Coletti. This function uses the MISO parameterization

    """
    
    """
    NOTES FOR MAINTAINER
    Outline of this function
    -------
        (1) f_in and f_out are parsed to generate an array of bands at which to
            estimate local models (bands) and a list of query points in each
            model at which to export solutions (query)
        (2) based on the model candidate set and problem size, functional
            parameterizations are generated which return linearized coefficient
            matrices (for solving) and the FRF matrix (for export)
        (3) computation time is estimated, and the LSQ algorithm is selected.
            Solving model parameters using interative LLSQ is the majority of
            computation time.
        (4) in parallel, local models are solved for each model in modelset and
            each frequency band. FRFs are exported. CTRL-F 'results = Parallel'
        (5) the winning model and corresponding FRM is returned in each band
        
    Notes on variable names
    -------
    U and Y are input and output data
    A,B,D are FRF numerator, transient numerator, and denom., respectively
    H is the frequency response matrix.
    LR is a function that produces linear coefficients (L) and remainder (R) for the local models
    See "A Practitioner’s Guide to Local FRF Estimation", K. Coletti for more information
    """
    
    #%% Process arguments
    U = references
    Y = responses
    f_in = abscissa
    del references; del responses; del abscissa
    
    df = float(f_in[1]-f_in[0]) # frequency step size
    if bandwidth is None: # local model bandwidth
        bandwidth = max( [min([300*df,40]) , max([30*df,10])] )
        # bounded below by the larger of 30 points and 10 Hz,
        # above by the smaller of 300 points and 40 Hz
    if f_out is None: # spacing between center frequencies for estimation
        f_out = f_in
    
    # Default parameters, hidden from user
    NitSK = 50 # max SK iterations
    conv_tol = .005 # SK parameter convergence tolerance
    
    #%% Calculate general helper vars
    Nfreq = f_in.size # number of frequencies
    Nin = U.shape[0]
    Nout = Y.shape[0]
    Nmod = len(modelset)
    U = U.reshape(Nin,Nfreq,order='F') # no realizations/frames needed
    Y = Y.reshape(Nout,Nfreq,order='F')
    f_in = f_in.reshape(-1,order='F')
    f_out = f_out.reshape(-1,order='F')
    
    if not Y.shape[-1] == U.shape[-1] == Nfreq:
        raise Exception('input arrays are not correctly sized')
    
    #%% Get estimation bands and query points (STEP 1)
    # Estimation bands contain the frequency bins used for parameter estimation,
    # and query points are the frequencies at which results are exported
    Nband = __round_odd(bandwidth/df) # number of data points in band
    N_half_band = int(Nband/2) # number of data points on either side of center
    max_from_center = export_ratio/2*bandwidth # maximum query distance from center point, except at ends
    
    bands,query = __parse_bands(N_half_band,max_from_center,f_in,f_out) # bands gives indices of f_in, and query gives r values for query. r=Hz/df.
    Ninds = bands.shape[0]    
    
    #%% Initialize parameterizations (STEP 2)
    LR,D,lsq_err,H = [[None]*Nmod,[None]*Nmod,[None]*Nmod,[None]*Nmod]
    Npar = np.zeros(Nmod)
    
    for k in range(Nmod-1,-1,-1):
        if transient:
            LR[k],D[k],lsq_err[k],H[k],Npar[k] = __parameterize(Nin,Nout,Nband,modelset[k])
        else:
            LR[k],D[k],lsq_err[k],H[k],Npar[k] = __parameterize_notrans(Nin,Nout,Nband,modelset[k])
            
        if Npar[k]>=Nband*Nout: # remove infeasible models
            warnings.warn('bandwidth too small for ' + str(modelset[k]) + ' model order')
            modelset.pop(k)
            Nmod -= 1
            LR.pop(k);D.pop(k);lsq_err.pop(k);H.pop(k);Npar = np.delete(Npar,k)
            
    dof = Nband*Nout-Npar
        
    #%% Compute (STEP 3/4)
    # Parallel function. # joblib can't parse lists (only arrays), so query is a parameter
    def __solve_ind(k,inq,lstsq):
        # Initialize. inq are radii of H to output in this band
        H_multi_model = np.empty((Nout,Nin,Nmod,inq.size),dtype=complex) # axis (0) output (1) input (2) model (3) frequency
        mdl = np.empty((Nmod,inq.size),dtype=complex)
        Uk = U[:,bands[k]]
        Yk = Y[:,bands[k]]
        
        for kk in range(Nmod): # model loop
            H_multi_model[:,:,kk,:],pstar = __solveLRM(Uk,Yk,LR[kk],D[kk],lsq_err[kk],H[kk],inq,conv_tol,NitSK,lstsq)
        
            # Model selection - MDL (see Pintelon, IEEE, 2021)
            mdl[kk,:] = pstar/dof[kk] * mt.exp( mt.log(2*Nband)*Npar[kk]/(dof[kk]-2) )
            
        return H_multi_model,mdl
    
    # Estimate time, choose algorithm - SEE APPENDIX NOTES 1 AND 2
    memory = psutil.virtual_memory().total/1e9
    n_parallel = min(psutil.cpu_count(),61,max_parallel)
    if memory/n_parallel < 2 and Nin*Nout > 500:
        warnings.warn('Many parallel processes relative to system RAM. For large problems, freezing may occur.')
    def test_lstsq_time(alg):
        start = time()
        __solve_ind( round(Ninds/2),query[round(Ninds/2)],alg )
        return time()-start
    if Nin*Nout < 500:
        time_numpy = test_lstsq_time(__lstsq_numpy)
    else:
        time_numpy = np.inf # numpy is very slow for large problems
    time_scipy = test_lstsq_time(__lstsq_scipy)
    
    if time_numpy < time_scipy:
        lstsq = __lstsq_numpy
    else:
        lstsq = __lstsq_scipy
        
    time_est = min(time_numpy,time_scipy)*Ninds/n_parallel
    if time_est > 10 and print_time: # parallel initialization takes 5 seconds
        print( 'Estimated time remaining: parallel init. + ' + str( round(1.1*time_est,1)) + ' seconds' )
    
    # Compute and extract results into arrays
    results = Parallel(n_jobs=n_parallel)(delayed(__solve_ind)(k,query[k],lstsq) for k in range(Ninds))
    H_multi_model = np.concatenate([x[0] for x in results],axis=3) # axis (0) Nout (1) Nin (2) model (3) Nind
    mdl = np.concatenate([x[1] for x in results],axis=1) # axis (0) model (1) Nind
            
    #%% Select winning models and export (STEP 5)
    msel = np.argmax(mdl,0) # axis. (0) freq. ind
    H = H_multi_model[:,:,msel,np.arange(f_out.size)] # axis. (0) Nout (1) Nin (2) freq. ind
    model_data = {'info': 'Model format is [H num. order, trans. num. order, denom. order].',
                  'modelset':modelset,
                  'model_selected':msel}
                
    # Return results
    return f_out, np.moveaxis(H,-1,0), model_data
  
#%% Local model solver
def __solveLRM(U,Y,LR,D,lsq_err,H,query,conv_tol,NitSK,lstsq):
    """
    Local model solver. This function occupies most of the FRF local modeling estimation
    time. Inputs are functions that accept parameter values and output local model component
    matrices used for least-squares solving and model exporting. Uses Sanathanan-Koerner
    iterations to solve the nonlinear rational LSTSQ problem iteratively
    """
    # SK iteration preparations
    L,R = LR(U,Y) # linearized transfer matrix and residual vector. Lin objective = D^{-1}*(L*t-R)
    Dp = np.ones(Y.shape) # initial scaling matrix. Axis (0) Nout (1) Nband
    t0s,lsq_min = [float('inf')]*2
    
    # Snathanan-Koererner Iterations
    for k in range(NitSK+5):
        # Define adjusted LLSQ/GLSQ problem
        X = L/Dp[:,:,np.newaxis] # left matrix in LLSQ stacked over r. Axis (0) Nout (1) Nband (2) Npar
        q = R/Dp # residual vector stacked over r. Axis (0) Nout (1) Nband
        
        # Solve and advance. Vast majority of compution time in all of get_frf
        t0 = lstsq(X,q) # axis. (0) Nout (1) Npar
    
        # Convergence
        if abs((t0-t0s)/t0).mean() < conv_tol:
            break # exit if convergence is reached
        else:
            Dp = D(t0) # store new adjustment matrix
        t0s = t0

        # Capture nonconvergence (in my experience, this is very rare)
        if k > NitSK:
            lsq_t0 = lsq_err(U,Y,t0)
            if lsq_t0 < lsq_min:
                lsq_min = lsq_t0
                t1 = t0
                
    t0=t1 if 't1' in locals() else t0
    
    return H(t0,query),-lsq_err(U,Y,t0) # Hstar, pstar (log density)

#%% Parameterizations. See "A Practitioner’s Guide to Local FRF Estimation", K. Coletti
def __parameterize(Nin,Nout,Nband,order):
    """
    Generate parameterization functions for estimation including transient removal
    Parameterizations are passed into __solveLRM to generate local models
    """
    # Components of f, because f can be iterated
    # Radius and exponent matrices. r is radius, s power, and A/B/D component matrices
    N_half_band = (Nband-1)/2
    r = np.arange(-N_half_band,N_half_band+1)
    rsA = r**np.arange(order[0]+1)[:,np.newaxis] # axis. (0) power (1) radius
    rsB = r**np.arange(order[1]+1)[:,np.newaxis]
    rsD = r**np.arange(order[2]+1)[:,np.newaxis]
    
    # Define theta break points
    bA = np.arange(Nout*Nin*(order[0]+1)) # transfer numerator
    bB = np.arange(bA[-1]+1,bA[-1]+Nout*(order[1]+1)+1) # transient numerator
    bD = np.arange(bB[-1]+1,bB[-1]+Nout*order[2]+1) # denominator (wrap Nout first, then s)
    Npar = bB[-1]+Nout*order[2]+1
    
    def lsq_err(U,Y,t):
        t = t.reshape(-1,order='F') # transform from SO format to MO
        
        # Build matrices
        A = np.einsum('ij,jk->ik',t[bA].reshape(-1,order[0]+1,order='F'),rsA).reshape(Nout,Nin,-1,order='F')
        B = np.einsum('ij,jk->ik',t[bB].reshape(-1,order[1]+1,order='F'),rsB) # axis (0) Nout (1) Nfreq
        catmat = np.concatenate( (np.ones((Nout,1)),t[bD][:,np.newaxis]),0 )
        D =  np.einsum('ij,jk->ik',catmat.reshape(Nout,-1,order='F'),rsD) # axis. (1) Nout (2) Nfreq
        
        # Compute sum of squared residuals
        Ysim = (np.einsum('jkl,kl->jl',A,U) + B)/D
        return (abs(Y-Ysim)**2).sum()
    
    def H(t,r_inq):
        t = t.reshape(-1,order='F') # transform from SO format to MO
        
        # Radius and exponent matrices
        rsA_inq = r_inq**np.arange(order[0]+1)[:,np.newaxis] # axis. (0) power (1) radius
        rsD_inq = r_inq**np.arange(order[2]+1)[:,np.newaxis]
        
        # Build matrices
        A = np.einsum('ij,jk->ik',t[bA].reshape(-1,order[0]+1,order='F'),rsA_inq).reshape(Nout,Nin,-1,order='F') # axis. (1) Nout (2) Nin (3) Nfreq
        temp = np.concatenate( (np.ones(Nout),t[bD]) )
        D = np.einsum('ij,jk->ik',temp.reshape(Nout,-1,order='F'),rsD_inq) # axis. (1) Nout (2) Nfreq
        
        return A/D[:,np.newaxis,:]

    # Build linearized (SK) transfer matrix L(U,y)*t + R(U,Y) = Dp^(-1) (D(t)*Y - N(t)*U - M(t))
    # Denominator matrix for SK iterations
    def D(t):
        return 1 + np.einsum( 'ij,lj->li',r[:,np.newaxis]**np.arange(1,order[2]+1) , t[:,np.arange(-order[2],0)] ) # axis (0) Nout (1) Nband
    
    def LR(U,Y):        
        # Compute residual
        R = -Y # axis (0) Nout (1) Nband
        
        # Compute components
        r1 = r[:,np.newaxis]
        A1 = Y.T.reshape(-1,1,Nout,order='F')* (r1**np.arange(1,order[2]+1))[:,:,np.newaxis] # D(t)*Y. Axis. (1) r (2) s (3) Nout
        A2 = (U.T[:,:,np.newaxis] * r1[:,:,np.newaxis] **np.arange(order[0]+1)).reshape(Nband,Nin*(order[0]+1),1,order='F') # A(t)*U. Axis. (1) r (2) Nin*s (3) Nout
        A3 = ( r1**np.arange(order[1]+1) )[:,:,np.newaxis] # B(t). Axis. (1) r (2) s (3) empty
        
        # Assemble into array
        if order[-1]>0:
            catmat = np.ones((1,1,Y.shape[0])) # A2 and A3 don't depend on Nout
            L = np.concatenate((-A2*catmat,-A3*catmat,A1),1).transpose(2,0,1) # axis (0) Nout (1) Nband (2) parameter
        else:
            L = np.concatenate((-A2,-A3),1).transpose(2,0,1) # axis (0) Nout (1) Nband (2) parameter
        
        return L,R

    return LR,D,lsq_err,H,Npar
def __parameterize_notrans(Nin,Nout,Nband,order):
    """
    Generate parameterization functions for estimation not including transient removal.
    Parameterizations are passed into __solveLRM to generate local models
    """
    # Components of f, because f can be iterated
    # Radius and exponent matrices. r is radius, s power, and A/B/D component matrices
    N_half_band = (Nband-1)/2
    r = np.arange(-N_half_band,N_half_band+1)
    rsA = r**np.arange(order[0]+1)[:,np.newaxis] # axis. (0) power (1) radius
    rsD = r**np.arange(order[2]+1)[:,np.newaxis]
    
    # Define theta break points
    bA = np.arange(Nout*Nin*(order[0]+1)) # transfer numerator
    bD = np.arange(bA[-1]+1,bA[-1]+Nout*order[2]+1) # denominator (wrap Nout first, then s)
    Npar = bA[-1]+Nout*order[2]+1
    
    def lsq_err(U,Y,t):
        t = t.reshape(-1,order='F') # transform from SO format to MO
        
        # Build matrices
        A = np.einsum('ij,jk->ik',t[bA].reshape(-1,order[0]+1,order='F'),rsA).reshape(Nout,Nin,-1,order='F')
        B = np.zeros((Nout,Nband)) # axis (0) Nout (1) Nfreq
        catmat = np.concatenate( (np.ones((Nout,1)),t[bD][:,np.newaxis]),0 )
        D =  np.einsum('ij,jk->ik',catmat.reshape(Nout,-1,order='F'),rsD) # axis. (1) Nout (2) Nfreq
        
        # Compute sum of squared residuals
        Ysim = (np.einsum('jkl,kl->jl',A,U) + B)/D
        return (abs(Y-Ysim)**2).sum()
    
    def H(t,r_inq):
        t = t.reshape(-1,order='F') # transform from SO format to MO
        
        # Radius and exponent matrices
        rsA_inq = r_inq**np.arange(order[0]+1)[:,np.newaxis] # axis. (0) power (1) radius
        rsD_inq = r_inq**np.arange(order[2]+1)[:,np.newaxis]
        
        # Build matrices
        A = np.einsum('ij,jk->ik',t[bA].reshape(-1,order[0]+1,order='F'),rsA_inq).reshape(Nout,Nin,-1,order='F') # axis. (1) Nout (2) Nin (3) Nfreq
        catmat = np.concatenate( (np.ones((Nout,1)),t[bD][:,np.newaxis]),0 )
        D = np.einsum('ij,jk->ik',catmat.reshape(Nout,-1,order='F'),rsD_inq) # axis. (1) Nout (2) Nfreq
        
        return A/D[:,np.newaxis,:]

    # Build linearized (SK) transfer matrix L(U,y)*t + R(U,Y) = Dp^(-1) (D(t)*Y - N(t)*U - M(t))
    # Denominator matrix for SK iterations
    def D(t):
        return 1 + np.einsum( 'ij,lj->li',r[:,np.newaxis]**np.arange(1,order[2]+1) , t[:,np.arange(-order[2],0)] ) # axis (0) Nout (1) Nband
    
    def LR(U,Y):        
        # Compute residual
        R = -Y # axis (0) Nout (1) Nband
        
        # Compute components
        r1 = r[:,np.newaxis]
        A1 = Y.T.reshape(-1,1,Nout,order='F')* (r1**np.arange(1,order[2]+1))[:,:,np.newaxis] # D(t)*Y. Axis. (1) r (2) s (3) Nout
        A2 = (U.T[:,:,np.newaxis] * r1[:,:,np.newaxis] **np.arange(order[0]+1)).reshape(Nband,Nin*(order[0]+1),1,order='F') # A(t)*U. Axis. (1) r (2) Nin*s (3) Nout
        
        # Assemble into array
        if order[-1]>0:
            catmat = np.ones((1,1,Y.shape[0])) # A2 and A3 don't depend on Nout
            L = np.concatenate((-A2*catmat,A1),1).transpose(2,0,1) # axis (0) Nout (1) Nband (2) parameter
        else:
            L = -A2.transpose(2,0,1) # axis (0) Nout (1) Nband (2) parameter
        
        return L,R

    return LR,D,lsq_err,H,Npar

#%% Frequency parser
def __parse_bands(N_half_band,max_from_center,f_in,f_out):
    """
    Local modeling is not constrained to the input frequency vector. This function
    processes input parameters and parses f_in into bands to model (bands).
    In addition, it returns query points (query) at which to export each
    local model
    """
    df_out = f_out[1]-f_out[0] # query point step size
    df_in = f_in[1]-f_in[0] # input frequencies steps size
    inq_per_band = mt.floor(2*max_from_center/df_out) + 1 # query points per band
    
    # Assign export indices to band numbers
    n_lo = (f_out <= f_in[N_half_band]+max_from_center).sum()
    n_hi = (f_out >= f_in[-N_half_band-1]-max_from_center).sum()
    n_mid = f_out.size - n_lo - n_hi
    
    # Parsing is different if n_lo and n_hi > 0
    is_lo = n_lo > 0
    is_hi = n_hi > 0
    
    # Get band assignment numbers for each entry in f_out
    temp = np.ones((inq_per_band,1))*np.arange(is_lo,mt.ceil(n_mid/inq_per_band)+1)
    temp = temp.reshape(-1,order='F')[:n_mid]
    inq_assign = np.concatenate(( np.zeros(n_lo) , temp , (temp.max()+1)*np.ones(n_hi) ))
    inq_assign = inq_assign.astype(dtype='int32')
    
    # Get indices of center points and build bands
    c_inds_hz = np.array([f_out[inq_assign==k].mean() for k in range(is_lo,inq_assign.max()+1-is_hi)]) # ideal center locations of each export
    c_inds = ((c_inds_hz-f_in[0])/df_in).round() # actual center locations of each export on f_in
    if abs(c_inds_hz - f_in[c_inds.astype('int32')]).max() > max_from_center:
        warnings.warn("""f_in is too sparse for f_out specification, resulting in LM extrapolation 
                      beyond the export_ratio. Consider using default specification or increasing bandwidth.""")
    lo_ind = np.array([N_half_band]) if is_lo else np.ones(0)
    hi_ind = np.array([f_in.size-1-N_half_band]) if is_hi else np.ones(0)
    c_inds = np.concatenate(( lo_ind,c_inds,hi_ind )).astype(dtype='int32')
    bands = np.arange(-N_half_band,N_half_band+1) + c_inds[:,np.newaxis] # axis (0) center ind (1) r
    bands = bands.astype(dtype='int32')
    
    # Get normalized extraction radii
    query = [(f_out[inq_assign==k] - ind)/df_in for k,ind in enumerate(f_in[c_inds])]
    
    return bands,query
    
#%% General functions
def __round_odd(x):
    y = int(x)
    return y + int(y%2==0)
def __lstsq_scipy(A,b): # extension of np.linalg.lstsq to 3-D arrays. Not vectorized.
    X = np.empty( np.array(A.shape)[[0,-1]] , dtype=complex ) # axis. (0) Nout [1] Npar
    for k,Ak in enumerate(A):
        X[k] = sp.linalg.lstsq(Ak,b[k])[0]
    return X
def __lstsq_numpy(A,b): # extension of np.linalg.lstsq to 3-D arrays. Not vectorized.
    X = np.empty( np.array(A.shape)[[0,-1]] , dtype=complex ) # axis. (0) Nout [1] Np
    for k,Ak in enumerate(A):
        X[k] = np.linalg.lstsq(Ak,b[k],rcond=None)[0]
    return X

"""
APPENDIX
Note 1: Windows OS restricts parallel processes to 61

Note 2: linalg.lstsq in scipy and numpy vary wildly in relative speed. In some cases,
numpy is 10x faster. For other problem sizes, scipy is 20x faster. Both are tested,
and the faster algorithm is selected.
"""