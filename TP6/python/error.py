import numpy as np

def endpoint_error(wobs, wgt):
    '''compute the end point error mean and std between 
    two optical_flow vectors - i.e L1'''
    epe = np.abs(wobs-wgt)
    return epe.mean(), epe.std()

def angular_error(wobs, wgt):
    '''e.g with angular error'''
    uobs = wobs[:,:,0]
    vobs = wobs[:,:,1]
    ugt = wgt[:,:,0]
    vgt = wgt[:,:,1]

    error = 1 + uobs*ugt + vobs*vgt
    divisor = np.sqrt( 1+np.power(ugt,2) + np.power(vgt,2))*(np.sqrt( 1 + np.power(uobs,2) + np.power(vobs,2)))
    ang_err = np.rad2deg(np.arccos(np.round(error/divisor, 6)))
    return np.mean(ang_err), np.std(ang_err)
