import numpy as np
# from scipy.special import expi
import jax.numpy as jnp
import jax.lax as lax
from jax.scipy.special import expi
from jax import jit

from prosail import spectral_lib


@jit
def run_prospect(n, cab, car,  cbrown, cw, cm, ant=0.0, 
                 prospect_version="D",  
                 nr=None, kab=None, kcar=None, kbrown=None, kw=None, 
                 km=None, kant=None, alpha=40.):
    """The PROSPECT model, versions 5 and D"""
    
    if prospect_version == "5":
        # Call the original PROSPECT-5. In case the user has supplied 
        # spectra, use them.
        wv, refl, trans = prospect_d (n, cab, car, cbrown, cw, cm, 0.0,
                    spectral_lib.prospect5.nr if nr is None else nr,
                    spectral_lib.prospect5.kab if kab is None else kab,
                    spectral_lib.prospect5.kcar if kcar is None else kcar,
                    spectral_lib.prospect5.kbrown \
                        if kbrown is None else kbrown, 
                    spectral_lib.prospect5.kw if kw is None else kw,
                    spectral_lib.prospect5.km if km is None else km,
                    np.zeros_like(spectral_lib.prospect5.km), 
                    alpha=alpha)
    elif prospect_version.upper() == "D":
        wv, refl, trans = prospect_d (n, cab, car, cbrown, cw, cm, ant,
                    spectral_lib.prospectd.nr if nr is None else nr,
                    spectral_lib.prospectd.kab if kab is None else kab,
                    spectral_lib.prospectd.kcar if kcar is None else kcar,
                    spectral_lib.prospectd.kbrown \
                        if kbrown is None else kbrown,
                    spectral_lib.prospectd.kw if kw is None else kw,
                    spectral_lib.prospectd.km if km is None else km,
                    spectral_lib.prospectd.kant if kant is None else kant, 
                    alpha=alpha)
    else:
        raise ValueError("prospect_version can only be 5 or D!")

    return wv, refl, trans

@jit
def calctav(alpha, nr):
    # Constants
    n2  = nr * nr
    npx = n2 + 1
    nm  = n2 - 1
    a   = (nr + 1) ** 2 / 2.0
    k   = -(n2 - 1) ** 2 / 4.0
    sa  = jnp.sin(jnp.deg2rad(alpha))

    # Use jax.lax.cond for conditional computation
    b1 = lax.cond(
        alpha != 90,
        lambda _: jnp.sqrt((sa * sa - npx / 2) ** 2 + k),
        lambda _: 0.0,
        operand=None
    )

    b2 = sa * sa - npx / 2
    b  = b1 - b2
    b3 = b ** 3
    a3 = a ** 3

    # Compute ts
    ts = (k**2 / (6 * b3) + k / b - b / 2) - (k**2 / (6 * a3) + k / a - a / 2)

    # Compute tp components
    tp1 = -2 * n2 * (b - a) / (npx ** 2)
    tp2 = -2 * n2 * npx * jnp.log(b / a) / (nm ** 2)
    tp3 = n2 * (1 / b - 1 / a) / 2
    tp4 = (16 * n2 ** 2 * (n2 ** 2 + 1) * 
           jnp.log((2 * npx * b - nm ** 2) / (2 * npx * a - nm ** 2)) / 
           (npx ** 3 * nm ** 2))
    tp5 = (16 * n2 ** 3 * 
           (1 / (2 * npx * b - nm ** 2) - 1 / (2 * npx * a - nm ** 2)) / 
           (npx ** 3))
    
    tp  = tp1 + tp2 + tp3 + tp4 + tp5
    tav = (ts + tp) / (2 * sa ** 2)

    return tav

@jit
def refl_trans_one_layer (alpha, nr, tau):
    # ***********************************************************************
    # reflectance and transmittance of one layer
    # ***********************************************************************
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    # ***********************************************************************
    # reflectivity and transmissivity at the interface
    #-------------------------------------------------   
    talf = calctav (alpha,nr)
    ralf = 1.0-talf
    t12 = calctav (90,nr)
    r12 = 1. - t12
    t21 = t12/(nr*nr)
    r21 = 1-t21

    # top surface side
    denom = 1. - r21*r21*tau*tau
    Ta = talf*tau*t21/denom
    Ra = ralf + r21*tau*Ta

    # bottom surface side
    t = t12*tau*t21/denom
    r = r12+r21*tau*t
    
    return r, t, Ra, Ta, denom


@jit
def prospect_d(N, cab, car, cbrown, cw, cm, ant,
                   nr, kab, kcar, kbrown, kw, km, kant,
                   alpha=40.):

    lambdas = jnp.arange(400, 2501)  # wavelengths
    n_lambdas = lambdas.shape[0]

    # Ensure leaf spectra have the correct shape
    spectra_list = [nr, kab, kcar, kbrown, kw, km, kant]
    n_elems_list = jnp.array([spectrum.shape[0] for spectrum in spectra_list])

    if not jnp.all(n_elems_list == n_lambdas):
        raise ValueError("Leaf spectra don't have the right shape!")

    kall = (cab * kab + car * kcar + ant * kant + cbrown * kbrown +
            cw * kw + cm * km) / N

    # Compute tau using jax.lax.cond for branch efficiency
    t1 = (1 - kall) * jnp.exp(-kall)
    t2 = kall ** 2 * (-expi(-kall))
    tau = jnp.where(kall > 0, t1 + t2, jnp.ones_like(kall))

    # Call the reflectance/transmittance function (must also be converted to JAX)
    r, t, Ra, Ta, denom = refl_trans_one_layer(alpha, nr, tau)

    # Stokes equations for multi-layer calculations
    D = jnp.sqrt((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t))
    rq = r ** 2
    tq = t ** 2
    a       = (1+rq-tq+D)/(2*r)
    b       = (1-rq+tq+D)/(2*t)

    bNm1    = jnp.power(b, N-1)
    bN2     = bNm1*bNm1
    a2      = a*a
    denom   = a2*bN2-1
    Rsub    = a*(bN2-1)/denom
    Tsub    = bNm1*(a2-1)/denom

    # Case of zero absorption
    j       = r+t >= 1.
    Tsub[j] = t[j]/(t[j]+(1-t[j])*(N-1))
    Rsub[j] = 1-Tsub[j]

    # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
    denom   = 1-Rsub*r
    tran    = Ta*Tsub/denom
    refl    = Ra+Ta*Rsub*t/denom

    return lambdas, refl, tran