import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax.scipy.special import expi
from jax import jit
from functools import partial

from prosail import spectral_lib

# Precompute wavelengths as a global constant.
WAVELENGTHS = jnp.arange(400, 2501)
N_LAMBDAS = WAVELENGTHS.shape[0]


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
                    spectral_lib.prospect5.kbrown if kbrown is None else kbrown, 
                    spectral_lib.prospect5.kw if kw is None else kw,
                    spectral_lib.prospect5.km if km is None else km,
                    np.zeros_like(spectral_lib.prospect5.km), 
                    alpha=alpha)
    elif prospect_version.upper() == "D":
        wv, refl, trans = prospect_d (n, cab, car, cbrown, cw, cm, ant,
                    spectral_lib.prospectd.nr if nr is None else nr,
                    spectral_lib.prospectd.kab if kab is None else kab,
                    spectral_lib.prospectd.kcar if kcar is None else kcar,
                    spectral_lib.prospectd.kbrown if kbrown is None else kbrown,
                    spectral_lib.prospectd.kw if kw is None else kw,
                    spectral_lib.prospectd.km if km is None else km,
                    spectral_lib.prospectd.kant if kant is None else kant, 
                    alpha=alpha)
    else:
        raise ValueError("prospect_version can only be 5 or D!")

    return wv, refl, trans

@partial(jit, static_argnums=(0,))
# @jit(static_argnums=(0,))
def calctav(alpha, nr):
    """
    Computes the TAV (transmittance) of a leaf layer given angle alpha (deg) 
    and refractive index nr. Assumes alpha is a compile-time constant.
    """
    n2  = nr * nr
    npx = n2 + 1
    nm  = n2 - 1
    a   = (nr + 1) ** 2 / 2.0
    k   = -(n2 - 1) ** 2 / 4.0

    sa  = jnp.sin(jnp.deg2rad(alpha))
    # Since alpha is static, we can use a Python if/else.
    if alpha != 90:
        sqrt_expr = jnp.sqrt((sa * sa - npx / 2) ** 2 + k)
        b1 = sqrt_expr
    else:
        b1 = 0.0

    b2 = sa * sa - npx / 2
    b  = b1 - b2
    b3 = b ** 3
    a3 = a ** 3

    ts = (k**2 / (6.0 * b3) + k / b - b / 2.0) - (k**2 / (6.0 * a3) + k / a - a / 2.0)

    tp1 = -2.0 * n2 * (b - a) / (npx ** 2)
    tp2 = -2.0 * n2 * npx * jnp.log(b / a) / (nm ** 2)
    tp3 = n2 * (1.0 / b - 1.0 / a) / 2.0
    tp4 = (16.0 * n2**2 * (n2**2 + 1.0) *
           jnp.log((2.0 * npx * b - nm**2) / (2.0 * npx * a - nm**2)) /
           (npx**3 * nm**2))
    tp5 = (16.0 * n2**3 *
           (1.0 / (2.0 * npx * b - nm**2) - 1.0 / (2.0 * npx * a - nm**2)) /
           (npx**3))

    tp  = tp1 + tp2 + tp3 + tp4 + tp5

    tav = (ts + tp) / (2.0 * sa**2)
    return tav


@jit  # Optionally jitted for efficiency
def refl_trans_one_layer(alpha, nr, tau):
    """
    Reflectance and transmittance of one layer.
    """
    talf = calctav(40, nr)
    ralf = 1.0 - talf
    t12  = calctav(90, nr)
    r12  = 1.0 - t12
    t21  = t12 / (nr * nr)
    r21  = 1.0 - t21

    denom = 1.0 - r21 * r21 * tau * tau
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta

    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t

    return r, t, Ra, Ta, denom


def prospect_d_check_shapes(spectra_list, expected_size):
    """
    Check that each spectrum in spectra_list has shape == (expected_size,).
    Raises ValueError if not.
    """
    for spectrum in spectra_list:
        if spectrum.shape[0] != expected_size:
            raise ValueError("Leaf spectra don't have the right shape!")


@partial(jit, static_argnums=(14,))
# @jit(static_argnums=(14,))
def prospect_d_jit(N, cab, car, cbrown, cw, cm, ant,
                   nr, kab, kcar, kbrown, kw, km, kant,
                   alpha=40.0):
    """
    JIT-compatible version of PROSPECT-D.
    Assumes spectra arrays all have the correct shape (2101) beforehand.
    `alpha` is a static argument.
    """
    # Use precomputed wavelengths.
    lambdas = WAVELENGTHS
    n_lambdas = N_LAMBDAS

    kall = (cab * kab + car * kcar + ant * kant +
            cbrown * kbrown + cw * kw + cm * km) / N

    t1 = (1.0 - kall) * jnp.exp(-kall)
    t2 = kall**2 * (-expi(-kall))
    tau = jnp.where(kall > 0.0, t1 + t2, jnp.ones_like(kall))

    r, t, Ra, Ta, denom = refl_trans_one_layer(alpha, nr, tau)

    D  = jnp.sqrt((1.0 + r + t) * (1.0 + r - t) *
                  (1.0 - r + t) * (1.0 - r - t))
    rq = r**2
    tq = t**2
    a_val = (1.0 + rq - tq + D) / (2.0 * r)
    b_val = (1.0 - rq + tq + D) / (2.0 * t)

    bNm1 = b_val**(N - 1)
    bN2  = bNm1 * bNm1
    a2   = a_val * a_val
    denom2 = a2 * bN2 - 1.0
    Rsub = a_val * (bN2 - 1.0) / denom2
    Tsub = bNm1 * (a2 - 1.0) / denom2

    # Case of zero absorption: where (r + t) >= 1
    j = (r + t) >= 1.0
    new_Tsub = t / (t + (1.0 - t) * (N - 1))
    Tsub = jnp.where(j, new_Tsub, Tsub)
    Rsub = jnp.where(j, 1.0 - new_Tsub, Rsub)

    denom3 = 1.0 - Rsub * r
    tran = Ta * Tsub / denom3
    refl = Ra + Ta * Rsub * t / denom3

    return lambdas, refl, tran


def prospect_d(
    N, cab, car, cbrown, cw, cm, ant,
    nr, kab, kcar, kbrown, kw, km, kant,
    alpha=40.0
):
    """
    Public-facing function that first checks shapes, then calls the JIT function.
    """
    # 1) Check shapes (outside the JIT context).
    spectra_list = [nr, kab, kcar, kbrown, kw, km, kant]
    prospect_d_check_shapes(spectra_list, N_LAMBDAS)

    # 2) Call the jitted version.
    return prospect_d_jit(
        N, cab, car, cbrown, cw, cm, ant,
        nr, kab, kcar, kbrown, kw, km, kant, alpha
    )



# import numpy as np
# # from scipy.special import expi
# import jax.numpy as jnp
# import jax.lax as lax
# from jax.scipy.special import expi
# from jax import jit

# from prosail import spectral_lib



# def run_prospect(n, cab, car,  cbrown, cw, cm, ant=0.0, 
#                  prospect_version="D",  
#                  nr=None, kab=None, kcar=None, kbrown=None, kw=None, 
#                  km=None, kant=None, alpha=40.):
#     """The PROSPECT model, versions 5 and D"""
    
#     if prospect_version == "5":
#         # Call the original PROSPECT-5. In case the user has supplied 
#         # spectra, use them.
#         wv, refl, trans = prospect_d (n, cab, car, cbrown, cw, cm, 0.0,
#                     spectral_lib.prospect5.nr if nr is None else nr,
#                     spectral_lib.prospect5.kab if kab is None else kab,
#                     spectral_lib.prospect5.kcar if kcar is None else kcar,
#                     spectral_lib.prospect5.kbrown \
#                         if kbrown is None else kbrown, 
#                     spectral_lib.prospect5.kw if kw is None else kw,
#                     spectral_lib.prospect5.km if km is None else km,
#                     np.zeros_like(spectral_lib.prospect5.km), 
#                     alpha=alpha)
#     elif prospect_version.upper() == "D":
#         wv, refl, trans = prospect_d (n, cab, car, cbrown, cw, cm, ant,
#                     spectral_lib.prospectd.nr if nr is None else nr,
#                     spectral_lib.prospectd.kab if kab is None else kab,
#                     spectral_lib.prospectd.kcar if kcar is None else kcar,
#                     spectral_lib.prospectd.kbrown \
#                         if kbrown is None else kbrown,
#                     spectral_lib.prospectd.kw if kw is None else kw,
#                     spectral_lib.prospectd.km if km is None else km,
#                     spectral_lib.prospectd.kant if kant is None else kant, 
#                     alpha=alpha)
#     else:
#         raise ValueError("prospect_version can only be 5 or D!")

#     return wv, refl, trans


# @jit
# def calctav(alpha, nr):
#     """
#     Computes the TAV (transmittance) of a leaf layer given angle alpha (deg) 
#     and refractive index nr. Works for scalar or array inputs of matching shape.
#     """

#     # Squares and combined constants
#     n2  = nr * nr
#     npx = n2 + 1
#     nm  = n2 - 1
#     a   = (nr + 1) ** 2 / 2.0
#     k   = -(n2 - 1) ** 2 / 4.0

#     # Sine of alpha (handling scalar or array alpha)
#     sa  = jnp.sin(jnp.deg2rad(alpha))

#     # Instead of lax.cond, do an elementwise where for alpha != 90
#     # This way, if alpha is an array, we produce an array consistently.
#     expr = (sa * sa - npx / 2) ** 2 + k
#     sqrt_expr = jnp.sqrt(expr)

#     # b1 is sqrt_expr if alpha != 90, else 0.0
#     # jnp.where(...) returns the same shape as alpha/nr.
#     b1 = jnp.where(alpha != 90, sqrt_expr, 0.0)

#     b2 = sa * sa - npx / 2
#     b  = b1 - b2
#     b3 = b ** 3
#     a3 = a ** 3

#     # ts
#     ts = (k**2 / (6.0 * b3) + k / b - b / 2.0) - (
#         k**2 / (6.0 * a3) + k / a - a / 2.0
#     )

#     # tp
#     tp1 = -2.0 * n2 * (b - a) / (npx ** 2)
#     tp2 = -2.0 * n2 * npx * jnp.log(b / a) / (nm ** 2)
#     tp3 = n2 * (1.0 / b - 1.0 / a) / 2.0
#     tp4 = (16.0 * n2**2 * (n2**2 + 1.0) *
#            jnp.log((2.0 * npx * b - nm**2) / (2.0 * npx * a - nm**2)) /
#            (npx**3 * nm**2))
#     tp5 = (16.0 * n2**3 *
#            (1.0 / (2.0 * npx * b - nm**2) - 1.0 / (2.0 * npx * a - nm**2)) /
#            (npx**3))

#     tp  = tp1 + tp2 + tp3 + tp4 + tp5

#     # Final TAV
#     tav = (ts + tp) / (2.0 * sa**2)

#     return tav


# @jit
# def refl_trans_one_layer (alpha, nr, tau):
#     # ***********************************************************************
#     # reflectance and transmittance of one layer
#     # ***********************************************************************
#     # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
#     # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
#     # Soc. Am., 59(10):1376-1379.
#     # ***********************************************************************
#     # reflectivity and transmissivity at the interface
#     #-------------------------------------------------   
#     talf = calctav (alpha,nr)
#     ralf = 1.0-talf
#     t12 = calctav (90,nr)
#     r12 = 1. - t12
#     t21 = t12/(nr*nr)
#     r21 = 1-t21

#     # top surface side
#     denom = 1. - r21*r21*tau*tau
#     Ta = talf*tau*t21/denom
#     Ra = ralf + r21*tau*Ta

#     # bottom surface side
#     t = t12*tau*t21/denom
#     r = r12+r21*tau*t
    
#     return r, t, Ra, Ta, denom



# def prospect_d_check_shapes(spectra_list, expected_size):
#     """
#     Check that each spectrum in spectra_list has shape == (expected_size,).
#     Raises ValueError if not.
#     """
#     for spectrum in spectra_list:
#         if spectrum.shape[0] != expected_size:
#             raise ValueError("Leaf spectra don't have the right shape!")


# @jit
# def prospect_d_jit(N, cab, car, cbrown, cw, cm, ant,
#                    nr, kab, kcar, kbrown, kw, km, kant,
#                    alpha=40.0):
#     """
#     JIT-compatible version of PROSPECT-D.
#     Assumes spectra arrays all have the correct shape (2101) beforehand.
#     """
#     # Wavelengths: shape (2101,)
#     lambdas = jnp.arange(400, 2501)
#     n_lambdas = lambdas.shape[0]

#     # Sum of absorption coefficients
#     kall = (cab * kab + car * kcar + ant * kant
#             + cbrown * kbrown + cw * kw + cm * km) / N

#     # Transmittance due to absorption
#     # "Case of zero absorption" handled via jnp.where
#     t1 = (1.0 - kall) * jnp.exp(-kall)
#     t2 = kall**2 * (-expi(-kall))  # uses the custom expi above
#     tau = jnp.where(kall > 0.0, t1 + t2, jnp.ones_like(kall))

#     # Single-layer reflection/transmittance
#     r, t, Ra, Ta, denom = refl_trans_one_layer(alpha, nr, tau)

#     # Stokes equations for multi-layer
#     D = jnp.sqrt((1.0 + r + t) * (1.0 + r - t)
#                  * (1.0 - r + t) * (1.0 - r - t))
#     rq = r**2
#     tq = t**2
#     a = (1.0 + rq - tq + D) / (2.0 * r)
#     b = (1.0 - rq + tq + D) / (2.0 * t)

#     bNm1 = b**(N - 1)
#     bN2 = bNm1 * bNm1
#     a2 = a * a
#     denom2 = a2 * bN2 - 1.0
#     Rsub = a * (bN2 - 1.0) / denom2
#     Tsub = bNm1 * (a2 - 1.0) / denom2

#     # "Case of zero absorption": where (r + t) >= 1
#     j = (r + t) >= 1.0
#     # new Tsub for the 'j' positions
#     new_Tsub = t / (t + (1.0 - t) * (N - 1))
#     # update Tsub elementwise
#     Tsub = jnp.where(j, new_Tsub, Tsub)
#     # update Rsub accordingly
#     Rsub = jnp.where(j, 1.0 - new_Tsub, Rsub)

#     # Combine layers
#     denom3 = 1.0 - Rsub * r
#     tran = Ta * Tsub / denom3
#     refl = Ra + Ta * Rsub * t / denom3

#     return lambdas, refl, tran

# # Example usage:
# def prospect_d(
#     N, cab, car, cbrown, cw, cm, ant,
#     nr, kab, kcar, kbrown, kw, km, kant,
#     alpha=40.0
# ):
#     """
#     Public-facing function that first checks shapes, then calls the JIT function.
#     """
#     # 1) Make sure shapes are correct (not inside JIT).
#     #    We expect shape=(2101,) based on lambdas = arange(400,2501).
#     n_lambdas = 2101
#     spectra_list = [nr, kab, kcar, kbrown, kw, km, kant]
#     prospect_d_check_shapes(spectra_list, n_lambdas)

#     # 2) Call the jitted version
#     return prospect_d_jit(
#         N, cab, car, cbrown, cw, cm, ant,
#         nr, kab, kcar, kbrown, kw, km, kant, alpha
#     )

# # def prospect_d(N, cab, car, cbrown, cw, cm, ant,
# #                    nr, kab, kcar, kbrown, kw, km, kant,
# #                    alpha=40.):

# #     lambdas = jnp.arange(400, 2501)  # wavelengths
# #     n_lambdas = lambdas.shape[0]

# #     # Ensure leaf spectra have the correct shape
# #     spectra_list = [nr, kab, kcar, kbrown, kw, km, kant]
# #     n_elems_list = jnp.array([spectrum.shape[0] for spectrum in spectra_list])

# #     if not jnp.all(n_elems_list == n_lambdas):
# #         raise ValueError("Leaf spectra don't have the right shape!")

# #     kall = (cab * kab + car * kcar + ant * kant + cbrown * kbrown +
# #             cw * kw + cm * km) / N

# #     # Compute tau using jax.lax.cond for branch efficiency
# #     t1 = (1 - kall) * jnp.exp(-kall)
# #     t2 = kall ** 2 * (-expi(-kall))
# #     tau = jnp.where(kall > 0, t1 + t2, jnp.ones_like(kall))

# #     # Call the reflectance/transmittance function (must also be converted to JAX)
# #     r, t, Ra, Ta, denom = refl_trans_one_layer(alpha, nr, tau)

# #     # Stokes equations for multi-layer calculations
# #     D = jnp.sqrt((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t))
# #     rq = r ** 2
# #     tq = t ** 2
# #     a       = (1+rq-tq+D)/(2*r)
# #     b       = (1-rq+tq+D)/(2*t)

# #     bNm1    = jnp.power(b, N-1)
# #     bN2     = bNm1 * bNm1
# #     a2      = a * a
# #     denom   = a2 * bN2 - 1
# #     Rsub    = a*(bN2 - 1)/denom
# #     Tsub    = bNm1*(a2 - 1)/denom

# #     # "Case of zero absorption" => We used to do:
# #     # Tsub[j] = t[j]/(t[j]+(1-t[j])*(N-1))  etc.

# #     # Instead, define the boolean mask "j"
# #     j = (r + t) >= 1.

# #     # We'll build new Tsub for the 'j' positions:
# #     new_Tsub = t / (t + (1 - t) * (N - 1))
# #     # Then update Tsub elementwise
# #     Tsub = jnp.where(j, new_Tsub, Tsub)

# #     # Then update Rsub accordingly
# #     Rsub = jnp.where(j, 1 - new_Tsub, Rsub)

# #     # Now continue:
# #     denom = 1 - Rsub * r
# #     tran = Ta * Tsub / denom
# #     refl = Ra + Ta * Rsub * t / denom

# #     return lambdas, refl, tran
