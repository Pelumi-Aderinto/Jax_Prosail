import jax
import jax.numpy as jnp

from functools import partial

# ---------------------------------------------------------------------
from prosail import spectral_lib      # for soil spectra, etc.
from .prospect_d import run_prospect_d  # The refactored function from your code
from .FourSAIL import foursail

# Utility function to handle "factor" picking
def _pick_factor(factor_str, all_outputs):
    """
    all_outputs is the 21-element tuple returned by foursail.
    We want to return either rsot (SDR), rddt (BHR), rsdt (DHR), rdot (HDR), or ALL.
    Indices in the tuple:
      #  0:tss, 1:too, 2:tsstoo,
      #  3:rdd, 4:tdd, 5:rsd, 6:tsd, 7:rdo, 8:tdo,
      #  9:rso, 10:rsos, 11:rsod,
      # 12:rddt, 13:rsdt, 14:rdot, 15:rsodt, 16:rsost, 17:rsot,
      # 18:gammasdf, 19:gammasdb, 20:gammaso
    # So we need:
    #  SDR -> index 17: rsot
    #  BHR -> index 12: rddt
    #  DHR -> index 13: rsdt
    #  HDR -> index 14: rdot
    """
    factor_str = factor_str.upper()
    if factor_str not in ["SDR", "BHR", "DHR", "HDR", "ALL"]:
        raise ValueError("'factor' must be one of SDR, BHR, DHR, HDR or ALL")

    if factor_str == "SDR":
        return all_outputs[17]  # rsot
    elif factor_str == "BHR":
        return all_outputs[12]  # rddt
    elif factor_str == "DHR":
        return all_outputs[13]  # rsdt
    elif factor_str == "HDR":
        return all_outputs[14]  # rdot
    else:
        # "ALL"
        return jnp.stack(
            [all_outputs[17], all_outputs[12], all_outputs[13], all_outputs[14]],
            axis=0
        )  # shape (4, 2101) or something similar.


# ---------------------------------------------------------------------
# 3) Single JAX function that merges:
#    - PROSPECT leaf optical
#    - Soil reflectance mix
#    - SAIL
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnums=(7, 26, 27))
def _run_prosail_sail_jax_inner(
    n, cab, car, cbrown, cw, cm, ant,
    alpha,  # static if you want
    # Soil brightness/moisture
    rsoil, psoil,
    # Leaf angle, hot-spot
    lai, lidfa, lidfb, hspot,
    # geometry
    tts, tto, psi,
    # leaf & soil spectra
    nr, kab, kcar_, kbrown, kw, km, kant,
    soil_spectrum1, soil_spectrum2,
    # type of lidf, factor
    lidftype, factor_str
):
    """
    Single *compiled* function that:
      1) Calls PROSPECT-D to get leaf reflectance/transmittance (2101-long).
      2) Mixes the soil reflectance (2101-long).
      3) Calls FOURSAIL for canopy reflectance (2101-long).
      4) Picks the user's 'factor' to return.

    Note:
      - This function expects all arrays to be jnp arrays already.
      - We do no shape checking inside. 
      - We rely on the outer Python function to handle "if rsoil0 is None" etc.
    """

    # 1) Leaf reflectance/trans (2101) using your run_prospect_d
    #    (Or directly call prospect_d_core_jit if you prefer.)
    _, leaf_refl, leaf_trans = run_prospect_d(
        N=n,
        cab=cab, car=car, cbrown=cbrown, cw=cw, cm=cm, ant=ant,
        nr=nr, kab=kab, kcar=kcar_, kbrown=kbrown,
        kw=kw, km=km, kant=kant,
        alpha=alpha
    )

    # 2) Soil reflectance (2101)
    #    rsoil*(psoil*soil_spectrum1 + (1-psoil)*soil_spectrum2)
    #    We'll do that in a single line:
    soil_ref = rsoil * (psoil * soil_spectrum1 + (1.0 - psoil)*soil_spectrum2)

    # 3) Call foursail:
    #    foursail(rho, tau, lidfa, lidfb, lidftype, lai, hspot, tts, tto, psi, rsoil)
    all_sail_outputs = foursail(
        leaf_refl, leaf_trans,
        lidfa, lidfb, lidftype,
        lai, hspot,
        tts, tto, psi,
        soil_ref
    )

    # 4) Pick the factor
    result = _pick_factor(factor_str, all_sail_outputs)  # shape could be (2101,) or (4,2101)
    return result


def run_prosail_sail_jax(
    n, cab, car, cbrown, cw, cm, lai, lidfa, lidfb, hspot,
    tts, tto, psi,
    alpha=40.0,
    prospect_version="D",
    ant=0.0,
    lidftype=2,
    factor="SDR",
    rsoil0=None, rsoil=None, psoil=None,
    soil_spectrum1=None, soil_spectrum2=None
):
    """
    Public-facing function that:
      1) Checks shapes / sets defaults for soils.
      2) Prepares any needed PROSPECT spectral data (nr, kab, etc.)
      3) Calls the single JIT-compiled `_run_prosail_sail_jax_inner`.
      4) Returns a (2101,) array or a (4, 2101) array if factor='ALL'.

    This merges your old `run_prosail()` logic with the new single-jit approach.
    """

    # 0) Standardize factor
    factor = factor.upper()
    if factor not in ["SDR", "BHR", "DHR", "HDR", "ALL"]:
        raise ValueError("'factor' must be one of SDR, BHR, DHR, HDR, ALL")

    # 1) Soil spectra: default or user-provided
    if soil_spectrum1 is not None:
        if len(soil_spectrum1) != 2101:
            raise ValueError("soil_spectrum1 must be 2101 long!")
        soil_spec1 = jnp.array(soil_spectrum1, dtype=jnp.float32)
    else:
        soil_spec1 = jnp.array(spectral_lib.soil.rsoil1, dtype=jnp.float32)

    if soil_spectrum2 is not None:
        if len(soil_spectrum2) != 2101:
            raise ValueError("soil_spectrum2 must be 2101 long!")
        soil_spec2 = jnp.array(soil_spectrum2, dtype=jnp.float32)
    else:
        soil_spec2 = jnp.array(spectral_lib.soil.rsoil2, dtype=jnp.float32)

    # If user directly provides a rsoil0 (2101 vector), we skip mixing?
    # But your code suggests that if rsoil0 is not None, you just use it as soil reflectance.
    # We'll replicate your logic:
    if rsoil0 is not None:
        # we assume rsoil0 is 2101 long
        if len(rsoil0) != 2101:
            raise ValueError("rsoil0 must be length 2101 if provided!")
        # For consistency with the single pipeline, let's handle it as well:
        # We'll just pass rsoil=1, psoil=1, soil_spectrum1=rsoil0, soil_spectrum2=0
        rsoil_val = 1.0
        psoil_val = 1.0
        soil_spec1 = jnp.array(rsoil0, dtype=jnp.float32)
        soil_spec2 = jnp.zeros_like(soil_spec1)
    else:
        # user must have provided rsoil & psoil
        if (rsoil is None) or (psoil is None):
            raise ValueError("If rsoil0 is not defined, then rsoil and psoil must be defined.")
        rsoil_val = float(rsoil)
        psoil_val = float(psoil)

    # 2) We need the correct PROSPECT spectral data:
    #    If version='5', we use spectral_lib.prospect5.*,
    #    if version='D', we use spectral_lib.prospectd.* .
    #    We do shape checks in Python (2101).
    if prospect_version.upper() == "5":
        nr_     = jnp.array(spectral_lib.prospect5.nr, dtype=jnp.float32)
        kab_    = jnp.array(spectral_lib.prospect5.kab, dtype=jnp.float32)
        kcar_   = jnp.array(spectral_lib.prospect5.kcar, dtype=jnp.float32)
        kbrown_ = jnp.array(spectral_lib.prospect5.kbrown, dtype=jnp.float32)
        kw_     = jnp.array(spectral_lib.prospect5.kw, dtype=jnp.float32)
        km_     = jnp.array(spectral_lib.prospect5.km, dtype=jnp.float32)
        kant_   = jnp.zeros_like(km_)  # no antho in prospect5
    elif prospect_version.upper() == "D":
        nr_     = jnp.array(spectral_lib.prospectd.nr, dtype=jnp.float32)
        kab_    = jnp.array(spectral_lib.prospectd.kab, dtype=jnp.float32)
        kcar_   = jnp.array(spectral_lib.prospectd.kcar, dtype=jnp.float32)
        kbrown_ = jnp.array(spectral_lib.prospectd.kbrown, dtype=jnp.float32)
        kw_     = jnp.array(spectral_lib.prospectd.kw, dtype=jnp.float32)
        km_     = jnp.array(spectral_lib.prospectd.km, dtype=jnp.float32)
        kant_   = jnp.array(spectral_lib.prospectd.kant, dtype=jnp.float32)
    else:
        raise ValueError("prospect_version must be '5' or 'D'")

    # 3) Call the single compiled function
    result = _run_prosail_sail_jax_inner(
        n, cab, car, cbrown, cw, cm, ant,
        alpha,
        rsoil_val, psoil_val,
        lai, lidfa, lidfb, hspot,
        tts, tto, psi,
        nr_, kab_, kcar_, kbrown_, kw_, km_, kant_,
        soil_spec1, soil_spec2,
        lidftype, factor
    )

    # 'result' is either shape (2101,) or (4,2101) if factor='ALL'.
    return result






# #!/usr/bin/env python
# import jax.numpy as jnp

# from prosail import spectral_lib

# from .prospect_d import run_prospect
# from .FourSAIL import foursail

# def run_prosail(n, cab, car,  cbrown, cw, cm, lai, lidfa, hspot,
#                 tts, tto, psi, ant=0.0, alpha=40., prospect_version="5", 
#                 typelidf=2, lidfb=0., factor="SDR",
#                 rsoil0=None, rsoil=None, psoil=None,
#                 soil_spectrum1=None, soil_spectrum2=None):
#     """Run the PROSPECT_5B and SAILh radiative transfer models. The soil
#     model is a linear mixture model, where two spectra are combined together as

#          rho_soil = rsoil*(psoil*soil_spectrum1+(1-psoil)*soil_spectrum2)
#     By default, ``soil_spectrum1`` is a dry soil, and ``soil_spectrum2`` is a
#     wet soil, so in that case, ``psoil`` is a surface soil moisture parameter.
#     ``rsoil`` is a  soil brightness term. You can provide one or the two
#     soil spectra if you want.  The soil spectra must be defined
#     between 400 and 2500 nm with 1nm spacing.

#     Parameters
#     ----------
#     n: float
#         Leaf layers
#     cab: float
#         leaf chlorophyll concentration
#     car: float
#         leaf carotenoid concentration
#     cbrown: float
#         senescent pigment
#     cw: float
#         equivalent leaf water
#     cm: float
#         leaf dry matter
#     lai: float
#         leaf area index
#     lidfa: float
#         a parameter for leaf angle distribution. If ``typliedf``=2, average
#         leaf inclination angle.
#     tts: float
#         Solar zenith angle
#     tto: float
#         Sensor zenith angle
#     psi: float
#         Relative sensor-solar azimuth angle ( saa - vaa )
#     ant: float
#         leaf anthocyanin concentration (default set to 0)
#     alpha: float
#         The alpha angle (in degrees) used in the surface scattering
#         calculations. By default it's set to 40 degrees.
#     prospect_version: str
#         Which PROSPECT version to use. We have "5" and "D"
#     typelidf: int, optional
#         The type of leaf angle distribution function to use. By default, is set
#         to 2.
#     lidfb: float, optional
#         b parameter for leaf angle distribution. If ``typelidf``=2, ignored
#     factor: str, optional
#         What reflectance factor to return:
#         * "SDR": directional reflectance factor (default)
#         * "BHR": bi-hemispherical r. f.
#         * "DHR": Directional-Hemispherical r. f. (directional illumination)
#         * "HDR": Hemispherical-Directional r. f. (directional view)
#         * "ALL": All of them
#     rsoil0: float, optional
#         The soil reflectance spectrum
#     rsoil: float, optional
#         Soil scalar 1 (brightness)
#     psoil: float, optional
#         Soil scalar 2 (moisture)
#     soil_spectrum1: 2101-element array
#         First component of the soil spectrum
#     soil_spectrum2: 2101-element array
#         Second component of the soil spectrum
#     Returns
#     --------
#     A reflectance factor between 400 and 2500 nm


#     """

#     factor = factor.upper()
#     if factor not in ["SDR", "BHR", "DHR", "HDR", "ALL"]:
#         raise ValueError("'factor' must be one of SDR, BHR, DHR, HDR or ALL")

#     if soil_spectrum1 is not None:
#         assert (len(soil_spectrum1) == 2101)
#     else:
#         soil_spectrum1 = spectral_lib.soil.rsoil1

#     if soil_spectrum2 is not None:
#         assert (len(soil_spectrum1) == 2101)
#     else:
#         soil_spectrum2 = spectral_lib.soil.rsoil2

#     if rsoil0 is None:
#         if (rsoil is None) or (psoil is None):
#             raise ValueError("If rsoil0 isn't define, then rsoil and psoil" + \
#                               " need to be defined!")
#         rsoil0 = rsoil * (
#         psoil * soil_spectrum1 + (1. - psoil) * soil_spectrum2)

#     wv, refl, trans = run_prospect (n, cab, car,  cbrown, cw, cm, ant=ant, 
#                  prospect_version=prospect_version, alpha=alpha)
    
#     [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
#          rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot,
#          gammasdf, gammasdb, gammaso] = foursail (refl, trans,  
#                                                   lidfa, lidfb, typelidf, 
#                                                   lai, hspot, 
#                                                   tts, tto, psi, rsoil0)

#     if factor == "SDR":
#         return rsot
#     elif factor == "BHR":
#         return rddt
#     elif factor == "DHR":
#         return rsdt
#     elif factor == "HDR":
#         return rdot
#     elif factor == "ALL":
#         return [rsot, rddt, rsdt, rdot]


# def run_sail(refl, trans, lai, lidfa, hspot, tts, tto, psi,
#              typelidf=2, lidfb=0., factor="SDR",
#              rsoil0=None, rsoil=None, psoil=None,
#              soil_spectrum1=None, soil_spectrum2=None):
#     """Run the SAILh radiative transfer model. The soil model is a linear
#     mixture model, where two spectra are combined together as

#          rho_soil = rsoil*(psoil*soil_spectrum1+(1-psoil)*soil_spectrum2)

#     By default, ``soil_spectrum1`` is a dry soil, and ``soil_spectrum2`` is a
#     wet soil, so in that case, ``psoil`` is a surface soil moisture parameter.
#     ``rsoil`` is a  soil brightness term. You can provide one or the two
#     soil spectra if you want. The soil spectra, and leaf spectra must be defined
#     between 400 and 2500 nm with 1nm spacing.

#     Parameters
#     ----------
#     refl: 2101-element array
#         Leaf reflectance
#     trans: 2101-element array
#         leaf transmittance
#     lai: float
#         leaf area index
#     lidfa: float
#         a parameter for leaf angle distribution. If ``typliedf``=2, average
#         leaf inclination angle.
#     hspot: float
#         The hotspot parameter
#     tts: float
#         Solar zenith angle
#     tto: float
#         Sensor zenith angle
#     psi: float
#         Relative sensor-solar azimuth angle ( saa - vaa )
#     typelidf: int, optional
#         The type of leaf angle distribution function to use. By default, is set
#         to 2.
#     lidfb: float, optional
#         b parameter for leaf angle distribution. If ``typelidf``=2, ignored
#     factor: str, optional
#         What reflectance factor to return:
#         * "SDR": directional reflectance factor (default)
#         * "BHR": bi-hemispherical r. f.
#         * "DHR": Directional-Hemispherical r. f. (directional illumination)
#         * "HDR": Hemispherical-Directional r. f. (directional view)
#         * "ALL": All of them
#     rsoil0: float, optional
#         The soil reflectance spectrum
#     rsoil: float, optional
#         Soil scalar 1 (brightness)
#     psoil: float, optional
#         Soil scalar 2 (moisture)
#     soil_spectrum1: 2101-element array
#         First component of the soil spectrum
#     soil_spectrum2: 2101-element array
#         Second component of the soil spectrum

#     Returns
#     --------
#     Directional surface reflectance between 400 and 2500 nm


#     """

#     factor = factor.upper()
#     if factor not in ["SDR", "BHR", "DHR", "HDR", "ALL"]:
#         raise ValueError("'factor' must be one of SDR, BHR, DHR, HDR or ALL")

#     if soil_spectrum1 is not None:
#         assert (len(soil_spectrum1) == 2101)
#     else:
#         soil_spectrum1 = spectral_lib.soil.rsoil1

#     if soil_spectrum2 is not None:
#         assert (len(soil_spectrum1) == 2101)
#     else:
#         soil_spectrum2 = spectral_lib.soil.rsoil2

#     if rsoil0 is None:
#         if (rsoil is None) or (psoil is None):
#             raise ValueError("If rsoil0 isn't define, then rsoil and psoil" + \
#                               " need to be defined!")
#         rsoil0 = rsoil * (
#         psoil * soil_spectrum1 + (1. - psoil) * soil_spectrum2)

    
#     [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
#          rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot,
#          gammasdf, gammasdb, gammaso] = foursail (refl, trans,  
#                                                   lidfa, lidfb, typelidf, 
#                                                   lai, hspot, 
#                                                   tts, tto, psi, rsoil0)

#     if factor == "SDR":
#         return rsot
#     elif factor == "BHR":
#         return rddt
#     elif factor == "DHR":
#         return rsdt
#     elif factor == "HDR":
#         return rdot
#     elif factor == "ALL":
#         return [rsot, rddt, rsdt, rdot]


# def run_thermal_sail(lam,  
#                      tveg, tsoil, tveg_sunlit, tsoil_sunlit, t_atm, 
#                      lai, lidfa, hspot, rsoil, 
#                      tts, tto, psi,
#                      refl=None, emv=None, ems=None,
#                      typelidf=2, lidfb=0):
#     c1 = 3.741856E-16
#     c2 = 14388.0
#     # Calculate the thermal emission from the different
#     # components using Planck's Law
#     top = (1.0e-6)*c1*(lam*1e-6)**(-5.)
#     Hc = top / ( jnp.exp ( c2/(lam*tveg))-1.)         # Shade leaves
#     Hh = top / ( jnp.exp ( c2/(lam*tveg_sunlit))-1.)  # Sunlit leaves
#     Hd = top / ( jnp.exp ( c2/(lam*tsoil))-1.)        # shade soil 
#     Hs = top / ( jnp.exp ( c2/(lam*tsoil_sunlit))-1.) # Sunlit soil
#     Hsky = top / ( jnp.exp ( c2/(lam*t_atm))-1.)      # Sky emission
    
#     # Emissivity calculations
#     if refl is not None and emv is None:
#         emv = 1. - refl # Assuming absorption is 1
#     if rsoil is not None and ems is None:
#         ems = 1. - rsoil
    
#     [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
#          rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot,
#          gammasdf, gammasdb, gammaso] = foursail (refl, jnp.zeros_like(refl),  
#                                                   lidfa, lidfb, typelidf, 
#                                                   lai, hspot, 
#                                                   tts, tto, psi, rsoil)
    
#     gammad = 1.0 - rdd - tdd
#     gammao = 1.0 - rdo - tdo - too

#     tso = tss*too+tss*(tdo+rsoil*rdd*too)/(1.0-rsoil*rdd)
#     ttot = (too+tdo)/(1.0-rsoil*rdd)
#     gammaot = gammao + ttot*rsoil*gammad
#     gammasot = gammaso + ttot*rsoil*gammasdf

#     aeev = gammaot
#     aees = ttot*ems

#     Lw = ( rdot*Hsky + 
#             (aeev*Hc + 
#             gammasot*emv*(Hh-Hc) + 
#             aees*Hd + 
#             tso*ems*(Hs-Hd)))/jnp.pi
    
#     dnoem1 = top/(Lw*jnp.pi)
#     Tbright = c2/(lam*jnp.log(dnoem1+1.0))
#     dir_em = 1.0 - rdot 
#     return Lw, Tbright, dir_em
        
