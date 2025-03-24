#!/usr/bin/env python
from math import  exp, radians

import jax.numpy as jnp
import jax.lax as lax
from jax import jit
import jax
from functools import partial

# import numpy as np
# import numba


@jax.jit
def volscatt(tts, tto, psi, ttl):
    """
    JAX version of the volscatt function.

    Parameters
    ----------
    tts : float
        Solar Zenith Angle (degrees).
    tto : float
        View Zenith Angle (degrees).
    psi : float
        Relative azimuth angle between view and sun (degrees).
    ttl : float
        Leaf inclination angle (degrees).

    Returns
    -------
    chi_s : float
        Interception function in the solar path.
    chi_o : float
        Interception function in the view path.
    frho : float
        Volume-scattering factor multiplied by leaf reflectance.
    ftau : float
        Volume-scattering factor multiplied by leaf transmittance.
    """
    # Precompute constants
    HALF_PI = jnp.pi * 0.5
    TWO_OVER_PI = 2.0 / jnp.pi
    DENOM_CONST = 2.0 * (jnp.pi ** 2)
    
    # Convert angles to radians once
    tts_rad = jnp.radians(tts)
    tto_rad = jnp.radians(tto)
    psi_rad = jnp.radians(psi)
    ttl_rad = jnp.radians(ttl)
    
    # Compute trigonometric values
    cts = jnp.cos(tts_rad)
    cto = jnp.cos(tto_rad)
    sts = jnp.sin(tts_rad)
    sto = jnp.sin(tto_rad)
    cospsi = jnp.cos(psi_rad)
    
    cttl = jnp.cos(ttl_rad)
    sttl = jnp.sin(ttl_rad)
    
    # Shorthand products
    cs = cttl * cts
    co = cttl * cto
    ss = sttl * sts
    so = sttl * sto
    
    # Compute cosbts and cosbto with fallbacks
    cosbts = jnp.where(jnp.abs(ss) > 1e-6, -cs / ss, 5.0)
    cosbto = jnp.where(jnp.abs(so) > 1e-6, -co / so, 5.0)
    
    # bts and ds: if |cosbts| < 1.0 then use arccos; else, use π and cs
    cond_bts = jnp.abs(cosbts) < 1.0
    bts = jnp.where(cond_bts, jnp.arccos(cosbts), jnp.pi)
    ds  = jnp.where(cond_bts, ss, cs)
    
    # Compute chi_s
    chi_s = TWO_OVER_PI * ((bts - HALF_PI) * cs + jnp.sin(bts) * ss)
    
    # bto and do_: nested conditions handled with jnp.where
    cond_bto = jnp.abs(cosbto) < 1.0
    bto = jnp.where(
        cond_bto,
        jnp.arccos(cosbto),
        jnp.where(tto < 90.0, jnp.pi, 0.0)
    )
    do_ = jnp.where(
        cond_bto,
        so,
        jnp.where(tto < 90.0, co, -co)
    )
    
    # Compute chi_o
    chi_o = TWO_OVER_PI * ((bto - HALF_PI) * co + jnp.sin(bto) * so)
    
    # btran values
    btran1 = jnp.abs(bts - bto)
    btran2 = jnp.pi - jnp.abs((bts + bto) - jnp.pi)
    
    # Determine bt1, bt2, bt3 using nested jnp.where
    bt1 = jnp.where(psi_rad <= btran1, psi_rad, btran1)
    bt2 = jnp.where(psi_rad <= btran1, btran1, jnp.where(psi_rad <= btran2, psi_rad, btran2))
    bt3 = jnp.where(psi_rad <= btran1, btran2, jnp.where(psi_rad <= btran2, btran2, psi_rad))
    
    # t1 and t2 computations
    t1_val = 2.0 * cs * co + ss * so * cospsi
    t2 = jnp.where(
        bt2 > 0.0,
        jnp.sin(bt2) * (2.0 * ds * do_ + ss * so * jnp.cos(bt1) * jnp.cos(bt3)),
        0.0
    )
    
    frho = ((jnp.pi - bt2) * t1_val + t2) / DENOM_CONST
    ftau = ((-bt2) * t1_val + t2) / DENOM_CONST
    
    # Ensure non-negative outputs
    frho = jnp.maximum(frho, 0.0)
    ftau = jnp.maximum(ftau, 0.0)
    
    return chi_s, chi_o, frho, ftau




@jax.jit
def weighted_sum_over_lidf(lidf, tts, tto, psi):
    """
    JAX version of weighted_sum_over_lidf.

    Parameters
    ----------
    lidf : 1D jnp.ndarray
        Leaf Inclination Distribution Function (length n_angles).
    tts, tto, psi : float
        Solar/view geometry angles (degrees).

    Returns
    -------
    ks, ko, bf, sob, sof : float
        Summed/weighted results over the entire LIDF distribution.
    """
    # Precompute angle conversions and related trigonometric values.
    tts_rad = jnp.radians(tts)
    tto_rad = jnp.radians(tto)
    
    cts    = jnp.cos(tts_rad)
    cto    = jnp.cos(tto_rad)
    ctscto = cts * cto

    n_angles   = lidf.shape[0]
    angle_step = 90.0 / n_angles
    # Compute the center of each inclination bin.
    litab = jnp.arange(n_angles) * angle_step + 0.5 * angle_step

    def per_angle(ili, lidf_i):
        # ili: inclination angle (degrees) for the current bin.
        ttl = ili  # ttl is just the inclination angle.
        ttl_rad = jnp.radians(ttl)
        cttl = jnp.cos(ttl_rad)

        # Call volscatt using the original geometry angles.
        chi_s, chi_o, frho, ftau = volscatt(tts, tto, psi, ttl)

        # Extinction coefficients.
        ksli = chi_s / cts
        koli = chi_o / cto

        # Area scattering coefficient fractions.
        sobli = frho * jnp.pi / ctscto
        sofli = ftau * jnp.pi / ctscto
        bfli  = cttl**2

        # Multiply by the LIDF weight.
        return ksli * lidf_i, koli * lidf_i, bfli * lidf_i, sobli * lidf_i, sofli * lidf_i

    # Apply per_angle vectorized over all inclination bins.
    ks_vals, ko_vals, bf_vals, sob_vals, sof_vals = jax.vmap(per_angle)(litab, lidf)

    # Sum results across all angles.
    ks  = jnp.sum(ks_vals)
    ko  = jnp.sum(ko_vals)
    bf  = jnp.sum(bf_vals)
    sob = jnp.sum(sob_vals)
    sof = jnp.sum(sof_vals)

    return ks, ko, bf, sob, sof


@jax.jit
def define_geometric_constants(tts, tto, psi):
    tts_rad = jnp.radians(tts)
    tto_rad = jnp.radians(tto)
    psi_rad = jnp.radians(psi)

    cts = jnp.cos(tts_rad)
    cto = jnp.cos(tto_rad)
    ctscto = cts * cto
    tants = jnp.tan(tts_rad)
    tanto = jnp.tan(tto_rad)
    cospsi = jnp.cos(psi_rad)
    dso = jnp.sqrt(tants**2 + tanto**2 - 2 * tants * tanto * cospsi)
    
    return cts, cto, ctscto, tants, tanto, cospsi, dso

@jax.jit
def hotspot_calculations(alf, lai, ko, ks):
    """
    Performs hotspot calculations with an integration loop.
    """
    fhot = lai * jnp.sqrt(ko * ks)
    
    # Integration parameters and initial values
    fint = (1.0 - jnp.exp(-alf)) * 0.05
    x1, y1, f1, sumint = 0.0, 0.0, 1.0, 0.0
    eps = 1e-10  # small epsilon to prevent division by zero

    def body_fun(istep, carry):
        x1, y1, f1, sumint = carry
        # Use jnp.where: for istep < 20 compute a specific value, else use 1.0
        x2 = jnp.where(istep < 20, -jnp.log(1.0 - istep * fint) / alf, 1.0)
        y2 = -(ko + ks) * lai * x2 + fhot * (1.0 - jnp.exp(-alf * x2)) / alf
        f2 = jnp.exp(y2)
        # Accumulate the integration sum; add eps to denominator to avoid division by zero.
        sumint = sumint + (f2 - f1) * (x2 - x1) / (y2 - y1 + eps)
        return x2, y2, f2, sumint

    # Run the loop for istep from 1 to 20 (inclusive)
    x1, y1, f1, sumint = lax.fori_loop(1, 21, body_fun, (x1, y1, f1, sumint))
    tsstoo = f1
    sumint = jnp.where(jnp.isnan(sumint), 0.0, sumint)
    return tsstoo, sumint

# Elementwise function operating on scalars.
def Jfunc1_element(k, l, t):
    eps = 1e-3
    del_ = (k - l) * t

    def normal_branch(_):
        return (jnp.exp(-l * t) - jnp.exp(-k * t)) / (k - l)

    def near_singular_branch(_):
        return 0.5 * t * (jnp.exp(-k * t) + jnp.exp(-l * t)) * (1.0 - (del_ ** 2) / 12.0)

    # Use lax.cond so that only the chosen branch is computed.
    return lax.cond(jnp.abs(del_) > eps,
                    normal_branch,
                    near_singular_branch,
                    operand=None)

# Wrapper to allow Jfunc1_element to work on arbitrarily shaped inputs.
def Jfunc1_wrapper(k, l, t):
    """
    Jfunc1_wrapper applies Jfunc1_element elementwise.
    It broadcasts inputs k, l, t to a common shape, flattens them,
    applies vmap over scalars, and reshapes the result back.
    """
    # Ensure inputs are at least 1D.
    k_arr = jnp.atleast_1d(k)
    l_arr = jnp.atleast_1d(l)
    t_arr = jnp.atleast_1d(t)

    # Determine the common target shape.
    target_shape = jnp.broadcast_shapes(k_arr.shape, l_arr.shape, t_arr.shape)

    # Broadcast each input to the target shape.
    k_arr = jnp.broadcast_to(k_arr, target_shape)
    l_arr = jnp.broadcast_to(l_arr, target_shape)
    t_arr = jnp.broadcast_to(t_arr, target_shape)

    # Flatten the arrays.
    flat_k = k_arr.ravel()
    flat_l = l_arr.ravel()
    flat_t = t_arr.ravel()

    # Apply Jfunc1_element to each scalar triple.
    flat_result = jax.vmap(Jfunc1_element)(flat_k, flat_l, flat_t)

    # Reshape the flat result back to the target shape.
    result = flat_result.reshape(target_shape)
    return result

Jfunc1 = jax.jit(Jfunc1_wrapper)


@jax.jit
def Jfunc2(k, l, t):
    """
    J2 function.

    Parameters
    ----------
    k, l, t : float or array

    Returns
    -------
    result : jnp.ndarray or float
        (1 - exp(-(k+l)*t)) / (k + l)
    """
    return (1.0 - jnp.exp(-(k + l) * t)) / (k + l)


# Helper function: ensure input has at least one batch dimension.
def ensure_batch(x):
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = x[None]
    elif x.ndim > 1:
        # If there’s an extra dimension, flatten it so that batch dimension is first.
        x = x.reshape(x.shape[0])
    return x

##########################################
# Batched Campbell Function (supports scalar too)
##########################################
@partial(jax.jit, static_argnums=(1,))
def campbell(alpha, n_elements=18):
    """
    Batched JAX version of Campbell's ellipsoidal LIDF.
    
    Parameters
    ----------
    alpha : jnp.ndarray or scalar
        Mean leaf angle in degrees. If scalar, treated as a batch of one.
    n_elements : int, optional
        Number of equally spaced inclination angles (default 18).
        
    Returns
    -------
    lidf : jnp.ndarray
        LIDF of shape (B, n_elements), where each row sums to 1.
    """
    alpha = ensure_batch(alpha)  # shape (B,)
    B = alpha.shape[0]
    
    # Compute excentricity factor per sample; shape (B,)
    excent = jnp.exp(-1.6184e-5 * alpha**3 +
                       2.1145e-3 * alpha**2 -
                       1.2390e-1 * alpha +
                       3.2491)
    # Reshape for broadcasting: shape (B, 1)
    excent = excent[:, None]

    step = 90.0 / n_elements
    # Bin boundaries in degrees, length = n_elements+1
    degrees = jnp.arange(n_elements + 1) * step  # shape (n_elements+1,)
    # Compute tl1 and tl2 (in radians) for each bin; these are scalars.
    tl1 = jnp.deg2rad(degrees[:-1])  # shape (n_elements,)
    tl2 = jnp.deg2rad(degrees[1:])   # shape (n_elements,)
    # Expand to (1, n_elements) for broadcasting
    tl1 = tl1[None, :]
    tl2 = tl2[None, :]

    # Compute intermediate values; shapes (B, n_elements)
    x1 = excent / jnp.sqrt(1.0 + excent**2 * jnp.tan(tl1)**2)
    x2 = excent / jnp.sqrt(1.0 + excent**2 * jnp.tan(tl2)**2)

    # --- Elementwise functions for a single sample ---
    # For a given sample, ex is a scalar and x1_row, x2_row are vectors of length n_elements.
    def freq_if_excent_eq_1_sample(x1_row, x2_row):
        # Use the constant computed from the bin boundaries.
        return jnp.abs(jnp.cos(tl1[0]) - jnp.cos(tl2[0]))
    
    def freq_if_excent_gt_1_sample(ex, x1_row, x2_row):
        alph = ex / jnp.sqrt(jnp.abs(1.0 - ex**2))
        alph2 = alph**2
        x12 = x1_row**2
        x22 = x2_row**2
        alpx1 = jnp.sqrt(alph2 + x12)
        alpx2 = jnp.sqrt(alph2 + x22)
        dum1 = x1_row * alpx1 + alph2 * jnp.log(x1_row + alpx1)
        dum2 = x2_row * alpx2 + alph2 * jnp.log(x2_row + alpx2)
        return jnp.abs(dum1 - dum2)
    
    def freq_if_excent_lt_1_sample(ex, x1_row, x2_row):
        alph = ex / jnp.sqrt(jnp.abs(1.0 - ex**2))
        alph2 = alph**2
        x12 = x1_row**2
        x22 = x2_row**2
        almx1 = jnp.sqrt(alph2 - x12)
        almx2 = jnp.sqrt(alph2 - x22)
        dum1 = x1_row * almx1 + alph2 * jnp.arcsin(x1_row / alph)
        dum2 = x2_row * almx2 + alph2 * jnp.arcsin(x2_row / alph)
        return jnp.abs(dum1 - dum2)
    
    def compute_freq_sample(ex, x1_row, x2_row):
        # Use lax.cond to choose between the gt and lt branches.
        return lax.cond(
            ex > 1.0,
            lambda _: freq_if_excent_gt_1_sample(ex, x1_row, x2_row),
            lambda _: freq_if_excent_lt_1_sample(ex, x1_row, x2_row),
            operand=None)
    
    def compute_freq_eq_or_ne_sample(ex, x1_row, x2_row):
        # If ex is (nearly) 1, use the eq branch; otherwise, use the computed branch.
        return lax.cond(
            jnp.isclose(ex, 1.0, atol=1e-9, rtol=1e-9),
            lambda _: freq_if_excent_eq_1_sample(x1_row, x2_row),
            lambda _: compute_freq_sample(ex, x1_row, x2_row),
            operand=None)
    
    # --- Apply the above elementwise functions via vmap over the batch dimension.
    # For each batch element, excent[i,0] is the scalar ex, and x1[i,:], x2[i,:] are the vectors.
    freq_each = jax.vmap(compute_freq_eq_or_ne_sample)(excent[:, 0], x1, x2)
    
    # Normalize each sample's frequency vector so that it sums to 1.
    sum0 = jnp.sum(freq_each, axis=1, keepdims=True)  # shape (B, 1)
    lidf = freq_each / sum0  # shape (B, n_elements)
    return lidf


##########################################
# Batched Verhoef-Bimodal Function (supports scalar too)
##########################################
@partial(jax.jit, static_argnums=(2,))
def verhoef_bimodal(a, b, n_elements=18):
    """
    Batched JAX version of Verhoef's bimodal LIDF.
    
    Parameters
    ----------
    a : jnp.ndarray or scalar
        Controls the average leaf slope. If scalar, treated as batch of one.
    b : jnp.ndarray or scalar
        Controls the distribution's bimodality. If scalar, treated as batch of one.
    n_elements : int, optional
        Number of equally spaced inclination angles (default 18).
        
    Returns
    -------
    lidf : jnp.ndarray
        LIDF of shape (B, n_elements) with each row summing to 1.
    """
    a = ensure_batch(a)  # shape (B,)
    b = ensure_batch(b)  # shape (B,)
    B = a.shape[0]
    
    # Create descending angles (in degrees) for the bins.
    step = 90.0 / n_elements
    angles = jnp.flip(jnp.arange(n_elements) * step)  # shape (n_elements,)
    
    # Define an elementwise compute_f for a single sample.
    def compute_f_sample(angle, a_elem, b_elem):
        tl1 = jnp.radians(angle)  # scalar
        # "If" branch for the sample.
        f_if = 1.0 - jnp.cos(tl1)
        # "Else" branch: iterative loop.
        def else_branch(_):
            eps = 1e-8
            max_iter = 100
            x = 2.0 * tl1  # scalar
            # Broadcast x to a scalar for this sample (already scalar).
            p = x
            delx = 1.0
            y = 0.0
            def body_fun(iter, state):
                x, delx, y = state
                new_y = a_elem * jnp.sin(x) + 0.5 * b_elem * jnp.sin(2.0 * x)
                dx = 0.5 * (new_y - x + p)
                new_x = x + dx
                new_delx = jnp.abs(dx)
                return (new_x, new_delx, new_y)
            x, delx, y = lax.fori_loop(0, max_iter, body_fun, (x, delx, y))
            return (2.0 * y + p) / jnp.pi
        f_else = else_branch(None)
        # Use lax.cond to choose the branch for this sample.
        return lax.cond(a_elem > 1.0, lambda _: f_if, lambda _: f_else, operand=None)
    
    # Now compute f for each sample and for each bin.
    # We want to apply compute_f_sample for each angle in the "angles" array and then accumulate.
    # Here we mimic the loop over angles.
    def body_fun(i, carry):
        freq, lidf = carry  # freq: (B,), lidf: (B, n_elements)
        angle = angles[i]   # scalar
        # Compute f for each sample at this angle using vmap.
        f = jax.vmap(lambda a_elem, b_elem: compute_f_sample(angle, a_elem, b_elem))(a, b)
        # Update: for each sample, new frequency = old frequency - f, and store that in column i.
        new_lidf = lidf.at[:, i].set(freq - f)
        return (f, new_lidf)
    
    freq_init = jnp.full((B,), 1.0)          # shape (B,)
    lidf_init = jnp.zeros((B, n_elements))     # shape (B, n_elements)
    _, lidf_out = lax.fori_loop(0, n_elements, body_fun, (freq_init, lidf_init))
    # Reverse along the bin dimension.
    lidf_out = lidf_out[:, ::-1]
    return lidf_out



# @jax.jit
@partial(jax.jit, static_argnums=(4,))
def foursail(rho, tau, lidfa, lidfb, lidftype, lai, hotspot,
                    tts, tto, psi, rsoil):
    """
    JAX version of FourSAIL without lax.cond, using a mask approach.

    - We compute "no canopy" outputs (shape = rsoil.shape).
    - We compute "canopy" outputs (same shape).
    - Then we combine them with a mask: "no_canopy_mask = (lai <= 0)".

    The final return is a tuple of 21 arrays, all shape = rsoil.shape.
    """

    # Use rsoil's shape as our reference for all arrays:
    ref_shape = rsoil.shape
    dtype     = rsoil.dtype

    # Create a mask: True where LAI <= 0, False where LAI > 0.
    # But note that if lai is a scalar and rsoil is an array (3, ),
    # we broadcast it. This is typically fine in JAX.
    no_canopy_mask = (lai <= 0.0)

    # ----------------------------------------------------------
    # 1) NO CANOPY outputs (as arrays, shape = ref_shape)
    # ----------------------------------------------------------
    ones  = jnp.ones(ref_shape, dtype=dtype)
    zeros = jnp.zeros(ref_shape, dtype=dtype)

    tss_no      = ones
    too_no      = ones
    tsstoo_no   = ones
    rdd_no      = zeros
    tdd_no      = ones
    rsd_no      = zeros
    tsd_no      = zeros
    rdo_no      = zeros
    tdo_no      = zeros
    rso_no      = zeros
    rsos_no     = zeros
    rsod_no     = zeros
    # soil-based terms remain shaped like rsoil
    rddt_no     = rsoil
    rsdt_no     = rsoil
    rdot_no     = rsoil
    rsodt_no    = zeros
    rsost_no    = rsoil
    rsot_no     = rsoil
    gammasdf_no = zeros
    gammasdb_no = zeros
    gammaso_no  = zeros

    # ----------------------------------------------------------
    # 2) CANOPY outputs (as arrays, shape = ref_shape)
    # ----------------------------------------------------------
    # Instead of calling a separate function, we just inline the canopy logic:
    # We'll also note that if lai is scalar > 0, everything broadcasts to (ref_shape,).

    # Geometry (assuming we have a define_geometric_constants JAX function).
    cts, cto, ctscto, tants, tanto, cospsi, dso = \
        define_geometric_constants(tts, tto, psi)

    # LIDF
    # For example, we might do an if-lidftype check once outside or always call verhoef_bimodal:
    # If you truly have different LIDF shapes, you'd either unify them or keep them scalar, etc.
    # Assume all samples have the same lidftype.
    # lidftype_scalar = jnp.asscalar(lidftype)  # or use jnp.asscalar(lidftype) if it's 0-d
    # lidftype_scalar = lidftype.item()
    lidftype = int(lidftype)
    lidf = jax.lax.cond(
        lidftype == 1,
        lambda _: verhoef_bimodal(lidfa, lidfb, 18),
        lambda _: campbell(lidfa, 18),
        operand=None
    )

    # lidf = campbell(lidfa, 18)
    # lidf = jax.lax.cond(
    #     lidftype == 1,
    #     lambda _: verhoef_bimodal(lidfa, lidfb, 18),
    #     lambda _: campbell(lidfa, 18),
    #     operand=None
    # )
                        

    ks, ko, bf, sob, sof = weighted_sum_over_lidf(lidf, tts, tto, psi)

    sdb = 0.5 * (ks + bf)
    sdf = 0.5 * (ks - bf)
    dob = 0.5 * (ko + bf)
    dof = 0.5 * (ko - bf)
    ddb = 0.5 * (1.0 + bf)
    ddf = 0.5 * (1.0 - bf)

    sigb = jnp.maximum(ddb * rho + ddf * tau, 1e-36)
    sigf = jnp.maximum(ddf * rho + ddb * tau, 1e-36)

    att = 1.0 - sigf
    m   = jnp.sqrt(att**2 - sigb**2)
    sb  = sdb * rho + sdf * tau
    sf  = sdf * rho + sdb * tau
    vb  = dob * rho + dof * tau
    vf  = dof * rho + dob * tau
    w   = sob * rho + sof * tau

    e1    = jnp.exp(-m * lai)
    e2    = e1**2
    rinf  = (att - m) / sigb
    rinf2 = rinf**2
    re    = rinf * e1
    denom = 1.0 - rinf2 * e2

    J1ks = Jfunc1(ks, m, lai)
    J2ks = Jfunc2(ks, m, lai)
    J1ko = Jfunc1(ko, m, lai)
    J2ko = Jfunc2(ko, m, lai)

    Pss = (sf + sb * rinf) * J1ks
    Qss = (sf * rinf + sb) * J2ks
    Pv  = (vf + vb * rinf) * J1ko
    Qv  = (vf * rinf + vb) * J2ko

    tdd_yes = (1.0 - rinf2) * e1 / denom
    rdd_yes = rinf * (1.0 - e2) / denom
    tsd_yes = (Pss - re * Qss) / denom
    rsd_yes = (Qss - re * Pss) / denom
    tdo_yes = (Pv - re * Qv) / denom
    rdo_yes = (Qv - re * Pv) / denom

    gammasdf_yes = (1.0 + rinf) * (J1ks - re * J2ks) / denom
    gammasdb_yes = (1.0 + rinf) * (-re * J1ks + J2ks) / denom

    tss_yes = jnp.exp(-ks * lai)
    too_yes = jnp.exp(-ko * lai)

    z   = Jfunc2(ks, ko, lai)
    g1  = (z - J1ks * too_yes) / (ko + m)
    g2  = (z - J1ko * tss_yes) / (ks + m)
    Tv1 = (vf * rinf + vb) * g1
    Tv2 = (vf + vb * rinf) * g2
    T1  = Tv1 * (sf + sb * rinf)
    T2  = Tv2 * (sf * rinf + sb)
    T3  = (rdo_yes * Qss + tdo_yes * Pss) * rinf

    rsod_yes = (T1 + T2 - T3) / (1.0 - rinf**2)

    T4       = Tv1 * (1.0 + rinf)
    T5       = Tv2 * (1.0 + rinf)
    T6       = (rdo_yes * J2ks + tdo_yes * J1ks) * (1.0 + rinf) * rinf
    gammasod_yes = (T4 + T5 - T6) / (1.0 - rinf**2)

    # Hotspot
    alf_init = 1e36
    alf = jnp.where(hotspot > 0.0,
                    (dso / hotspot) * 2.0 / (ks + ko),
                    alf_init)

    def pure_hotspot(_):
        tss_   = jnp.exp(-ks * lai)
        sumint = (1.0 - tss_) / (ks * lai)
        return (tss_, sumint)

    def outside_hotspot(_):
        return hotspot_calculations(alf, lai, ko, ks)

    tsstoo_yes, sumint = jax.lax.cond(
        jnp.isclose(alf, 0.0, atol=1e-15),
        pure_hotspot,
        outside_hotspot,
        operand=None
    )

    rsos_yes    = w * lai * sumint
    gammasos_yes= ko * lai * sumint
    rso_yes     = rsos_yes + rsod_yes
    gammaso_yes = gammasos_yes + gammasod_yes

    dn   = jnp.maximum(1.0 - rsoil * rdd_yes, 1e-36)
    rddt_yes = rdd_yes + tdd_yes * rsoil * tdd_yes / dn
    rsdt_yes = rsd_yes + (tsd_yes + tss_yes) * rsoil * tdd_yes / dn
    rdot_yes = rdo_yes + tdd_yes * rsoil * (tdo_yes + too_yes) / dn
    rsodt_yes= ((tss_yes + tsd_yes) * tdo_yes 
                + (tsd_yes + tss_yes * rsoil * rdd_yes) * too_yes
               ) * rsoil / dn
    rsost_yes = rso_yes + tsstoo_yes * rsoil
    rsot_yes  = rsost_yes + rsodt_yes

    # ----------------------------------------------------------
    # 3) Combine with the mask
    # ----------------------------------------------------------
    # final_x = jnp.where(no_canopy_mask, x_no, x_yes) 
    # for each variable.

    tss      = jnp.where(no_canopy_mask, tss_no,      tss_yes)
    too      = jnp.where(no_canopy_mask, too_no,      too_yes)
    tsstoo   = jnp.where(no_canopy_mask, tsstoo_no,   tsstoo_yes)
    rdd      = jnp.where(no_canopy_mask, rdd_no,      rdd_yes)
    tdd      = jnp.where(no_canopy_mask, tdd_no,      tdd_yes)
    rsd      = jnp.where(no_canopy_mask, rsd_no,      rsd_yes)
    tsd      = jnp.where(no_canopy_mask, tsd_no,      tsd_yes)
    rdo      = jnp.where(no_canopy_mask, rdo_no,      rdo_yes)
    tdo      = jnp.where(no_canopy_mask, tdo_no,      tdo_yes)
    rso      = jnp.where(no_canopy_mask, rso_no,      rso_yes)
    rsos     = jnp.where(no_canopy_mask, rsos_no,     rsos_yes)
    rsod     = jnp.where(no_canopy_mask, rsod_no,     rsod_yes)
    rddt     = jnp.where(no_canopy_mask, rddt_no,     rddt_yes)
    rsdt     = jnp.where(no_canopy_mask, rsdt_no,     rsdt_yes)
    rdot     = jnp.where(no_canopy_mask, rdot_no,     rdot_yes)
    rsodt    = jnp.where(no_canopy_mask, rsodt_no,    rsodt_yes)
    rsost    = jnp.where(no_canopy_mask, rsost_no,    rsost_yes)
    rsot     = jnp.where(no_canopy_mask, rsot_no,     rsot_yes)
    gammasdf = jnp.where(no_canopy_mask, gammasdf_no, gammasdf_yes)
    gammasdb = jnp.where(no_canopy_mask, gammasdb_no, gammasdb_yes)
    gammaso  = jnp.where(no_canopy_mask, gammaso_no,  gammaso_yes)

    return (tss[0], too[0], tsstoo[0],
            rdd, tdd, rsd, tsd, rdo, tdo,
            rso, rsos, rsod,
            rddt, rsdt, rdot, rsodt, rsost, rsot,
            gammasdf, gammasdb, gammaso)




















# #!/usr/bin/env python
# from math import  exp, radians

# import jax.numpy as jnp
# import jax.lax as lax
# from jax import jit
# import jax
# from functools import partial

# # import numpy as np
# # import numba


# @jax.jit
# def volscatt(tts, tto, psi, ttl):
#     """
#     JAX version of the volscatt function.

#     Parameters
#     ----------
#     tts : float
#         Solar Zenith Angle (degrees).
#     tto : float
#         View Zenith Angle (degrees).
#     psi : float
#         Relative azimuth angle between view and sun (degrees).
#     ttl : float
#         Leaf inclination angle (degrees).

#     Returns
#     -------
#     chi_s : float
#         Interception function in the solar path.
#     chi_o : float
#         Interception function in the view path.
#     frho : float
#         Volume-scattering factor multiplied by leaf reflectance.
#     ftau : float
#         Volume-scattering factor multiplied by leaf transmittance.
#     """
#     # Convert angles to radians via JAX
#     cts   = jnp.cos(jnp.radians(tts))
#     cto   = jnp.cos(jnp.radians(tto))
#     sts   = jnp.sin(jnp.radians(tts))
#     sto   = jnp.sin(jnp.radians(tto))
#     psir  = jnp.radians(psi)
#     cospsi= jnp.cos(psir)

#     cttl  = jnp.cos(jnp.radians(ttl))
#     sttl  = jnp.sin(jnp.radians(ttl))

#     # Shorthand products
#     cs = cttl * cts
#     co = cttl * cto
#     ss = sttl * sts
#     so = sttl * sto

#     # Handle cosbts
#     # If |ss| > 1e-6 => cosbts = -cs/ss, else 5.0
#     cosbts = jnp.where(jnp.abs(ss) > 1e-6, -cs / ss, 5.0)

#     # Handle cosbto
#     # If |so| > 1e-6 => cosbto = -co/so, else 5.0
#     cosbto = jnp.where(jnp.abs(so) > 1e-6, -co / so, 5.0)

#     # bts & ds
#     # if |cosbts| < 1.0: bts = arccos(cosbts), ds = ss
#     # else:              bts = pi,            ds = cs
#     bts = jnp.where(jnp.abs(cosbts) < 1.0, jnp.arccos(cosbts), jnp.pi)
#     ds  = jnp.where(jnp.abs(cosbts) < 1.0, ss, cs)

#     # chi_s
#     chi_s = 2.0 / jnp.pi * ((bts - jnp.pi * 0.5) * cs + jnp.sin(bts) * ss)

#     # bto & do_
#     # if |cosbto| < 1.0 => bto=arccos(cosbto), do_=so
#     # else => if tto<90 => bto=pi, do_=co; else => bto=0, do_=-co
#     bto = jnp.where(
#         jnp.abs(cosbto) < 1.0,
#         jnp.arccos(cosbto),
#         jnp.where(tto < 90.0, jnp.pi, 0.0)
#     )
#     do_ = jnp.where(
#         jnp.abs(cosbto) < 1.0,
#         so,
#         jnp.where(tto < 90.0, co, -co)
#     )

#     # chi_o
#     chi_o = 2.0 / jnp.pi * ((bto - jnp.pi * 0.5) * co + jnp.sin(bto) * so)

#     # btran1, btran2
#     btran1 = jnp.abs(bts - bto)
#     btran2 = jnp.pi - jnp.abs((bts + bto) - jnp.pi)

#     # We have nested "if" logic:
#     # if psir <= btran1:
#     #   bt1=psir,  bt2=btran1, bt3=btran2
#     # else:
#     #   if psir <= btran2: bt1=btran1, bt2=psir,   bt3=btran2
#     #   else:              bt1=btran1, bt2=btran2, bt3=psir

#     def if_psir_leq_btran1(_):
#         return psir, btran1, btran2

#     def else_psir_leq_btran1(_):
#         def if_psir_leq_btran2(_):
#             return btran1, psir, btran2

#         def else_psir_leq_btran2(_):
#             return btran1, btran2, psir

#         return jax.lax.cond(psir <= btran2,
#                             if_psir_leq_btran2,
#                             else_psir_leq_btran2,
#                             operand=None)

#     bt1, bt2, bt3 = jax.lax.cond(
#         psir <= btran1,
#         if_psir_leq_btran1,
#         else_psir_leq_btran1,
#         operand=None
#     )

#     # t1 & t2
#     t1 = 2.0 * cs * co + ss * so * cospsi
#     t2 = jnp.where(
#         bt2 > 0.0,
#         jnp.sin(bt2) * (2.0 * ds * do_ + ss * so * jnp.cos(bt1) * jnp.cos(bt3)),
#         0.0
#     )

#     denom = 2.0 * (jnp.pi ** 2)
#     frho = ((jnp.pi - bt2) * t1 + t2) / denom
#     ftau = ((-bt2) * t1 + t2) / denom

#     # Clip frho, ftau at 0
#     frho = jnp.where(frho < 0.0, 0.0, frho)
#     ftau = jnp.where(ftau < 0.0, 0.0, ftau)

#     return (chi_s, chi_o, frho, ftau)



# @jax.jit
# def weighted_sum_over_lidf(lidf, tts, tto, psi):
#     """
#     JAX version of weighted_sum_over_lidf.

#     Parameters
#     ----------
#     lidf : 1D jnp.ndarray
#         Leaf Inclination Distribution Function (length n_angles).
#     tts, tto, psi : float
#         Solar/view geometry angles (degrees).

#     Returns
#     -------
#     ks, ko, bf, sob, sof : float
#         Summed/weighted results over the entire LIDF distribution.
#     """
#     # Precompute cosines
#     cts     = jnp.cos(jnp.radians(tts))
#     cto     = jnp.cos(jnp.radians(tto))
#     ctscto  = cts * cto

#     n_angles   = lidf.shape[0]
#     angle_step = 90.0 / n_angles
#     # Center of each inclination bin
#     litab = jnp.arange(n_angles) * angle_step + 0.5 * angle_step

#     def per_angle(ili, lidf_i):
#         # ili: inclination angle in degrees
#         # lidf_i: LIDF weight for that inclination
#         ttl = ili
#         cttl = jnp.cos(jnp.radians(ttl))

#         chi_s, chi_o, frho, ftau = volscatt(tts, tto, psi, ttl)

#         # Extinction coefficients
#         ksli  = chi_s / cts
#         koli  = chi_o / cto

#         # Area scattering coefficient fractions
#         sobli = frho * jnp.pi / ctscto
#         sofli = ftau * jnp.pi / ctscto
#         bfli  = cttl**2

#         # Multiply each by the LIDF weight
#         return (
#             ksli  * lidf_i,
#             koli  * lidf_i,
#             bfli  * lidf_i,
#             sobli * lidf_i,
#             sofli * lidf_i,
#         )

#     # Vectorize over all inclination angles
#     ks_vals, ko_vals, bf_vals, sob_vals, sof_vals = jax.vmap(per_angle)(litab, lidf)

#     # Sum across all angles
#     ks  = jnp.sum(ks_vals)
#     ko  = jnp.sum(ko_vals)
#     bf  = jnp.sum(bf_vals)
#     sob = jnp.sum(sob_vals)
#     sof = jnp.sum(sof_vals)

#     return ks, ko, bf, sob, sof



# @jit
# def define_geometric_constants(tts, tto, psi):
#     cts = jnp.cos(jnp.radians(tts))
#     cto = jnp.cos(jnp.radians(tto))
#     ctscto = cts * cto
#     tants = jnp.tan(jnp.radians(tts))
#     tanto = jnp.tan(jnp.radians(tto))
#     cospsi = jnp.cos(jnp.radians(psi))
#     dso = jnp.sqrt(tants**2 + tanto**2 - 2 * tants * tanto * cospsi)
    
#     return cts, cto, ctscto, tants, tanto, cospsi, dso

# @jit
# def hotspot_calculations(alf, lai, ko, ks):
#     fhot = lai * jnp.sqrt(ko * ks)

#     # Integration using JAX-friendly approach
#     x1 = 0.0
#     y1 = 0.0
#     f1 = 1.0
#     fint = (1.0 - jnp.exp(-alf)) * 0.05
#     sumint = 0.0

#     def body_fun(istep, carry):
#         x1, y1, f1, sumint = carry

#         x2 = lax.cond(
#             istep < 20,
#             lambda _: -jnp.log(1.0 - istep * fint) / alf,
#             lambda _: 1.0,
#             operand=None
#         )
#         y2 = -(ko + ks) * lai * x2 + fhot * (1.0 - jnp.exp(-alf * x2)) / alf
#         f2 = jnp.exp(y2)
#         sumint += (f2 - f1) * (x2 - x1) / (y2 - y1)
        
#         return x2, y2, f2, sumint

#     x1, y1, f1, sumint = lax.fori_loop(1, 21, body_fun, (x1, y1, f1, sumint))
#     tsstoo = f1
#     sumint = lax.cond(jnp.isnan(sumint), lambda _: 0.0, lambda _: sumint, operand=None)

#     return tsstoo, sumint


# def Jfunc1_element(k, l, t):
#     eps = 1e-3
#     del_ = (k - l) * t

#     def normal_branch(_):
#         return (jnp.exp(-l * t) - jnp.exp(-k * t)) / (k - l)

#     def near_singular_branch(_):
#         return 0.5 * t * (jnp.exp(-k * t) + jnp.exp(-l * t)) * (1.0 - (del_**2) / 12.0)

#     return jax.lax.cond(jnp.abs(del_) > eps,
#                         normal_branch,
#                         near_singular_branch,
#                         operand=None)

# def Jfunc1_wrapper(k, l, t):
#     # Ensure inputs are at least 1D.
#     k_arr = jnp.atleast_1d(k)
#     l_arr = jnp.atleast_1d(l)
#     t_arr = jnp.atleast_1d(t)
    
#     # Determine the common target shape from the inputs.
#     target_shape = jnp.broadcast_shapes(k_arr.shape, l_arr.shape, t_arr.shape)
    
#     # Broadcast each input to the target shape.
#     k_arr = jnp.broadcast_to(k_arr, target_shape)
#     l_arr = jnp.broadcast_to(l_arr, target_shape)
#     t_arr = jnp.broadcast_to(t_arr, target_shape)
    
#     # Apply the elementwise function using vmap.
#     result = jax.vmap(Jfunc1_element)(k_arr, l_arr, t_arr)
    
#     # If the result is a single element array, extract the scalar.
#     if result.shape == (1,):
#         return result[0]
#     return result

# Jfunc1 = jax.jit(Jfunc1_wrapper)

# # @jax.jit
# # def Jfunc1(k, l, t):
# #     """
# #     J1 function with avoidance of the near-singularity problem.

# #     Parameters
# #     ----------
# #     k : float or array
# #     l : float or array
# #     t : float or array

# #     Returns
# #     -------
# #     result : jnp.ndarray or float
# #         Same shape as 'l', containing the piecewise-defined result.
# #     """
# #     eps = 1e-3
# #     del_ = (k - l) * t

# #     def normal_branch(_):
# #         return (jnp.exp(-l * t) - jnp.exp(-k * t)) / (k - l)

# #     def near_singular_branch(_):
# #         return 0.5 * t * (jnp.exp(-k * t) + jnp.exp(-l * t)) * (1.0 - (del_**2) / 12.0)

# #     return jax.lax.cond(jnp.abs(del_) > eps, normal_branch, near_singular_branch, operand=None)


# #     # eps = 1e-3
# #     # del_ = (k - l) * t
# #     # # For the "normal" branch (|del_| > eps)
# #     # normal_branch = (jnp.exp(-l * t) - jnp.exp(-k * t)) / (k - l)

# #     # # For the "near-singular" branch (|del_| <= eps)
# #     # near_singular_branch = (
# #     #     0.5 * t * (jnp.exp(-k * t) + jnp.exp(-l * t)) * (1.0 - (del_**2) / 12.0)
# #     # )

# #     # # Piecewise selection
# #     # result = jnp.where(
# #     #     jnp.abs(del_) > eps,
# #     #     normal_branch,
# #     #     near_singular_branch
# #     # )
# #     # return result


# @jax.jit
# def Jfunc2(k, l, t):
#     """
#     J2 function.

#     Parameters
#     ----------
#     k, l, t : float or array

#     Returns
#     -------
#     result : jnp.ndarray or float
#         (1 - exp(-(k+l)*t)) / (k + l)
#     """
#     return (1.0 - jnp.exp(-(k + l) * t)) / (k + l)

# @partial(jit, static_argnums=(2,))
# def verhoef_bimodal(a, b, n_elements=18):
#     """
#     JAX-accelerated version of the Verhoef's bimodal LIDF function.
    
#     Parameters
#     ----------
#     a : float
#         Controls the average leaf slope.
#     b : float
#         Controls the distribution's bimodality.
#         Requirement: |a| + |b| < 1 for physically meaningful LIDF.
#     n_elements : int
#         Number of equally spaced inclination angles. Default 18.

#     Returns
#     -------
#     lidf : jnp.ndarray
#         Leaf Inclination Distribution Function at equally spaced angles.
#     """
#     step   = 90.0 / n_elements
#     angles = (jnp.arange(n_elements) * step)[::-1]  # descending angles

#     def compute_f(angle):
#         """Compute the 'f' value for a single angle (tl1)."""
#         tl1 = jnp.radians(angle)

#         def if_branch(_):
#             # If a > 1.0, use the simple expression
#             return 1.0 - jnp.cos(tl1)

#         def else_branch(_):
#             # Otherwise do the iterative approach in a while_loop
#             # We solve for f by iterating until delx < eps or max_iter is reached.
#             eps      = 1e-8
#             x0       = 2.0 * tl1    # initial guess
#             p        = x0          # store this to use after iteration
#             delx0    = 1.0
#             y0       = 0.0
#             max_iter = 100         # maximum safe iterations

#             def cond_fun(state):
#                 i, x, delx, y = state
#                 # keep iterating if not converged and haven't exceeded max_iter
#                 return jnp.logical_and(delx >= eps, i < max_iter)

#             def body_fun(state):
#                 i, x, delx, y = state
#                 new_y     = a * jnp.sin(x) + 0.5 * b * jnp.sin(2.0 * x)
#                 dx        = 0.5 * (new_y - x + p)
#                 new_x     = x + dx
#                 new_delx  = jnp.abs(dx)
#                 return (i+1, new_x, new_delx, new_y)

#             init_state               = (0, x0, delx0, y0)
#             i_final, x_final, _, y_final = lax.while_loop(cond_fun, body_fun, init_state)

#             return (2.0 * y_final + p) / jnp.pi

#         # Choose which branch to run based on condition (a > 1.0)
#         return lax.cond(a > 1.0, if_branch, else_branch, operand=None)

#     # We'll keep track of 'freq' as we go and fill lidf.
#     def body_fun(i, carry):
#         freq, lidf = carry
#         angle = angles[i]
#         f     = compute_f(angle)           # get f from either branch
#         new_lidf = lidf.at[i].set(freq - f)  # store freq-f at index i
#         return (f, new_lidf)

#     freq_init = 1.0
#     lidf_init = jnp.zeros(n_elements)
#     _, lidf_out = lax.fori_loop(0, n_elements, body_fun, (freq_init, lidf_init))

#     # Reverse the array to match the original code’s final line: lidf[::-1]
#     lidf_out = lidf_out[::-1]
#     return lidf_out



# @partial(jit, static_argnums=(1,))
# def campbell(alpha, n_elements=18):
#     """
#     JAX version of Campbell's ellipsoidal LIDF.
    
#     Parameters
#     ----------
#     alpha : float
#         Mean leaf angle in degrees (e.g. alpha=57 for a spherical LIDF).
#     n_elements : int
#         Number of equally spaced inclination angles.

#     Returns
#     -------
#     lidf : jnp.ndarray
#         Leaf Inclination Distribution Function of length n_elements.
#     """

#     # Convert alpha to a JAX float (double precision)
#     alpha = jnp.asarray(alpha)
    
#     # Excentricity factor from empirical polynomial (Campbell's formula)
#     excent = jnp.exp(
#         -1.6184e-5*alpha**3
#         + 2.1145e-3*alpha**2
#         - 1.2390e-1*alpha
#         + 3.2491
#     )

#     step = 90.0 / n_elements
#     i_array = jnp.arange(n_elements)  # 0..n_elements-1

#     def freq_for_i(i):
#         """
#         Compute the 'freq' for a particular sub-interval i in [0..n_elements-1].
#         Translated directly from the original for-loop logic.
#         """
#         # Convert i to the angles [degrees -> radians]
#         tl1 = (i * step)     * jnp.pi / 180.0
#         tl2 = ((i + 1) * step) * jnp.pi / 180.0

#         # x1 and x2 from the original code
#         x1 = excent / jnp.sqrt(1.0 + excent**2 * jnp.tan(tl1)**2)
#         x2 = excent / jnp.sqrt(1.0 + excent**2 * jnp.tan(tl2)**2)

#         def freq_if_excent_eq_1(_):
#             # excent == 1 => freq = abs(cos(tl1) - cos(tl2))
#             return jnp.abs(jnp.cos(tl1) - jnp.cos(tl2))

#         def freq_if_excent_ne_1(_):
#             # We have two sub-branches: excent > 1 or excent < 1
#             def freq_if_excent_gt_1(_):
#                 # excent > 1
#                 # alph  = excent / sqrt(excent^2 - 1)
#                 alph = excent / jnp.sqrt(jnp.abs(1.0 - excent**2))
#                 alph2 = alph**2
#                 x12 = x1**2
#                 x22 = x2**2

#                 alpx1 = jnp.sqrt(alph2 + x12)
#                 alpx2 = jnp.sqrt(alph2 + x22)
#                 # dum for x1
#                 dum1 = x1 * alpx1 + alph2 * jnp.log(x1 + alpx1)
#                 # dum for x2
#                 dum2 = x2 * alpx2 + alph2 * jnp.log(x2 + alpx2)
#                 return jnp.abs(dum1 - dum2)

#             def freq_if_excent_lt_1(_):
#                 # excent < 1
#                 # alph  = excent / sqrt(1 - excent^2)
#                 alph = excent / jnp.sqrt(jnp.abs(1.0 - excent**2))
#                 alph2 = alph**2
#                 x12 = x1**2
#                 x22 = x2**2

#                 almx1 = jnp.sqrt(alph2 - x12)
#                 almx2 = jnp.sqrt(alph2 - x22)
#                 # dum for x1
#                 dum1 = x1 * almx1 + alph2 * jnp.arcsin(x1 / alph)
#                 # dum for x2
#                 dum2 = x2 * almx2 + alph2 * jnp.arcsin(x2 / alph)
#                 return jnp.abs(dum1 - dum2)

#             return lax.cond(excent > 1.0,
#                                 freq_if_excent_gt_1,
#                                 freq_if_excent_lt_1,
#                                 operand=None)

#         # top-level cond to check excent ~ 1 or not
#         # (float compare can be tricky, but we follow the original logic)
#         return lax.cond(
#             jnp.isclose(excent, 1.0, atol=1e-9, rtol=1e-9),
#             freq_if_excent_eq_1,
#             freq_if_excent_ne_1,
#             operand=None
#         )

#     # Vectorize freq_for_i over i_array
#     freq = jax.vmap(freq_for_i)(i_array)

#     # Normalize so that sum of freq = 1
#     sum0 = jnp.sum(freq)
#     lidf = freq / sum0
#     return lidf


# @jax.jit
# def foursail(rho, tau, lidfa, lidfb, lidftype, lai, hotspot,
#                     tts, tto, psi, rsoil):
#     """
#     JAX version of FourSAIL without lax.cond, using a mask approach.

#     - We compute "no canopy" outputs (shape = rsoil.shape).
#     - We compute "canopy" outputs (same shape).
#     - Then we combine them with a mask: "no_canopy_mask = (lai <= 0)".

#     The final return is a tuple of 21 arrays, all shape = rsoil.shape.
#     """

#     # Use rsoil's shape as our reference for all arrays:
#     ref_shape = rsoil.shape
#     dtype     = rsoil.dtype

#     # Create a mask: True where LAI <= 0, False where LAI > 0.
#     # But note that if lai is a scalar and rsoil is an array (3, ),
#     # we broadcast it. This is typically fine in JAX.
#     no_canopy_mask = (lai <= 0.0)

#     # ----------------------------------------------------------
#     # 1) NO CANOPY outputs (as arrays, shape = ref_shape)
#     # ----------------------------------------------------------
#     ones  = jnp.ones(ref_shape, dtype=dtype)
#     zeros = jnp.zeros(ref_shape, dtype=dtype)

#     tss_no      = ones
#     too_no      = ones
#     tsstoo_no   = ones
#     rdd_no      = zeros
#     tdd_no      = ones
#     rsd_no      = zeros
#     tsd_no      = zeros
#     rdo_no      = zeros
#     tdo_no      = zeros
#     rso_no      = zeros
#     rsos_no     = zeros
#     rsod_no     = zeros
#     # soil-based terms remain shaped like rsoil
#     rddt_no     = rsoil
#     rsdt_no     = rsoil
#     rdot_no     = rsoil
#     rsodt_no    = zeros
#     rsost_no    = rsoil
#     rsot_no     = rsoil
#     gammasdf_no = zeros
#     gammasdb_no = zeros
#     gammaso_no  = zeros

#     # ----------------------------------------------------------
#     # 2) CANOPY outputs (as arrays, shape = ref_shape)
#     # ----------------------------------------------------------
#     # Instead of calling a separate function, we just inline the canopy logic:
#     # We'll also note that if lai is scalar > 0, everything broadcasts to (ref_shape,).

#     # Geometry (assuming we have a define_geometric_constants JAX function).
#     cts, cto, ctscto, tants, tanto, cospsi, dso = \
#         define_geometric_constants(tts, tto, psi)

#     # LIDF
#     # For example, we might do an if-lidftype check once outside or always call verhoef_bimodal:
#     # If you truly have different LIDF shapes, you'd either unify them or keep them scalar, etc.
#     lidf = jax.lax.cond(
#         lidftype == 1,
#         lambda _: verhoef_bimodal(lidfa, lidfb, 18),
#         lambda _: campbell(lidfa, 18),
#         operand=None
#     )

#     ks, ko, bf, sob, sof = weighted_sum_over_lidf(lidf, tts, tto, psi)

#     sdb = 0.5 * (ks + bf)
#     sdf = 0.5 * (ks - bf)
#     dob = 0.5 * (ko + bf)
#     dof = 0.5 * (ko - bf)
#     ddb = 0.5 * (1.0 + bf)
#     ddf = 0.5 * (1.0 - bf)

#     sigb = jnp.maximum(ddb * rho + ddf * tau, 1e-36)
#     sigf = jnp.maximum(ddf * rho + ddb * tau, 1e-36)

#     att = 1.0 - sigf
#     m   = jnp.sqrt(att**2 - sigb**2)
#     sb  = sdb * rho + sdf * tau
#     sf  = sdf * rho + sdb * tau
#     vb  = dob * rho + dof * tau
#     vf  = dof * rho + dob * tau
#     w   = sob * rho + sof * tau

#     e1    = jnp.exp(-m * lai)
#     e2    = e1**2
#     rinf  = (att - m) / sigb
#     rinf2 = rinf**2
#     re    = rinf * e1
#     denom = 1.0 - rinf2 * e2

#     J1ks = Jfunc1(ks, m, lai)
#     J2ks = Jfunc2(ks, m, lai)
#     J1ko = Jfunc1(ko, m, lai)
#     J2ko = Jfunc2(ko, m, lai)

#     Pss = (sf + sb * rinf) * J1ks
#     Qss = (sf * rinf + sb) * J2ks
#     Pv  = (vf + vb * rinf) * J1ko
#     Qv  = (vf * rinf + vb) * J2ko

#     tdd_yes = (1.0 - rinf2) * e1 / denom
#     rdd_yes = rinf * (1.0 - e2) / denom
#     tsd_yes = (Pss - re * Qss) / denom
#     rsd_yes = (Qss - re * Pss) / denom
#     tdo_yes = (Pv - re * Qv) / denom
#     rdo_yes = (Qv - re * Pv) / denom

#     gammasdf_yes = (1.0 + rinf) * (J1ks - re * J2ks) / denom
#     gammasdb_yes = (1.0 + rinf) * (-re * J1ks + J2ks) / denom

#     tss_yes = jnp.exp(-ks * lai)
#     too_yes = jnp.exp(-ko * lai)

#     z   = Jfunc2(ks, ko, lai)
#     g1  = (z - J1ks * too_yes) / (ko + m)
#     g2  = (z - J1ko * tss_yes) / (ks + m)
#     Tv1 = (vf * rinf + vb) * g1
#     Tv2 = (vf + vb * rinf) * g2
#     T1  = Tv1 * (sf + sb * rinf)
#     T2  = Tv2 * (sf * rinf + sb)
#     T3  = (rdo_yes * Qss + tdo_yes * Pss) * rinf

#     rsod_yes = (T1 + T2 - T3) / (1.0 - rinf**2)

#     T4       = Tv1 * (1.0 + rinf)
#     T5       = Tv2 * (1.0 + rinf)
#     T6       = (rdo_yes * J2ks + tdo_yes * J1ks) * (1.0 + rinf) * rinf
#     gammasod_yes = (T4 + T5 - T6) / (1.0 - rinf**2)

#     # Hotspot
#     alf_init = 1e36
#     alf = jnp.where(hotspot > 0.0,
#                     (dso / hotspot) * 2.0 / (ks + ko),
#                     alf_init)

#     def pure_hotspot(_):
#         tss_   = jnp.exp(-ks * lai)
#         sumint = (1.0 - tss_) / (ks * lai)
#         return (tss_, sumint)

#     def outside_hotspot(_):
#         return hotspot_calculations(alf, lai, ko, ks)

#     tsstoo_yes, sumint = jax.lax.cond(
#         jnp.isclose(alf, 0.0, atol=1e-15),
#         pure_hotspot,
#         outside_hotspot,
#         operand=None
#     )

#     rsos_yes    = w * lai * sumint
#     gammasos_yes= ko * lai * sumint
#     rso_yes     = rsos_yes + rsod_yes
#     gammaso_yes = gammasos_yes + gammasod_yes

#     dn   = jnp.maximum(1.0 - rsoil * rdd_yes, 1e-36)
#     rddt_yes = rdd_yes + tdd_yes * rsoil * tdd_yes / dn
#     rsdt_yes = rsd_yes + (tsd_yes + tss_yes) * rsoil * tdd_yes / dn
#     rdot_yes = rdo_yes + tdd_yes * rsoil * (tdo_yes + too_yes) / dn
#     rsodt_yes= ((tss_yes + tsd_yes) * tdo_yes 
#                 + (tsd_yes + tss_yes * rsoil * rdd_yes) * too_yes
#                ) * rsoil / dn
#     rsost_yes = rso_yes + tsstoo_yes * rsoil
#     rsot_yes  = rsost_yes + rsodt_yes

#     # ----------------------------------------------------------
#     # 3) Combine with the mask
#     # ----------------------------------------------------------
#     # final_x = jnp.where(no_canopy_mask, x_no, x_yes) 
#     # for each variable.

#     tss      = jnp.where(no_canopy_mask, tss_no,      tss_yes)
#     too      = jnp.where(no_canopy_mask, too_no,      too_yes)
#     tsstoo   = jnp.where(no_canopy_mask, tsstoo_no,   tsstoo_yes)
#     rdd      = jnp.where(no_canopy_mask, rdd_no,      rdd_yes)
#     tdd      = jnp.where(no_canopy_mask, tdd_no,      tdd_yes)
#     rsd      = jnp.where(no_canopy_mask, rsd_no,      rsd_yes)
#     tsd      = jnp.where(no_canopy_mask, tsd_no,      tsd_yes)
#     rdo      = jnp.where(no_canopy_mask, rdo_no,      rdo_yes)
#     tdo      = jnp.where(no_canopy_mask, tdo_no,      tdo_yes)
#     rso      = jnp.where(no_canopy_mask, rso_no,      rso_yes)
#     rsos     = jnp.where(no_canopy_mask, rsos_no,     rsos_yes)
#     rsod     = jnp.where(no_canopy_mask, rsod_no,     rsod_yes)
#     rddt     = jnp.where(no_canopy_mask, rddt_no,     rddt_yes)
#     rsdt     = jnp.where(no_canopy_mask, rsdt_no,     rsdt_yes)
#     rdot     = jnp.where(no_canopy_mask, rdot_no,     rdot_yes)
#     rsodt    = jnp.where(no_canopy_mask, rsodt_no,    rsodt_yes)
#     rsost    = jnp.where(no_canopy_mask, rsost_no,    rsost_yes)
#     rsot     = jnp.where(no_canopy_mask, rsot_no,     rsot_yes)
#     gammasdf = jnp.where(no_canopy_mask, gammasdf_no, gammasdf_yes)
#     gammasdb = jnp.where(no_canopy_mask, gammasdb_no, gammasdb_yes)
#     gammaso  = jnp.where(no_canopy_mask, gammaso_no,  gammaso_yes)

#     return (tss[0], too[0], tsstoo[0],
#             rdd, tdd, rsd, tsd, rdo, tdo,
#             rso, rsos, rsod,
#             rddt, rsdt, rdot, rsodt, rsost, rsot,
#             gammasdf, gammasdb, gammaso)





# # rho    = jnp.array([0.05, 0.10, 0.15])
# # tau    = jnp.array([0.02, 0.03, 0.05])
# # rsoil  = jnp.array([0.3,  0.3,  0.3])

# # lidfa    = 1.0
# # lidfb    = 0.0
# # lidftype = 1   # Suppose we want verhoef_bimodal_jax
# # lai       = 3.0
# # hotspot   = 0.2
# # tts       = 30.0
# # tto       = 45.0
# # psi       = 10.0

# # results = foursail(rho, tau, lidfa, lidfb, lidftype, lai, hotspot,
# #                        tts, tto, psi, rsoil)
# # # 'results' is a tuple of length 21 with each sub-result.
# # print("FourSAIL results (21 items):", results)
