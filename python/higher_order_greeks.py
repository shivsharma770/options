#!/usr/bin/env python3
r"""
higher_order_greeks.py -- analytic Black--Scholes Greeks, first through third
order, for Research Milestone 4.

This is a *measurement* module: it computes closed-form sensitivities of the
Black--Scholes value V(S, sigma, tau, r) analytically, for use by the empirical
study. It does not modify or replace the C++ pricing engine; it is the Python
analogue used to compute the higher-order Greeks the engine does not expose, and
every formula is cross-checked against a finite-difference bump of the same BS
price (`finite_diff_greek`).

Conventions
-----------
* Continuous dividend yield ``q``; ``tau`` is time to maturity in years.
* ``d1 = (ln(S/K) + (r - q + sigma^2/2) tau) / (sigma sqrt(tau))``,
  ``d2 = d1 - sigma sqrt(tau)``.
* Vega, Vanna, Vomma, ... are reported per unit of volatility (i.e. per 1.00,
  not per vol-point); Charm/Veta/Color are per year of calendar time. Callers
  rescale (``*0.01`` per vol-point, ``/365`` per day) as needed.
* "t" (calendar time) increases as "tau" (maturity) decreases, so time-decay
  Greeks satisfy d/dt = -d/dtau. Charm/Veta/Color below follow the standard
  d/dt convention (Wikipedia "Greeks (finance)", Haug 2007).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

phi = norm.pdf
N = norm.cdf


def _d1d2(S, K, tau, sigma, r, q):
    S, K, tau, sigma = map(np.asarray, (S, K, tau, sigma))
    srt = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / srt
    d2 = d1 - srt
    return d1, d2, srt


def bs_price(S, K, tau, sigma, r=0.0, q=0.0, call=True):
    d1, d2, _ = _d1d2(S, K, tau, sigma, r, q)
    dfq, dfr = np.exp(-q * tau), np.exp(-r * tau)
    if call:
        return S * dfq * N(d1) - K * dfr * N(d2)
    return K * dfr * N(-d2) - S * dfq * N(-d1)


def greeks(S, K, tau, sigma, r=0.0, q=0.0, call=True):
    """Return a dict of first- and higher-order Greeks (arrays if inputs are)."""
    S, K, tau, sigma = map(lambda x: np.asarray(x, dtype=float), (S, K, tau, sigma))
    d1, d2, srt = _d1d2(S, K, tau, sigma, r, q)
    dfq, dfr = np.exp(-q * tau), np.exp(-r * tau)
    pdf1 = phi(d1)
    st = np.sqrt(tau)

    # --- first order ------------------------------------------------------
    delta = dfq * (N(d1) if call else N(d1) - 1.0)
    vega = S * dfq * pdf1 * st
    gamma = dfq * pdf1 / (S * srt)
    theta = (                                   # per year (d/dt = -d/dtau)
        -S * dfq * pdf1 * sigma / (2 * st)
        + (q * S * dfq * N(d1) - r * K * dfr * N(d2)) if call else
        -S * dfq * pdf1 * sigma / (2 * st)
        - (q * S * dfq * N(-d1) - r * K * dfr * N(-d2))
    )
    rho = (K * tau * dfr * (N(d2) if call else -N(-d2)))

    # --- second order -----------------------------------------------------
    vanna = -dfq * pdf1 * d2 / sigma            # d(delta)/d(sigma) = d(vega)/dS
    vomma = vega * d1 * d2 / sigma              # d(vega)/d(sigma)   (Volga)
    if call:
        charm = q * dfq * N(d1) - dfq * pdf1 * (2 * (r - q) * tau - d2 * srt) / (2 * tau * srt)
    else:
        charm = -q * dfq * N(-d1) - dfq * pdf1 * (2 * (r - q) * tau - d2 * srt) / (2 * tau * srt)
    veta = S * dfq * pdf1 * st * (q + (r - q) * d1 / srt - (1 + d1 * d2) / (2 * tau))

    # --- third order ------------------------------------------------------
    speed = -gamma / S * (d1 / srt + 1.0)       # d(gamma)/dS
    zomma = gamma * (d1 * d2 - 1.0) / sigma     # d(gamma)/d(sigma)
    color = dfq * pdf1 / (2 * S * tau * srt) * (
        2 * q * tau + 1.0 + (2 * (r - q) * tau - d2 * srt) / srt * d1)
    ultima = -vega / sigma ** 2 * (d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2)

    return dict(price=bs_price(S, K, tau, sigma, r, q, call),
                delta=delta, vega=vega, gamma=gamma, theta=theta, rho=rho,
                vanna=vanna, vomma=vomma, charm=charm, veta=veta,
                speed=speed, zomma=zomma, color=color, ultima=ultima)


# Map each Greek to the (order in S, order in sigma) it differentiates the
# price by, for the finite-difference validator (time Greeks handled by tau).
def finite_diff_greek(name, S, K, tau, sigma, r=0.0, q=0.0, call=True,
                      hS=1e-3, hv=1e-4, ht=1e-4):
    """Central finite-difference of the BS price, matching `greeks[name]`.
    Used only to validate the closed-form expressions."""
    def V(s=S, v=sigma, t=tau):
        return bs_price(s, K, t, v, r, q, call)
    dS = hS * S
    if name == "delta":
        return (V(s=S + dS) - V(s=S - dS)) / (2 * dS)
    if name == "vega":
        return (V(v=sigma + hv) - V(v=sigma - hv)) / (2 * hv)
    if name == "gamma":
        return (V(s=S + dS) - 2 * V() + V(s=S - dS)) / dS ** 2
    if name == "theta":              # d/dt = -d/dtau
        return -(V(t=tau + ht) - V(t=tau - ht)) / (2 * ht)
    if name == "vanna":
        return (V(s=S + dS, v=sigma + hv) - V(s=S + dS, v=sigma - hv)
                - V(s=S - dS, v=sigma + hv) + V(s=S - dS, v=sigma - hv)) / (4 * dS * hv)
    if name == "vomma":
        return (V(v=sigma + hv) - 2 * V() + V(v=sigma - hv)) / hv ** 2
    if name == "charm":              # d(delta)/dt = -d(delta)/dtau
        dp = (V(s=S + dS, t=tau + ht) - V(s=S - dS, t=tau + ht)) / (2 * dS)
        dm = (V(s=S + dS, t=tau - ht) - V(s=S - dS, t=tau - ht)) / (2 * dS)
        return -(dp - dm) / (2 * ht)
    if name == "veta":               # d(vega)/dt = -d(vega)/dtau
        vp = (V(v=sigma + hv, t=tau + ht) - V(v=sigma - hv, t=tau + ht)) / (2 * hv)
        vm = (V(v=sigma + hv, t=tau - ht) - V(v=sigma - hv, t=tau - ht)) / (2 * hv)
        return -(vp - vm) / (2 * ht)
    if name == "speed":
        return (V(s=S + 2 * dS) - 2 * V(s=S + dS) + 2 * V(s=S - dS) - V(s=S - 2 * dS)) / (2 * dS ** 3)
    if name == "zomma":              # d(gamma)/d(sigma)
        gp = (V(s=S + dS, v=sigma + hv) - 2 * V(v=sigma + hv) + V(s=S - dS, v=sigma + hv)) / dS ** 2
        gm = (V(s=S + dS, v=sigma - hv) - 2 * V(v=sigma - hv) + V(s=S - dS, v=sigma - hv)) / dS ** 2
        return (gp - gm) / (2 * hv)
    if name == "color":              # d(gamma)/dt = -d(gamma)/dtau
        gp = (V(s=S + dS, t=tau + ht) - 2 * V(t=tau + ht) + V(s=S - dS, t=tau + ht)) / dS ** 2
        gm = (V(s=S + dS, t=tau - ht) - 2 * V(t=tau - ht) + V(s=S - dS, t=tau - ht)) / dS ** 2
        return -(gp - gm) / (2 * ht)
    if name == "ultima":
        return (V(v=sigma + 2 * hv) - 2 * V(v=sigma + hv) + 2 * V(v=sigma - hv) - V(v=sigma - 2 * hv)) / (2 * hv ** 3)
    if name == "rho":
        return (bs_price(S, K, tau, sigma, r + hv, q, call)
                - bs_price(S, K, tau, sigma, r - hv, q, call)) / (2 * hv)
    raise ValueError(name)


if __name__ == "__main__":
    # Self-validation: analytic vs finite-difference at a representative point.
    S, K, tau, sig, r, q = 100.0, 105.0, 0.25, 0.20, 0.03, 0.01
    g = greeks(S, K, tau, sig, r, q, call=True)
    print(f"{'greek':>8} {'analytic':>14} {'finite-diff':>14} {'rel.err':>10}")
    for name in ["delta", "vega", "gamma", "theta", "rho", "vanna", "vomma",
                 "charm", "veta", "speed", "zomma", "color", "ultima"]:
        a = float(g[name]); f = float(finite_diff_greek(name, S, K, tau, sig, r, q))
        rel = abs(a - f) / (abs(f) + 1e-12)
        print(f"{name:>8} {a:>14.6g} {f:>14.6g} {rel:>10.2e}")
