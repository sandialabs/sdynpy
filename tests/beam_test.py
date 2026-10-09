"""Tests for the rectangular beam properties in sdynpy.fem.sdynpy_beam."""
import numpy as np
import pytest
from scipy.linalg import eigh

from sdynpy.fem.sdynpy_beam import beamkm, rect_beam_props

E, RHO, NU = 70e9, 2700.0, 0.33

# St. Venant torsion coefficient beta in J = beta * a * c**3 (a long side, c short side),
# from the classical tables (Timoshenko & Goodier, Roark) for several side ratios a/c
TORSION_BETA = {1.0: 0.1406, 1.5: 0.1958, 2.0: 0.2287, 3.0: 0.2633, 10.0: 0.3123}


def torsion_constant(props, nu=NU):
    """Recover J from the returned jg using G = E / (2 (1 + nu))."""
    return props['jg'] / (E / (2 * (1 + nu)))


@pytest.mark.parametrize("ratio, beta", list(TORSION_BETA.items()))
def test_torsion_constant_matches_the_classical_table(ratio, beta):
    c = 0.01
    a = ratio * c
    props = rect_beam_props(E, RHO, NU, a, c)
    assert torsion_constant(props) == pytest.approx(beta * a * c**3, rel=2e-3)


def test_torsion_constant_does_not_depend_on_which_side_is_b():
    assert rect_beam_props(E, RHO, NU, 0.01, 0.03)['jg'] == pytest.approx(
        rect_beam_props(E, RHO, NU, 0.03, 0.01)['jg'])


def test_shear_modulus_uses_one_plus_poissons_ratio():
    b = 0.02
    j_square = 0.1406 * b**4
    props = rect_beam_props(E, RHO, 0.33, b)
    assert props['jg'] == pytest.approx(j_square * E / (2 * 1.33), rel=2e-3)
    # G(nu=0) / G(nu=0.5) = 3 / 2 with (1 + nu); the old (1 - nu) gave 1 / 2
    ratio = rect_beam_props(E, RHO, 0.0, b)['jg'] / rect_beam_props(E, RHO, 0.5, b)['jg']
    assert ratio == pytest.approx(1.5)


def test_torsion_constant_is_below_the_polar_moment():
    b, h = 0.01, 0.03
    j = torsion_constant(rect_beam_props(E, RHO, NU, b, h))
    polar = b * h * (b**2 + h**2) / 12
    assert j < polar
    assert polar / j == pytest.approx(3.17, rel=0.02)   # 3.17x too high for a 1:3 rectangle


def test_other_properties_are_unchanged():
    b, h = 0.01, 0.03
    props = rect_beam_props(E, RHO, NU, b, h)
    assert props['ae'] == pytest.approx(E * b * h)
    assert props['ei1'] == pytest.approx(E * b * h**3 / 12)
    assert props['ei2'] == pytest.approx(E * b**3 * h / 12)
    assert props['mass_per_length'] == pytest.approx(RHO * b * h)
    assert props['tmmi_per_length'] == pytest.approx(RHO * b * h * (b**2 + h**2) / 12)


def test_nelem_repeats_the_values():
    props = rect_beam_props(E, RHO, NU, 0.01, 0.02, nelem=5)
    assert all(v.shape == (5,) for v in props.values())


def test_cantilever_first_torsion_frequency_matches_the_closed_form():
    """1 m cantilever, 10 mm x 30 mm: f = sqrt(G J / (rho Ip)) / (4 L) for the first torsion mode."""
    b, h, length, nel = 0.01, 0.03, 1.0, 40
    x = np.linspace(0, length, nel + 1)
    nodes = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    conn = np.column_stack([np.arange(nel), np.arange(1, nel + 1)])
    bend1 = np.tile([0.0, 1.0, 0.0], (nel, 1))
    k, m = beamkm(nodes, conn, bend1, **rect_beam_props(E, RHO, NU, b, h, nel))
    free = np.arange(6, k.shape[0])                       # clamp the first node
    w2, vecs = eigh(k[np.ix_(free, free)], m[np.ix_(free, free)])
    freqs = np.sqrt(np.abs(w2)) / (2 * np.pi)

    first_torsion = None
    for i in range(len(freqs)):
        parts = np.linalg.norm(vecs[:, i].reshape(-1, 6), axis=0)
        if parts[3] > 0 and max(parts[:3]) < 1e-6 * parts[3]:
            first_torsion = freqs[i]
            break

    g = E / (2 * (1 + NU))
    j = TORSION_BETA[3.0] * h * b**3
    polar_inertia = RHO * b * h * (b**2 + h**2) / 12
    expected = np.sqrt(g * j / polar_inertia) / (4 * length)
    assert first_torsion == pytest.approx(expected, rel=5e-3)
