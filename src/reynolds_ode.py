"""Reynolds et al. (2006) acute inflammation ODE system.

Reference: Reynolds A, Rubin J, Clermont G, Day J, Vodovotz Y, Ermentrout GB.
"A reduced mathematical model of the acute inflammatory response: I. Derivation of
model and analysis of anti-inflammation." J Theor Biol 242(1):220-36. (2006)
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


REYNOLDS_PARAMS = {
    # Pathogen (Eq. 9)
    'k_pg':    0.6,      # pathogen growth rate (above bistability threshold 0.5137)
    'P_inf':   20.0,     # pathogen carrying capacity
    'k_pm':    0.6,      # nonspecific M clearance coefficient
    's_m':     0.005,    # M production rate (quasi-steady-state)
    'mu_m':    0.002,    # M saturation constant
    'k_mp':    0.01,     # pathogen effect on M
    'k_pn':    1.8,      # N*-mediated clearance rate
    'c_1':     0.28,     # anti-inflammatory inhibition threshold

    # Early pro-inflammatory mediator N* (Eq. 10)
    's_nr':    0.08,     # max N* production rate
    'k_nn':    0.01,     # N* self-activation (positive feedback — essential for bistability)
    'k_np':    0.1,      # pathogen activation of N*
    'k_nd':    0.02,     # damage activation of N*
    'mu_nr':   0.12,     # Michaelis-Menten constant for N* production
    'mu_n':    0.05,     # activated N* decay rate

    # Late pro-inflammatory / DAMP mediator D (Eq. 11)
    'k_dn':    0.35,     # max DAMP production rate (from paper Table 1)
    'x_dn':    0.06,     # Hill function half-saturation for damage production
    'mu_d':    0.02,     # D decay rate

    # Anti-inflammatory mediator CA (Eq. 12)
    's_c':     0.0125,   # constitutive CA source (from paper)
    'k_cn':    0.04,     # max CA production coefficient
    'k_cnd':   48.0,     # D effectiveness relative to N* (from paper)
    'mu_c':    0.1,      # CA decay rate

    # Tissue damage fraction (phenomenological extension, not in paper)
    'k_f':     0.1,      # N*-driven damage rate
    'k_fh':    0.1,      # tissue repair rate (equal to k_f → equilibrium f* = N*)
}


def reynolds_ode(t: float, x: np.ndarray, params: Dict) -> np.ndarray:
    """
    Reynolds 2006 ODE system (Eqs. 9–12 with phenomenological f dynamics).

    State vector: x = [P, N*, D, CA, f]
    (h = 1 - f is derived algebraically, not integrated)

    References:
        Reynolds A, Rubin J, Clermont G, Day J, Vodovotz Y, Ermentrout GB.
        "A reduced mathematical model of the acute inflammatory response."
        J Theor Biol 242(1):220–36. (2006)

    Args:
        t: Current time (hours)
        x: State vector [P, N*, D, CA, f]
        params: Parameters dict

    Returns:
        dx/dt: Derivative vector [dP/dt, dN*/dt, dD/dt, dCA/dt, df/dt]
    """
    P, Nstar, D, CA, f = x
    p = params
    h = 1.0 - f

    # Shared anti-inflammatory inhibition function f(V) = V / (1 + (CA/c_1)²)
    def anti_inhib(V):
        return V / (1.0 + (CA / p['c_1'])**2)

    # dP/dt (Eq. 9): Pathogen dynamics
    # Growth - nonspecific M clearance - N*-mediated clearance
    M_ss = p['s_m'] / (p['mu_m'] + p['k_mp'] * P)  # quasi-steady-state M
    dP = (p['k_pg'] * P * (1 - P / p['P_inf'])
          - p['k_pm'] * M_ss * P
          - p['k_pn'] * anti_inhib(Nstar) * P)

    # dN*/dt (Eq. 10): Early pro-inflammatory mediator activation
    # R = stimulus (P, D, and POSITIVE FEEDBACK k_nn·N* — critical for bistability)
    R = p['k_nn'] * Nstar + p['k_np'] * P + p['k_nd'] * D
    fR = anti_inhib(R)
    dNstar = p['s_nr'] * fR / (p['mu_nr'] + fR) - p['mu_n'] * Nstar

    # dD/dt (Eq. 11): DAMP production via 6th-order Hill function
    # Hill function f_s(V) = V⁶/(x_dn⁶ + V⁶) provides switch-like behavior
    fNstar_D = anti_inhib(Nstar)
    hill6 = fNstar_D**6 / (p['x_dn']**6 + fNstar_D**6)
    dD = p['k_dn'] * hill6 - p['mu_d'] * D

    # dCA/dt (Eq. 12): Anti-inflammatory response
    # Both N* and D (via k_cnd=48) drive CA; constitutive source s_c
    stim = anti_inhib(Nstar + p['k_cnd'] * D)
    dCA = p['s_c'] + p['k_cn'] * stim / (1.0 + stim) - p['mu_c'] * CA

    # df/dt: Tissue damage fraction (phenomenological extension)
    # With k_f = k_fh, equilibrium is f* = Nstar, so f directly tracks inflammation
    df = p['k_f'] * Nstar * (1 - f) - p['k_fh'] * h * f

    return np.array([dP, dNstar, dD, dCA, df])


def solve_reynolds(
    params: Dict,
    x0: np.ndarray,
    t_eval: np.ndarray,
    method: str = 'Radau'
) -> Dict:
    """
    Solve the Reynolds ODE system.

    Args:
        params: Parameters dict
        x0: Initial conditions [P0, N*0, D0, CA0, f0]
        t_eval: Time points at which to evaluate solution
        method: ODE solver method ('Radau' for stiffness, 'RK45' for speed)

    Returns:
        Dictionary with keys:
            't': Time points
            'P', 'Nstar', 'D', 'CA', 'f': State variable trajectories
            'h': Derived health variable (1 - f)
            'success': Boolean flag indicating successful integration
            'message': Status message from solver
    """
    def system(t, x):
        return reynolds_ode(t, x, params)

    # Dense output for high-resolution evaluation
    sol = solve_ivp(
        system,
        t_span=(t_eval[0], t_eval[-1]),
        y0=x0,
        t_eval=t_eval,
        method=method,
        dense_output=True,
        events=None,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.25
    )

    if not sol.success:
        raise ValueError(f"ODE integration failed: {sol.message}")

    return {
        't': sol.t,
        'P': sol.y[0, :],
        'Nstar': sol.y[1, :],
        'D': sol.y[2, :],
        'CA': sol.y[3, :],
        'f': sol.y[4, :],
        'h': 1.0 - sol.y[4, :],
        'success': sol.success,
        'message': sol.message,
    }


def get_outcome(sol: Dict, tol: float = 1e-3) -> str:
    """
    Classify outcome based on paper's three attractors.

    Uses final values of P, N*, D to match paper's classification:
    - Health: P ≈ 0, N* ≈ 0, D ≈ 0
    - Aseptic death: P ≈ 0, but N* > 0 and/or D > 0
    - Septic death: P > 0, N* > 0, D > 0

    Args:
        sol: Solution dict from solve_reynolds with keys 't', 'P', 'Nstar', 'D', 'CA', 'f', 'h'
        tol: Threshold to distinguish zero from non-zero

    Returns:
        Outcome class: 'resolution', 'chronic', or 'death'
    """
    P_final = sol['P'][-1]
    Nstar_final = sol['Nstar'][-1]
    D_final = sol['D'][-1]

    if P_final < tol and Nstar_final < tol and D_final < tol:
        return 'resolution'        # health attractor
    elif P_final < tol and (Nstar_final >= tol or D_final >= tol):
        return 'chronic'           # aseptic death attractor
    else:
        return 'death'             # septic death attractor (P > tol)


def plot_phase_portrait(ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot phase portrait of outcome basins in P0 vs f(72h) space.

    Solves the system for a range of initial pathogen loads and plots
    the basin structure showing the three outcome regions.

    Args:
        ax: Matplotlib axes object (creates new if None)

    Returns:
        Axes object with phase portrait
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Grid of initial pathogen loads
    P0_values = np.linspace(0.1, 25.0, 100)
    f_final_values = []
    outcomes = []

    # Initial conditions (all other variables start at 0)
    x0_base = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    t_eval = np.linspace(0, 72, 500)

    for P0 in P0_values:
        x0 = x0_base.copy()
        x0[0] = P0
        try:
            sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)
            f_final = sol['f'][-1]
            f_final_values.append(f_final)
            outcomes.append(get_outcome(sol))
        except ValueError:
            f_final_values.append(np.nan)
            outcomes.append('unknown')

    # Color by outcome
    colors = {
        'resolution': '#2ECC71',
        'chronic': '#F39C12',
        'death': '#E74C3C',
        'unknown': '#95A5A6'
    }

    for outcome in ['resolution', 'chronic', 'death', 'unknown']:
        mask = np.array(outcomes) == outcome
        ax.scatter(
            P0_values[mask],
            np.array(f_final_values)[mask],
            color=colors[outcome],
            label=outcome,
            s=50,
            alpha=0.7
        )

    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Initial Pathogen Load (P₀)', fontsize=12)
    ax.set_ylabel('Final Tissue Damage (f at t=72h)', fontsize=12)
    ax.set_title('Reynolds Model: Outcome Basins', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    return ax


def plot_trajectories(
    solutions: list,
    ax: Optional[plt.Axes] = None,
    title: str = 'Example Trajectories'
) -> plt.Axes:
    """
    Plot example trajectories for multiple solutions.

    Args:
        solutions: List of solution dicts from solve_reynolds
        ax: Matplotlib axes object (creates new if None)
        title: Plot title

    Returns:
        Axes object with trajectories
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    colors_outcome = {
        'resolution': '#2ECC71',
        'chronic': '#F39C12',
        'death': '#E74C3C',
    }

    variables = ['Nstar', 'CA', 'f']
    linestyles = {
        'Nstar': '-',
        'CA': '--',
        'f': ':',
    }

    for sol in solutions:
        outcome = get_outcome(sol)
        color = colors_outcome[outcome]

        for var in variables:
            ax.plot(
                sol['t'],
                sol[var],
                linestyle=linestyles[var],
                color=color,
                linewidth=2,
                label=f"{var} ({outcome})",
                alpha=0.8
            )

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('State Variable Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


if __name__ == '__main__':
    # Test: Reproduce Figure 5 canonical scenarios
    print("Testing Reynolds ODE with paper's Figure 5 scenarios...\n")

    # CA baseline at health equilibrium
    ca_health = REYNOLDS_PARAMS['s_c'] / REYNOLDS_PARAMS['mu_c']  # 0.125

    # Test scenarios from Figure 5: (kpg, P0) pairs
    test_scenarios = {
        'resolution': {'kpg': 0.3, 'P0': 1.0},
        'chronic':    {'kpg': 0.3, 'P0': 1.5},
        'death':      {'kpg': 0.6, 'P0': 1.0},
    }
    test_solutions = {}

    # Integrate to 1000h for reliable outcome classification (paper converges slower)
    t_eval = np.linspace(0, 1000, 4001)

    for expected_outcome, scenario in test_scenarios.items():
        kpg = scenario['kpg']
        P0 = scenario['P0']

        # Set up parameters with scenario-specific kpg
        params = REYNOLDS_PARAMS.copy()
        params['k_pg'] = kpg

        # Initial condition: [P0, N*=0, D=0, CA=health_baseline, f=0]
        x0 = np.array([P0, 0.0, 0.0, ca_health, 0.0])

        try:
            sol = solve_reynolds(params, x0, t_eval)
            actual_outcome = get_outcome(sol)
            test_solutions[expected_outcome] = sol

            status = "[PASS]" if actual_outcome == expected_outcome else "[FAIL]"
            print(f"{status} kpg={kpg:.1f}, P0={P0:4.1f} -> {actual_outcome:12s} (expected {expected_outcome})")
            print(f"       Final: P={sol['P'][-1]:.4f}, N*={sol['Nstar'][-1]:.4f}, D={sol['D'][-1]:.4f}")
        except Exception as e:
            print(f"[FAIL] kpg={kpg:.1f}, P0={P0:4.1f} -> ERROR: {e}")

    # Plot both figures
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_phase_portrait(axes[0])
    plot_trajectories(
        list(test_solutions.values()),
        axes[1],
        title='Representative Outcome Trajectories'
    )

    plt.tight_layout()
    plt.savefig('reynolds_ode_test.png', dpi=150, bbox_inches='tight')
    print("[PASS] Saved test plot: reynolds_ode_test.png")
    print("\nStep 2: Reynolds ODE implementation complete.")
