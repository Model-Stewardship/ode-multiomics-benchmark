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
    # Pathogen dynamics
    'k_pg':    0.5,      # pathogen growth rate
    'P_inf':   20.0,     # pathogen carrying capacity
    'k_pm':    1.8,      # max pathogen clearance rate by N*
    's_dm':    0.05,     # half-saturation of N* clearance of pathogen
    'mu_p':    0.2,      # pathogen natural decay

    # Early pro-inflammatory mediator (N*)
    's_nr':    0.08,     # max production rate of N*
    'epsilon_nr': 0.32,  # anti-inflammatory suppression threshold (CA)
    's_nm':    0.1,      # half-saturation pathogen activation of N*
    's_nd':    0.1,      # half-saturation damage activation of N*
    'N_inf':   0.6,      # N* carrying capacity (max activation)
    'mu_nr':   0.12,     # N* decay rate

    # Late pro-inflammatory / DAMP mediator (D)
    'k_dn':    0.02,     # N*-driven DAMP production
    'k_df':    0.02,     # tissue-damage-driven DAMP production
    'mu_d':    0.05,     # D decay rate

    # Anti-inflammatory mediator (CA)
    's_c':     0.04,     # max CA production rate
    's_cn':    0.08,     # half-saturation of N* driving CA
    'mu_c':    0.1,      # CA decay rate

    # Tissue damage
    'k_f':     0.01,     # N*-driven tissue damage rate
    'k_fh':    0.12,     # tissue repair rate (health-dependent)
}


def reynolds_ode(t: float, x: np.ndarray, params: Dict) -> np.ndarray:
    """
    Reynolds 2006 ODE system.

    State vector: x = [P, N*, D, CA, f]
    (h = 1 - f is derived algebraically, not integrated)

    Args:
        t: Current time (hours)
        x: State vector [P, N*, D, CA, f]
        params: Parameters dict

    Returns:
        dx/dt: Derivative vector [dP/dt, dN*/dt, dD/dt, dCA/dt, df/dt]
    """
    P, Nstar, D, CA, f = x
    p = params

    # Derived quantity
    h = 1.0 - f

    # dP/dt: Pathogen dynamics
    dP = (p['k_pg'] * P * (1 - P / p['P_inf'])
          - p['k_pm'] * (Nstar / (p['s_dm'] + Nstar)) * P
          - p['mu_p'] * P)

    # dN*/dt: Early pro-inflammatory mediator
    dNstar = (p['s_nr'] / (1 + (CA / p['epsilon_nr'])**2)
              * (P / (p['s_nm'] + P) + D / (p['s_nd'] + D))
              * (1 - Nstar / p['N_inf'])
              - p['mu_nr'] * Nstar)

    # dD/dt: Late pro-inflammatory / DAMP mediator
    dD = p['k_dn'] * Nstar + p['k_df'] * f - p['mu_d'] * D

    # dCA/dt: Anti-inflammatory mediator
    dCA = p['s_c'] * (Nstar**2) / (p['s_cn']**2 + Nstar**2) - p['mu_c'] * CA

    # df/dt: Tissue damage fraction
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
        max_step=np.inf
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


def get_outcome(f_final: float) -> str:
    """
    Classify outcome based on final tissue damage f(t=72h).

    Args:
        f_final: Tissue damage fraction at t=72h

    Returns:
        Outcome class: 'resolution', 'chronic', or 'death'
    """
    if f_final < 0.1:
        return 'resolution'
    elif f_final < 0.5:
        return 'chronic'
    else:
        return 'death'


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
            outcomes.append(get_outcome(f_final))
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
        outcome = get_outcome(sol['f'][-1])
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
    # Test: Verify the three outcome trajectories
    print("Testing Reynolds ODE with canonical initial pathogen loads...")

    test_P0s = {'resolution': 2.0, 'chronic': 7.0, 'death': 15.0}
    test_solutions = {}

    t_eval = np.linspace(0, 72, 1000)

    for expected_outcome, P0 in test_P0s.items():
        x0 = np.array([P0, 0.0, 0.0, 0.0, 0.0])
        try:
            sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)
            actual_outcome = get_outcome(sol['f'][-1])
            test_solutions[expected_outcome] = sol

            status = "[PASS]" if actual_outcome == expected_outcome else "[FAIL]"
            print(f"{status} P0={P0:5.1f} -> {actual_outcome:10s} (expected {expected_outcome})")
            print(f"       f(72h)={sol['f'][-1]:.3f}, integration success={sol['success']}")
        except Exception as e:
            print(f"[FAIL] P0={P0:5.1f} -> ERROR: {e}")

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
