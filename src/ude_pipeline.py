"""UDE + SINDy Pipeline: Neural ODE learning and symbolic equation recovery.

Reference: Rackauckas et al. (2021) Universal Differential Equations.
Learns unknown rate functions from observed multiomics data via hybrid NN + known dynamics,
then recovers interpretable equations via SINDy symbolic regression.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import warnings

# Try to import torchdiffeq; provide fallback if unavailable
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

# Try to import pysindy; provide fallback if unavailable
try:
    import pysindy as ps
    HAS_PYSINDY = True
except ImportError:
    HAS_PYSINDY = False

from src.reynolds_ode import REYNOLDS_PARAMS

warnings.filterwarnings('ignore')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UDENet(nn.Module):
    """
    Neural network term for Universal Differential Equations.

    Learns the unknown pathogen clearance rate as a function of observable state.
    Input: [N*, CA, f] (3 observed variables)
    Output: scalar correction to pathogen clearance rate
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),  # Ensure non-negative output (rate)
        )
        # Initialize weights to be small for stability
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Observed state [N*, CA, f] of shape (..., 3)

        Returns:
            Scalar clearance rate correction of shape (...)
        """
        return self.net(z)


def ude_system(
    t: float,
    x: torch.Tensor,
    nn_model: nn.Module,
    known_params: Dict,
) -> torch.Tensor:
    """
    UDE ODE system: known dynamics + learned neural network term.

    State: x = [P, N*, D, CA, f]
    - P: unobserved pathogen (but estimated per patient)
    - N*, CA, f: observed
    - D: unobserved but derived from known biology

    The neural network learns the pathogen clearance rate as f(N*, CA, f).

    Args:
        t: Current time (unused but required by torchdiffeq)
        x: State vector [P, N*, D, CA, f]
        nn_model: Neural network for learned term
        known_params: Fixed biological parameters

    Returns:
        dx/dt: Derivative vector [dP/dt, dN*/dt, dD/dt, dCA/dt, df/dt]
    """
    # Clamp states to valid ranges to prevent numerical issues
    x = torch.clamp(x, min=0.0)
    x = x.clone()

    P, Nstar, D, CA, f = x[0], x[1], x[2], x[3], x[4]
    f = torch.clamp(f, 0.0, 1.0)  # Tissue damage fraction must be in [0, 1]
    p = known_params

    # Derived quantity
    h = 1.0 - f

    # Observed state fed to neural network
    z = torch.stack([Nstar, CA, f], dim=-1)
    # Ensure z has batch dimension for NN
    if z.dim() == 1:
        z = z.unsqueeze(0)
    nn_clearance = nn_model(z)
    # Squeeze all dimensions except the last to get scalar
    nn_clearance = nn_clearance.squeeze()

    # dP/dt: Pathogen dynamics with learned clearance
    dP = (p['k_pg'] * P * (1 - P / p['P_inf'])
          - nn_clearance * P  # Learned term replaces mechanistic clearance
          - p['mu_p'] * P)

    # dN*/dt: Known early pro-inflammatory mediator dynamics
    dNstar = (p['s_nr'] / (1 + (CA / p['epsilon_nr'])**2)
              * (P / (p['s_nm'] + P) + D / (p['s_nd'] + D))
              * (1 - Nstar / p['N_inf'])
              - p['mu_nr'] * Nstar)

    # dD/dt: Known late pro-inflammatory mediator dynamics
    dD = p['k_dn'] * Nstar + p['k_df'] * f - p['mu_d'] * D

    # dCA/dt: Known anti-inflammatory mediator dynamics
    dCA = p['s_c'] * (Nstar**2) / (p['s_cn']**2 + Nstar**2) - p['mu_c'] * CA

    # df/dt: Known tissue damage dynamics
    df = p['k_f'] * Nstar * (1 - f) - p['k_fh'] * h * f

    # Ensure all have same shape for stacking
    dP = dP.squeeze() if dP.dim() > 0 else dP
    dNstar = dNstar.squeeze() if dNstar.dim() > 0 else dNstar
    dD = dD.squeeze() if dD.dim() > 0 else dD
    dCA = dCA.squeeze() if dCA.dim() > 0 else dCA
    df = df.squeeze() if df.dim() > 0 else df

    # Replace NaN/Inf with zeros to avoid breaking ODE integration
    for var in [dP, dNstar, dD, dCA, df]:
        if torch.isnan(var).any() or torch.isinf(var).any():
            var = torch.where(torch.isnan(var) | torch.isinf(var), torch.zeros_like(var), var)

    return torch.stack([dP, dNstar, dD, dCA, df], dim=0)


def train_ude(
    patients: List[Dict],
    nn_model: nn.Module,
    known_params: Dict,
    config: Optional[Dict] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train UDE neural network on patient cohort.

    Args:
        patients: List of patient dicts with obs_Nstar, obs_CA, obs_f, obs_t
        nn_model: UDENet instance to train
        known_params: Fixed biological parameters
        config: Training configuration dict
        verbose: Print progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    if not HAS_TORCHDIFFEQ:
        return nn_model, {'error': 'torchdiffeq not available'}

    if config is None:
        config = {
            'n_epochs': 500,
            'lr': 1e-3,
            'batch_size': 16,
            'max_grad_norm': 1.0,
            'random_seed': 42,
        }

    torch.manual_seed(config.get('random_seed', 42))
    np.random.seed(config.get('random_seed', 42))

    nn_model = nn_model.to(DEVICE)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=config.get('lr', 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )

    history = {'epoch': [], 'loss': [], 'lr': []}

    n_epochs = config.get('n_epochs', 500)
    if verbose:
        pbar = tqdm(range(n_epochs), desc='UDE Training')
    else:
        pbar = range(n_epochs)

    n_failed_ode = 0
    first_epoch_printed = False

    for epoch in pbar:
        # Shuffle patients
        shuffled_idx = np.random.permutation(len(patients))
        batch_size = config.get('batch_size', 16)

        total_loss = 0.0
        n_batches = 0
        n_epoch_failed = 0
        valid_losses = []

        for batch_start in range(0, len(patients), batch_size):
            batch_idx = shuffled_idx[batch_start:batch_start + batch_size]
            batch_loss = 0.0
            batch_count = 0

            for idx in batch_idx:
                patient = patients[idx]

                # Prepare data
                t_obs = torch.tensor(patient['obs_t'], dtype=torch.float32, device=DEVICE)
                obs_Nstar = torch.tensor(patient['obs_Nstar'], dtype=torch.float32, device=DEVICE)
                obs_CA = torch.tensor(patient['obs_CA'], dtype=torch.float32, device=DEVICE)
                obs_f = torch.tensor(patient['obs_f'], dtype=torch.float32, device=DEVICE)

                # Check for NaN in input data
                if torch.isnan(obs_Nstar).any() or torch.isnan(obs_CA).any() or torch.isnan(obs_f).any():
                    if epoch == 0 and verbose:
                        print(f"Warning: Input data contains NaN for patient {idx}")
                    n_epoch_failed += 1
                    continue

                # Initial condition: true initial state (P0, Nstar=0, D=0, CA=0, f=0)
                P0_init = torch.tensor(patient['P0'], dtype=torch.float32, device=DEVICE)
                x0 = torch.stack([P0_init,
                                  torch.tensor(0.0, device=DEVICE),  # Nstar(0) = 0
                                  torch.tensor(0.0, device=DEVICE),  # D(0) = 0
                                  torch.tensor(0.0, device=DEVICE),  # CA(0) = 0
                                  torch.tensor(0.0, device=DEVICE)], dim=0)  # f(0) = 0

                # Convert known_params to torch tensors
                torch_params = {k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
                               for k, v in known_params.items()}

                # Solve UDE forward
                try:
                    def ode_func(t, x):
                        return ude_system(t, x, nn_model, torch_params)

                    pred_traj = odeint(ode_func, x0, t_obs, method='rk4', rtol=1e-3, atol=1e-4)
                except Exception as e:
                    if epoch == 0 and verbose:
                        print(f"ODE integration failed for patient {idx}: {e}")
                    n_epoch_failed += 1
                    n_failed_ode += 1
                    continue

                # Extract observed predictions
                obs_pred_Nstar = pred_traj[:, 1]
                obs_pred_CA = pred_traj[:, 3]
                obs_pred_f = pred_traj[:, 4]

                # Check for NaN in predictions
                if torch.isnan(obs_pred_Nstar).any() or torch.isnan(obs_pred_CA).any() or torch.isnan(obs_pred_f).any():
                    if epoch == 0 and verbose:
                        print(f"Warning: NaN in predictions for patient {idx}")
                    n_epoch_failed += 1
                    continue

                # Loss on observed variables only
                loss = (F.mse_loss(obs_pred_Nstar, obs_Nstar) +
                       F.mse_loss(obs_pred_CA, obs_CA) +
                       F.mse_loss(obs_pred_f, obs_f))

                if torch.isnan(loss) or torch.isinf(loss):
                    if epoch == 0 and verbose:
                        print(f"Warning: NaN/Inf loss for patient {idx}: {loss.item()}")
                        print(f"  Nstar pred range: [{obs_pred_Nstar.min():.4f}, {obs_pred_Nstar.max():.4f}]")
                        print(f"  CA pred range: [{obs_pred_CA.min():.4f}, {obs_pred_CA.max():.4f}]")
                        print(f"  f pred range: [{obs_pred_f.min():.4f}, {obs_pred_f.max():.4f}]")
                    n_epoch_failed += 1
                    continue

                batch_loss += loss
                batch_count += 1
                n_batches += 1
                valid_losses.append(loss.item())

            if batch_count > 0:
                batch_loss = batch_loss / batch_count
                total_loss += batch_loss.item()

                optimizer.zero_grad()
                batch_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    nn_model.parameters(),
                    max_norm=config.get('max_grad_norm', 1.0)
                )

                optimizer.step()

        # Average loss over epoch
        if len(valid_losses) > 0:
            avg_loss = np.mean(valid_losses)
        else:
            avg_loss = float('nan')
            if verbose:
                print(f"Epoch {epoch}: No valid training samples. Failed ODE: {n_epoch_failed}/{len(patients)}, Failed loss: {len([x for batch_start in range(0, len(patients), batch_size) for idx in shuffled_idx[batch_start:batch_start + batch_size]])}")
            if epoch == 0 and verbose:
                print("ERROR: UDE training failed - no valid patient trajectories computed")
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'epoch': f'{epoch + 1}/{n_epochs}'})

    if verbose:
        pbar.close()

    return nn_model, history


def extract_sindy_equation(
    nn_model: nn.Module,
    state_range: Dict[str, Tuple[float, float]],
    sindy_config: Optional[Dict] = None,
    verbose: bool = False,
) -> Tuple[Optional[object], Dict]:
    """
    Extract symbolic equation from learned neural network via SINDy.

    Args:
        nn_model: Trained UDENet
        state_range: Dict with min/max for each state variable
                     e.g., {'Nstar': (0, 1), 'CA': (0, 1), 'f': (0, 1)}
        sindy_config: SINDy configuration
        verbose: Print progress

    Returns:
        Tuple of (sindy_model, results_dict)
    """
    if not HAS_PYSINDY:
        return None, {'error': 'pysindy not available'}

    if sindy_config is None:
        sindy_config = {'degree': 2, 'threshold': 0.05}

    # Create grid of states
    npoints = 20
    Nstar_range = state_range.get('Nstar', (0, 1))
    CA_range = state_range.get('CA', (0, 1))
    f_range = state_range.get('f', (0, 1))

    Nstar_vals = np.linspace(Nstar_range[0], Nstar_range[1], npoints)
    CA_vals = np.linspace(CA_range[0], CA_range[1], npoints)
    f_vals = np.linspace(f_range[0], f_range[1], npoints)

    # Sample random points in state space
    np.random.seed(42)
    n_samples = npoints ** 3
    Nstar_samp = np.random.uniform(Nstar_range[0], Nstar_range[1], n_samples)
    CA_samp = np.random.uniform(CA_range[0], CA_range[1], n_samples)
    f_samp = np.random.uniform(f_range[0], f_range[1], n_samples)

    state_grid = np.column_stack([Nstar_samp, CA_samp, f_samp])

    # Evaluate neural network on grid
    with torch.no_grad():
        z_tensor = torch.tensor(state_grid, dtype=torch.float32, device=DEVICE)
        nn_outputs = nn_model(z_tensor).cpu().numpy().flatten()

    # SINDy fitting
    try:
        feature_names = ['Nstar', 'CA', 'f']
        lib = ps.PolynomialLibrary(degree=sindy_config.get('degree', 2), include_interaction=True)

        sindy_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=sindy_config.get('threshold', 0.05)),
            feature_library=lib,
            feature_names=feature_names,
        )
        sindy_model.fit(state_grid, x_dot=nn_outputs)

        if verbose:
            print("\nSINDy Recovered Equation:")
            sindy_model.print()

        # Compute fit quality
        nn_pred_grid = sindy_model.predict(state_grid)
        r2_sindy = 1 - np.sum((nn_outputs - nn_pred_grid) ** 2) / np.sum((nn_outputs - np.mean(nn_outputs)) ** 2)

        results = {
            'r2': float(r2_sindy),
            'n_features': len(sindy_model.coefficients()[0]),
            'equation': sindy_model.equations(),
        }

        return sindy_model, results

    except Exception as e:
        if verbose:
            print(f"SINDy fitting failed: {e}")
        return None, {'error': str(e)}


def run_ude_pipeline(
    cohort: List[Dict],
    known_params: Dict = None,
    config: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run complete UDE + SINDy pipeline on patient cohort.

    Args:
        cohort: List of patient dicts
        known_params: Fixed biological parameters (uses REYNOLDS_PARAMS if None)
        config: Configuration dict
        verbose: Print progress

    Returns:
        Results dict with trained model, history, and recovered equations
    """
    if not HAS_TORCHDIFFEQ:
        return {'error': 'torchdiffeq not available'}

    if known_params is None:
        known_params = REYNOLDS_PARAMS.copy()

    if config is None:
        config = {
            'nn_hidden_dim': 32,
            'n_epochs': 500,
            'lr': 1e-3,
            'batch_size': 16,
            'max_grad_norm': 1.0,
            'random_seed': 42,
            'sindy_degree': 2,
            'sindy_threshold': 0.05,
        }

    results = {
        'config': config,
        'n_patients': len(cohort),
    }

    if verbose:
        print(f"Running UDE pipeline on {len(cohort)} patients...")

    # Step 1: Build neural network
    if verbose:
        print("  Step 1: Initializing neural network...")

    nn_model = UDENet(input_dim=3, hidden_dim=config.get('nn_hidden_dim', 32), output_dim=1)
    results['nn_model'] = nn_model

    # Step 2: Train UDE
    if verbose:
        n_epochs = config.get('n_epochs', 500)
        batch_size = config.get('batch_size', 16)
        lr = config.get('lr', 1e-3)
        print(f"  Step 2: Training UDE (epochs={n_epochs}, batch_size={batch_size}, lr={lr})...")

    trained_model, history = train_ude(
        cohort,
        nn_model,
        known_params,
        config=config,
        verbose=verbose,
    )

    results['training_history'] = history
    results['nn_model'] = trained_model

    if 'error' in history:
        if verbose:
            print(f"  Training failed: {history['error']}")
        return results

    # Step 3: SINDy equation extraction
    if verbose:
        print("  Step 3: Extracting symbolic equations via SINDy...")

    state_range = {
        'Nstar': (0, 0.8),
        'CA': (0, 0.5),
        'f': (0, 1.0),
    }

    sindy_config = {
        'degree': config.get('sindy_degree', 2),
        'threshold': config.get('sindy_threshold', 0.05),
    }

    sindy_model, sindy_results = extract_sindy_equation(
        trained_model,
        state_range,
        sindy_config,
        verbose=verbose,
    )

    results['sindy_model'] = sindy_model
    results['sindy_results'] = sindy_results

    if verbose:
        print("  UDE pipeline complete.")

    return results


if __name__ == '__main__':
    print("Testing UDE pipeline implementation...")

    if not HAS_TORCHDIFFEQ:
        print("ERROR: torchdiffeq not available. Install with: pip install torchdiffeq")
    else:
        # Load a small cohort for testing
        from src.synthetic_data import generate_cohort

        cohort_small = generate_cohort(N_resolution=3, N_chronic=2, N_death=2, seed=42, verbose=False)

        print(f"Running UDE on {len(cohort_small)} test patients...")

        config = {
            'nn_hidden_dim': 16,
            'n_epochs': 20,  # Short for testing
            'lr': 1e-2,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
            'sindy_degree': 2,
            'sindy_threshold': 0.05,
        }

        results = run_ude_pipeline(cohort_small, config=config, verbose=True)

        print("\nUDE Pipeline Results:")
        print(f"  Model: {type(results['nn_model'])}")
        print(f"  Training history: {len(results['training_history'].get('loss', []))} epochs")
        if 'loss' in results['training_history']:
            print(f"  Final loss: {results['training_history']['loss'][-1]:.6f}")

        if 'sindy_results' in results:
            sindy = results['sindy_results']
            if 'error' not in sindy:
                print(f"  SINDy R²: {sindy.get('r2', 'N/A')}")
                print(f"  SINDy n_features: {sindy.get('n_features', 'N/A')}")

    print("\nStep 7: UDE pipeline implementation complete.")
