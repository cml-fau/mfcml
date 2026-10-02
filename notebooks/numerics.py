"""Fixed-step ODE integrators shared by the course notebooks.

These are the schemes derived in notebook 02, section 2: one explicit Euler step
(global error O(h)), one classical Runge-Kutta step (global error O(h^4)), and the
loop that walks either of them along a time grid.  Both steppers take the same
arguments, so `odeint` can be handed whichever one a section wants to illustrate.

Everything is written for torch tensors, so a trajectory stays differentiable with
respect to the field's parameters and to x0 -- which is what makes `odeint` usable
as the forward pass of a neural ODE (notebook 02) and as the state equation of an
optimal control problem (notebook 05).
"""

import torch

__all__ = ["euler_step", "rk4_step", "odeint"]


def euler_step(f, t, x, h):
    """One explicit Euler step: follow the tangent at the current point."""
    return x + h * f(t, x)


def rk4_step(f, t, x, h):
    """One classical RK4 step: four slope evaluations, combined 1:2:2:1."""
    k1 = f(t, x)
    k2 = f(t + h / 2, x + h / 2 * k1)
    k3 = f(t + h / 2, x + h / 2 * k2)
    k4 = f(t + h, x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint(f, x0, ts, step=rk4_step):
    """Integrate dx/dt = f(t, x) from x0 through the time grid ts.

    Returns the whole trajectory, stacked along a new leading axis, so that
    `odeint(...)[-1]` is the terminal state and `[:, i]` is the i-th coordinate.
    The grid may decrease, which integrates backwards in time.
    """
    xs = [x0]
    for t0, t1 in zip(ts[:-1], ts[1:]):
        xs.append(step(f, t0, xs[-1], t1 - t0))
    return torch.stack(xs)
