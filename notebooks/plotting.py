"""Shared plotting helpers for the course notebooks.

The notebooks are read as one sequence, so every figure in them should look the
same.  This module holds the house style and the figure shapes that recur, so a
notebook cell can stay close to the mathematics it illustrates:

    from plotting import *          # palette, rcParams and the helpers below

Everything here takes numpy arrays or torch tensors interchangeably.

Layout
    panels, label, curves                  -- the three workhorses
Recurring figures
    decision_boundary, contour_path, phase, complex_plane, heatmap, show_images
Animations
    animate, animate_points, animate_trajectory, animate_pendulum, animate_images
Schematics (drawing with no mathematical content in it)
    block, arrow, feedback_diagram, spring, spring_mass_damper, spring_mass_artist,
    maze_values, milestones
Small utilities
    smooth, cut_jumps, image_grid
"""

import numpy as np
import matplotlib

# The repo venv defaults to the tkagg backend, under which plt.show() spins
# forever at ~450% CPU inside nbclient instead of rendering.  Pin the inline
# backend here so that forgetting `%matplotlib inline` is no longer fatal.
try:
    matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:                                      # not in IPython (e.g. plain pytest)
    matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from IPython.display import HTML

__all__ = [
    "BLUE", "ORANGE", "AQUA", "YELLOW", "MAGENTA", "GREY", "INK",
    "RULE", "GRID", "GHOST", "NOTE", "CYCLE",
    "style", "panels", "label", "curves",
    "decision_boundary", "contour_path", "phase", "complex_plane", "heatmap",
    "image_grid", "show_images",
    "animate", "animate_points", "animate_trajectory", "animate_pendulum", "animate_images",
    "block", "arrow", "feedback_diagram",
    "spring", "spring_mass_damper", "spring_mass_artist", "maze_values", "milestones",
    "smooth", "cut_jumps",
]

# --------------------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------------------
# Five colorblind-safe hues carry the data; the greys carry everything that is not data.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CYCLE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

INK = "#0b0b0b"      # emphasis: the exact answer, the optimum, a decision boundary
GREY = "#9a9890"     # a secondary curve: the uncontrolled system, a reference run
RULE = "#c3c2b7"     # axis spines, zero lines, contour lines
GRID = "#e1e0d9"     # the grid, when it is on
GHOST = "#dcdbd3"    # a background cloud the eye should look past
NOTE = "#6b6a63"     # annotation text


def style(grid=None, figsize=None, dpi=None, **rc):
    """Apply the house style.  Called once on import; call again to adjust it."""
    plt.rcParams.update({
        "figure.figsize": figsize or (6, 4),
        "figure.dpi": dpi or 100,
        "figure.autolayout": True,              # so no cell ever calls tight_layout()
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "axes.grid": bool(grid),
        "grid.color": GRID,
        "axes.edgecolor": RULE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": NOTE,
        "ytick.color": NOTE,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": GRID,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "image.cmap": "RdBu_r",
        "animation.html": "jshtml",
        "figure.max_open_warning": 0,
        **rc,
    })


style()


def _np(a):
    """numpy view of an array, a torch tensor (even one that requires grad), or a list."""
    if hasattr(a, "detach"):
        a = a.detach()
    if hasattr(a, "cpu"):
        a = a.cpu()
    return np.asarray(a)


# --------------------------------------------------------------------------------------
# Layout: the three workhorses
# --------------------------------------------------------------------------------------
def panels(ncols=1, nrows=1, size=(3.7, 3.2), ratios=None, figsize=None, **kw):
    """A grid of panels, sized so that `panels(3)` is a comfortable row of three.

    `ratios` sets the width ratios of a row, or the height ratios of a column.
    """
    gridspec_kw = dict(kw.pop("gridspec_kw", {}) or {})
    if ratios is not None:
        gridspec_kw["width_ratios" if ncols > 1 else "height_ratios"] = ratios
    return plt.subplots(nrows, ncols, figsize=figsize or (size[0] * ncols, size[1] * nrows),
                        gridspec_kw=gridspec_kw, **kw)


def label(ax, xlabel=None, ylabel=None, title=None, legend=None, zero=None,
          xlim=None, ylim=None, equal=False, ticks=True, off=False, **kw):
    """Label an axes and finish it off.

    `legend=None` adds a legend exactly when something on the axes carries a label.
    `zero` draws rules through the origin: "h", "v" or "both".
    """
    ax.set(**{k: v for k, v in
              dict(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim).items()
              if v is not None}, **kw)
    if zero in ("h", "both"):
        ax.axhline(0, color=RULE, lw=1, zorder=0)
    if zero in ("v", "both"):
        ax.axvline(0, color=RULE, lw=1, zorder=0)
    if equal:
        ax.set_aspect("equal")
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if off:
        ax.axis("off")
    if legend is None:
        legend = bool(ax.get_legend_handles_labels()[1])
    if legend:
        ax.legend(**(legend if isinstance(legend, dict) else {}))
    return ax


def curves(ax, x, series, log=None, **kw):
    """Plot several labelled curves against a shared `x`, then label the axes.

    `series` is an array (one unlabelled curve) or a dict mapping each label to
    its y-values, optionally paired with a colour or a dict of line kwargs:

        curves(ax, t, {"no control": (free, GREY), "minimum energy": steered},
               xlabel="$t$", ylabel="$x_1$", title="Position")

    `log` selects the scale: "y", "x" or "xy".
    """
    plot = {None: ax.plot, "y": ax.semilogy, "x": ax.semilogx, "xy": ax.loglog}[log]
    x = None if x is None else _np(x)
    if not isinstance(series, dict):
        series = {"": series}
    for name, spec in series.items():
        opts = {}
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], (str, dict)):
            spec, opt = spec
            opts = {"color": opt} if isinstance(opt, str) else dict(opt)
        y = _np(spec)
        args = (y,) if x is None else (x, y)
        plot(*args, label=name or None, **opts)
    return label(ax, **kw)


# --------------------------------------------------------------------------------------
# Recurring figures
# --------------------------------------------------------------------------------------
def decision_boundary(ax, g1, g2, probs, X=None, y=None, colors=(BLUE, ORANGE), cbar=True,
                      cbar_label=r"$P(\mathrm{class}\ 1 \mid x)$", **kw):
    """Class probability over a grid, its 1/2 level set, and the data on top."""
    cf = ax.contourf(_np(g1), _np(g2), _np(probs), levels=20, cmap="RdBu_r",
                     alpha=0.55, vmin=0, vmax=1)
    ax.contour(_np(g1), _np(g2), _np(probs), levels=[0.5], colors=INK, linewidths=1.5)
    if X is not None:
        X = _np(X)
        c = np.asarray(colors)[_np(y)] if y is not None else colors[0]
        ax.scatter(X[:, 0], X[:, 1], c=c, s=14, edgecolors="white", lw=0.4)
    if cbar:
        ax.figure.colorbar(cf, ax=ax, label=cbar_label)
    kw.setdefault("xlabel", "$x_1$")
    kw.setdefault("ylabel", "$x_2$")
    return label(ax, **kw)


def contour_path(ax, G1, G2, Z, path=None, optimum=None, levels=None, color=ORANGE,
                 path_label=None, marker="o-", **kw):
    """A loss surface in contour, with an optimiser's path across it."""
    Z = _np(Z)
    if levels is None:
        levels = np.geomspace(max(Z.min(), 1e-6) + 0.05, Z.max(), 12)
    ax.contour(_np(G1), _np(G2), Z, levels=levels, colors=GRID, linewidths=0.8)
    if path is not None:
        p = _np(path)
        ax.plot(p[:, 0], p[:, 1], marker, color=color, ms=3.5, lw=1.2, label=path_label)
    if optimum is not None:
        o = _np(optimum)
        ax.plot(o[0], o[1], "*", color=INK, ms=12,
                label=kw.pop("optimum_label", None))
    return label(ax, **kw)


def phase(ax, trajectories, labels=None, colors=None, start=None, target=None, **kw):
    """Trajectories in the plane, with the initial state and the target marked."""
    trs = [trajectories] if _np(trajectories[0]).ndim == 0 or np.ndim(trajectories[0]) == 1 \
        else list(trajectories)
    trs = [_np(t) for t in trs]
    colors = colors or CYCLE
    for i, tr in enumerate(trs):
        ax.plot(tr[:, 0], tr[:, 1], color=colors[i % len(colors)],
                label=labels[i] if labels else None)
    if start is not None:
        ax.plot(*_np(start)[:2], "o", color=INK, ms=6)
    if target is not None:
        ax.plot(*_np(target)[:2], "*", color=ORANGE, ms=13)
    return label(ax, **kw)


def complex_plane(ax, **kw):
    """The Re/Im axes a pole plot needs, with rules through the origin."""
    kw.setdefault("xlabel", "Re")
    kw.setdefault("ylabel", "Im")
    return label(ax, zero="both", **kw)


def heatmap(ax, M, cmap="Blues", vmin=None, vmax=None, aspect=None, cbar=True,
            cbar_label=None, xticklabels=None, yticklabels=None, **kw):
    """A matrix as an image, with an optional colorbar and word tick labels."""
    im = ax.imshow(_np(M), cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    for axis, words in (("x", xticklabels), ("y", yticklabels)):
        if words is None:
            continue
        getattr(ax, f"set_{axis}ticks")(range(len(words)))
        getattr(ax, f"set_{axis}ticklabels")(words, fontsize=7,
                                             rotation=90 if axis == "x" else 0)
    if cbar:
        ax.figure.colorbar(im, ax=ax, label=cbar_label)
    label(ax, **kw)
    return im


def image_grid(imgs, nrow=8):
    """Tile a batch of (B, 1, H, W) images into one 2-D array."""
    a = _np(imgs)
    a = a.reshape(-1, *a.shape[-2:])
    h, w = a.shape[-2:]
    a = a[: len(a) // nrow * nrow].reshape(-1, nrow, h, w)
    return a.transpose(0, 2, 1, 3).reshape(-1, nrow * w)


def show_images(imgs, nrow=8, title=None, ax=None, figsize=None, cmap="gray_r",
                vmin=-1, vmax=1):
    """A batch of images as one tiled, unlabelled picture."""
    tiled = image_grid(imgs, nrow)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (nrow, tiled.shape[0] / tiled.shape[1] * nrow))
    ax.imshow(tiled, cmap=cmap, vmin=vmin, vmax=vmax)
    return label(ax, title=title, off=True)


# --------------------------------------------------------------------------------------
# Animations
# --------------------------------------------------------------------------------------
def animate(fig, update, frames, interval=80, **kw):
    """FuncAnimation, closing `fig` so the still frame is not shown alongside the player."""
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, **kw)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def _title(ax, spec, k):
    if spec is not None:
        ax.set_title(spec(k) if callable(spec) else spec)


def animate_points(traj, color=BLUE, s=10, background=None, title=None, decorate=None,
                   figsize=(4.4, 3.9), interval=110, pad=0.3, **kw):
    """A cloud of points moving: `traj` has shape (n_frames, n_points, 2)."""
    traj = _np(traj)
    fig, ax = plt.subplots(figsize=figsize)
    if background is not None:
        bg = _np(background)
        ax.plot(bg[:, 0], bg[:, 1], ".", ms=0.4, color=GHOST)
    lo, hi = traj.min(axis=(0, 1)) - pad, traj.max(axis=(0, 1)) + pad
    kw.setdefault("xlim", (lo[0], hi[0]))
    kw.setdefault("ylim", (lo[1], hi[1]))
    kw.setdefault("equal", True)
    label(ax, **kw)
    if decorate is not None:
        decorate(ax)
    sc = ax.scatter(traj[0][:, 0], traj[0][:, 1], s=s, c=color)

    def update(k):
        sc.set_offsets(traj[k])
        _title(ax, title, k)

    return animate(fig, update, len(traj), interval)


def animate_trajectory(xy, color=BLUE, title=None, decorate=None, figsize=(5.0, 4.2),
                       interval=70, trail=None, **kw):
    """One point tracing a path, with the whole path drawn faintly underneath."""
    xy = _np(xy)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xy[:, 0], xy[:, 1], color=GHOST, lw=1)
    if decorate is not None:
        decorate(ax)
    kw.setdefault("equal", True)
    kw.setdefault("ticks", False)
    label(ax, **kw)
    tail, = ax.plot([], [], color=color, lw=1.4)
    dot, = ax.plot([], [], "o", ms=5, color=ORANGE)

    def update(k):
        j = max(0, k - trail) if trail else 0
        tail.set_data(xy[j:k + 1, 0], xy[j:k + 1, 1])
        dot.set_data(xy[k:k + 1, 0], xy[k:k + 1, 1])
        _title(ax, title, k)

    return animate(fig, update, len(xy), interval)


def animate_pendulum(theta, dt=None, t=None, frames=None, trail=0, companion=None,
                     mark=None, mode=None, color=BLUE, figsize=None, interval=60,
                     ylabel=r"$\theta$"):
    """A pendulum swinging, `theta` measured from upright (so the bob is at sin/cos).

    `frames` are the indices of `theta` to animate.  With `companion` -- a second
    series at the same resolution as `theta` -- a time plot is drawn alongside and
    filled in as the animation runs; `mark` puts a vertical marker on it.
    """
    theta = _np(theta)
    t = np.arange(len(theta)) * (dt or 1.0) if t is None else _np(t)
    frames = np.arange(len(theta)) if frames is None else _np(frames)

    if companion is None:
        fig, axp = plt.subplots(figsize=figsize or (3.6, 3.6))
        axc = None
    else:
        fig, (axp, axc) = panels(2, ratios=[1, 1.25], figsize=figsize or (9.5, 3.8))

    axp.add_patch(Circle((0, 0), 1.0, fill=False, ec=GRID, lw=1, ls=":"))
    axp.add_patch(Circle((0, 0), 0.055, color=INK, zorder=4))
    axp.plot(0, 1, "*", color=ORANGE, ms=16, zorder=2)
    rod, = axp.plot([], [], color=color, lw=3, zorder=3)
    bob, = axp.plot([], [], "o", color=color, ms=12, zorder=4)
    path, = axp.plot([], [], color=color, lw=1, alpha=0.25)
    label(axp, xlim=(-1.45, 1.45), ylim=(-1.35, 1.6), equal=True, off=True)

    live = None
    if axc is not None:
        c = _np(companion)
        axc.plot(t, c, color="#d8d7cf", lw=1.5)
        if mark is not None:
            axc.axvline(mark, color=ORANGE, ls=":", lw=1.2)
        live, = axc.plot([], [], color=color, lw=1.8)
        label(axc, xlabel="$t$", ylabel=ylabel, xlim=(float(t[0]), float(t[-1])), zero="h")

    def update(k):
        i = frames[k]
        a = theta[i]
        px, py = np.sin(a), np.cos(a)
        rod.set_data([0, px], [0, py])
        bob.set_data([px], [py])
        if trail:
            past = theta[frames[max(0, k - trail):k + 1]]
            path.set_data(np.sin(past), np.cos(past))
        if live is not None:
            live.set_data(t[:i + 1], _np(companion)[:i + 1])
        suffix = f"   ({mode(i)})" if mode else ""
        axp.set_title(f"$t = {t[i]:.2f}$ s{suffix}")

    return animate(fig, update, len(frames), interval)


def animate_images(panels_, title=None, cmap="gray_r", vmin=-1, vmax=1,
                   figsize=(6, 2.2), interval=160):
    """A sequence of ready-made 2-D pictures, played as a film."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(_np(panels_[0]), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")

    def update(k):
        im.set_data(_np(panels_[k]))
        _title(ax, title, k)

    return animate(fig, update, len(panels_), interval)


# --------------------------------------------------------------------------------------
# Schematics -- drawing with no mathematical content in it
# --------------------------------------------------------------------------------------
def block(ax, x, y, w, h, text, ec=INK, fc="white", fontsize=10):
    """A rounded box with a label at its centre."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, lw=1.5, ec=ec, fc=fc,
                                boxstyle="round,pad=0.02,rounding_size=0.07"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p, q, color=INK, text=None, dy=0.18, fontsize=9):
    """An arrow from p to q, optionally labelled above its midpoint."""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center",
                fontsize=fontsize, color=color)


def feedback_diagram(ax):
    """Open loop above, closed loop below: what the feedback loop adds."""
    arrow(ax, (0.3, 1.6), (1.35, 1.6), ORANGE, "$u(t)$")
    block(ax, 2.1, 1.6, 1.5, 0.75, r"$\dot x = Ax + Bu$", fc="#eef3fb")
    arrow(ax, (2.85, 1.6), (4.0, 1.6), BLUE, "$x(t)$")
    ax.text(0.3, 2.25, "open loop — plan once, hope", fontsize=9.5, color=NOTE)

    ax.text(0.3, 0.6, "closed loop — measure and react", fontsize=9.5, color=NOTE)
    arrow(ax, (0.3, -0.4), (0.95, -0.4), GREY, "$r$")
    ax.add_patch(Circle((1.15, -0.4), 0.17, fc="white", ec=INK, lw=1.4))
    ax.text(1.15, -0.4, "$-$", ha="center", va="center", fontsize=9)
    arrow(ax, (1.32, -0.4), (1.75, -0.4), ORANGE)
    block(ax, 2.35, -0.4, 1.0, 0.7, r"$-K$", fc="#fdf0e8", ec=ORANGE)
    arrow(ax, (2.85, -0.4), (3.35, -0.4), ORANGE, "$u$")
    block(ax, 4.15, -0.4, 1.5, 0.75, r"$\dot x = Ax + Bu$", fc="#eef3fb")
    arrow(ax, (4.9, -0.4), (6.0, -0.4), BLUE, "$x$")
    ax.plot([5.5, 5.5], [-0.4, -1.5], color=BLUE, lw=1.4)
    ax.plot([5.5, 1.15], [-1.5, -1.5], color=BLUE, lw=1.4)
    arrow(ax, (1.15, -1.5), (1.15, -0.6), BLUE)
    ax.text(3.3, -1.78, "the loop that makes it robust", fontsize=9, color=BLUE, ha="center")
    return label(ax, xlim=(0, 6.3), ylim=(-2.1, 2.6), equal=True, off=True)


def spring(x0, x1, y, coils=9, amp=0.12):
    """Polyline of a zig-zag spring drawn from x0 to x1 at height y."""
    xs = np.linspace(x0, x1, 2 * coils + 3)
    ys = np.full_like(xs, y)
    ys[2:-1:2], ys[3:-1:2] = y + amp, y - amp
    return xs, ys


def spring_mass_damper(ax, pos=0.0):
    """Plan-style schematic of the spring-mass-damper, with the mass at `pos`."""
    ax.add_patch(plt.Rectangle((-2.2, -0.75), 0.14, 1.5, color=GREY))            # wall
    for yy in np.linspace(-0.75, 0.55, 8):                                       # hatching
        ax.plot([-2.4, -2.06], [yy, yy + 0.2], color=GREY, lw=1)
    ax.plot(*spring(-2.06, pos - 0.45, 0.34), color=BLUE, lw=1.6)                # spring
    ax.plot([-2.06, -1.45], [-0.34, -0.34], color=AQUA, lw=1.6)                  # damper
    ax.add_patch(plt.Rectangle((-1.45, -0.56), 0.5, 0.44, fc="white", ec=AQUA, lw=1.6))
    ax.plot([-1.2, pos - 0.45], [-0.34, -0.34], color=AQUA, lw=1.6)
    ax.add_patch(FancyBboxPatch((pos - 0.45, -0.55), 0.9, 1.1, lw=1.4,           # mass
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="#eef3fb", ec=INK))
    ax.text(pos, 0, "$m$", ha="center", va="center", fontsize=12)
    arrow(ax, (pos + 0.55, 0), (pos + 1.3, 0), ORANGE)
    ax.text(pos + 0.92, 0.22, "$u(t)$", color=ORANGE, ha="center", fontsize=11)
    ax.text(-1.5, 0.62, "$k$", color=BLUE, fontsize=11)
    ax.text(-1.35, -1.0, "$c$", color=AQUA, fontsize=11)
    ax.plot([-2.06, 2.8], [-1.35, -1.35], color=RULE, lw=1)                      # position axis
    ax.plot(0, -1.35, "|", color=RULE, ms=9)
    ax.annotate("", xy=(pos, -1.35), xytext=(0, -1.35),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(pos / 2, -1.68, "$x_1$", ha="center", fontsize=11)
    return label(ax, xlim=(-2.6, 3.1), ylim=(-2.0, 1.2), equal=True, off=True)


def spring_mass_artist(ax, y, color, text):
    """A minimal, updatable spring-mass drawing on row `y`.  Returns update(position)."""
    ax.plot([-2.1, -2.1], [y - 0.55, y + 0.55], color=GREY, lw=4, solid_capstyle="butt")
    line, = ax.plot([], [], color=color, lw=1.5)
    mass = FancyBboxPatch((0, y - 0.3), 0.7, 0.6, boxstyle="round,pad=0.01,rounding_size=0.05",
                          fc="white", ec=color, lw=1.8)
    ax.add_patch(mass)
    ax.text(-2.3, y, text, ha="right", va="center", fontsize=9, color=color)

    def update(p):
        line.set_data(*spring(-2.1, p - 0.35, y, coils=7, amp=0.1))
        mass.set_x(p - 0.35)

    return update


def maze_values(ax, maze, cells, terminal, moves, V, policy, vmin=0.0, vmax=1.0, **kw):
    """A grid world: the value function in colour, the greedy policy as arrows."""
    maze = np.asarray(maze)
    cells, V, policy = np.asarray(cells), _np(V), _np(policy)
    terminal = np.asarray(terminal)
    field = np.full(maze.shape, np.nan)
    field[tuple(cells.T)] = np.where(terminal, np.nan, V)
    ax.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.imshow(np.where(maze == "#", 0.0, np.nan), cmap="gray", vmin=0, vmax=1)
    for s, (i, j) in enumerate(cells):
        if terminal[s]:
            ax.text(j, i, maze[i, j], ha="center", va="center", fontsize=9, weight="bold")
        else:
            d = np.asarray(moves)[policy[s]]
            ax.arrow(j - .2 * d[1], i - .2 * d[0], .3 * d[1], .3 * d[0],
                     head_width=.2, color=INK, lw=.7)
    return label(ax, ticks=False, **kw)


def milestones(ax, items, color=BLUE, heights=(0.45, -0.45, 1.0, -1.0), fontsize=6.5):
    """A dated timeline: `items` is a list of (year, caption)."""
    years = [y for y, _ in items]
    ax.hlines(0, min(years) - 8, max(years) + 11, color=GREY, lw=1)
    for k, (year, name) in enumerate(items):
        h = heights[k % len(heights)]
        ax.vlines(year, 0, h, color=color, lw=0.8)
        ax.plot(year, 0, "o", ms=4, color=color)
        ax.text(year, 1.18 * h, f"{year}\n{name}", ha="center", va="center", fontsize=fontsize)
    return label(ax, ylim=(-1.8, 1.8), off=True)


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
def smooth(y, w=20):
    """Moving average over a window of `w` samples."""
    return np.convolve(_np(y), np.ones(w) / w, "valid")


def cut_jumps(a, threshold=np.pi):
    """Insert NaN where a wrapped angle jumps, so the plot does not draw vertical lines."""
    a = _np(a).astype(float)
    return np.where(np.abs(np.diff(a, prepend=a[0])) > threshold, np.nan, a)
