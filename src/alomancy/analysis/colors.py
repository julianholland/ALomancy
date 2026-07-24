"""ALomancy colour scheme for all plots.

Derived from the brand primary #6C2FBE. Call ``setup_alomancy_style()`` at
the top of any plotting function to apply the scheme globally to the current
matplotlib session.
"""

from pathlib import Path

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt

_LOGO_PATH: Path = Path(__file__).parent / "alomancy_logo.png"

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------

ALOMANCY_PURPLE: str = "#6C2FBE"

# Categorical palette for multi-series plots (e.g. committee members).
# Starts with the brand violet, cycles through complementary hues that remain
# legible on a white/near-white background.
PALETTE: list[str] = [
    "#6C2FBE",  # brand violet   — primary
    "#0891B2",  # cyan           — complement
    "#059669",  # emerald
    "#D97706",  # amber
    "#E11D48",  # rose
    "#4338CA",  # indigo
    "#DB2777",  # hot pink
    "#0D9488",  # teal
    "#9333EA",  # bright violet  — lighter brand relative
    "#B45309",  # burnt amber
]

# ---------------------------------------------------------------------------
# Fixed reference colours
# ---------------------------------------------------------------------------

STAGE2_COLOR: str = "#3A106E"  # darkened brand violet — stage-2 / sentinel lines
DIAGONAL_COLOR: str = "#374151"  # dark charcoal         — parity y=x diagonal
GRID_COLOR: str = "#C1A8F0"  # light tint of brand   — grid lines
GRID_ALPHA: float = 0.3


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------


def add_logo_watermark(
    fig: matplotlib.figure.Figure,
    alpha: float = 0.30,
    size: float = 0.11,
) -> None:
    """Stamp the ALomancy logo as a semi-transparent watermark on *fig*.

    Must be called **after** ``tight_layout`` so the inset axes is not
    repositioned. The logo's white background is stripped to RGBA transparency
    so only the purple shape shows through — visible on any background colour.
    """
    if not _LOGO_PATH.exists():
        return

    import numpy as np
    from PIL import Image

    # Load and make white pixels transparent so the watermark
    # shows cleanly over any plot background.
    pil_img = Image.open(str(_LOGO_PATH)).convert("RGBA")
    data = np.array(pil_img, dtype=np.float32)
    # Pixels where all RGB channels are near-white → transparent
    white_mask = (data[..., 0] > 220) & (data[..., 1] > 220) & (data[..., 2] > 220)
    data[white_mask, 3] = 0.0
    logo_rgba = data / 255.0  # float32 in [0, 1] for matplotlib

    # Normalised height that keeps the logo square in pixel space.
    fig_w = fig.get_figwidth()
    fig_h = fig.get_figheight()
    height = size * (fig_w / fig_h)

    pad = 0.01
    ax_wm = fig.add_axes(
        [1.0 - size - pad, 1.0 - height - pad, size, height],
        label="_alomancy_watermark",
    )
    ax_wm.imshow(logo_rgba, alpha=alpha, aspect="equal")
    ax_wm.set_axis_off()
    ax_wm.patch.set_alpha(0.0)
    ax_wm.set_zorder(100)


def setup_alomancy_style() -> None:
    """Apply the ALomancy colour scheme to the current matplotlib session."""
    plt.rcParams.update(
        {
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "grid.alpha": GRID_ALPHA,
            "grid.color": GRID_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFE",
            "axes.edgecolor": "#4B5563",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.framealpha": 0.85,
            "legend.edgecolor": GRID_COLOR,
        }
    )
