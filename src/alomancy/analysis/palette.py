"""ALomancy brand colours, free of matplotlib so the terminal panel
(remote_submission/progress.py, ``alomancy status``) can use them without
importing it. analysis/colors.py re-exports them for plots."""

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
PENDING_COLOR: str = "#7E5FB8"  # muted mid-violet — not-yet-submitted jobs
GRID_ALPHA: float = 0.3
