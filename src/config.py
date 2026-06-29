from pathlib import Path
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DATA_PATHS = {
    "root": DATA_DIR,
    "business": DATA_DIR / "business_cards_MDQ.parquet",
    "consumer": DATA_DIR / "consumer_cards_MDQ.parquet",
    "merchants": DATA_DIR / "merchants_reference.parquet",
}

GRAPHS_DIR = PROJECT_ROOT / "graphs"

RANDOM_STATE = 67
MCC_GROUPS = {
    "advertising": ["7311"],
    "it_services": ["7372", "4816"],
    "subscription": ["5968"],
    "telecom": ["4814"],
}
COUNTRIES = ["Kazakhstan", "US", "Ireland"]

# Colors
BIZ = "#C1666B"  # muted rose  — Business
CON = "#F4A940"  # warm amber  — Consumer
BIZ_ALPHA = 0.80
CON_ALPHA = 0.65
MC_DARK = "#1A1F36"
MC_GRAY = "#6B7280"
MC_LIGHT = "#F9FAFB"
EDGES = "#E5E7EB"

PLT_PARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor": MC_LIGHT,
    "axes.edgecolor": EDGES,
    "axes.labelcolor": MC_DARK,
    "xtick.color": MC_GRAY,
    "ytick.color": MC_GRAY,
    "text.color": MC_DARK,
    "grid.color": EDGES,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 11,
}


def mc_title(ax, text):
    ax.set_title(text, fontsize=12, fontweight="600", color=MC_DARK, pad=10)


def mc_legend(ax):
    ax.legend(
        handles=[
            mpatches.Patch(color=BIZ, alpha=BIZ_ALPHA, label="Business"),
            mpatches.Patch(color=CON, alpha=CON_ALPHA, label="Consumer"),
        ],
        fontsize=10,
        framealpha=0.7,
    )
