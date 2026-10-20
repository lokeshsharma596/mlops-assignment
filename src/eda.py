"""Task 1 - exploratory data analysis. Saves figures to reports/figures/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import FIGURES_DIR, NUMERIC_FEATURES, TARGET  # noqa: E402
from src.data import load_clean  # noqa: E402

sns.set_theme(style="whitegrid")


def run() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()

    print(df.describe().T)
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])

    # 1. class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET, data=df, ax=ax, palette=["#4C9F70", "#D1495B"], hue=TARGET,
                  legend=False)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_title("Class balance")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)

    # 2. histograms of numeric features, split by target
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, element="step")
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "histograms.png", dpi=150)
    plt.close(fig)

    # 3. correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. boxplots (outliers) per numeric feature
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax, hue=TARGET, legend=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "boxplots.png", dpi=150)
    plt.close(fig)

    # 5. categorical features vs target
    cats = ["sex", "cp", "exang", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cats):
        sns.countplot(data=df, x=col, hue=TARGET, ax=ax)
        ax.set_title(f"{col} vs target")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "categorical_vs_target.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    run()
