"""
A/B Test Analysis for Delivery App

This script implements a full A/B testing workflow:
- Sample validation (A/A test)
- Sample size estimation (power analysis)
- Exploratory data analysis (EDA)
- Non-parametric hypothesis testing
- Parametric validation via log transformation

Datasets:
- aa_test.csv
- ab_test.csv
Both datasets are expected in the working directory.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import tt_ind_solve_power


CONTROL = "Control"
EXPERIMENTAL = "Experimental"
ALPHA = 0.05


def wait_for_user() -> None:
    input("\nPress any key to continue\n")

def print_stats_test_result(
        test_name: str,
        statistic_label: str,
        statistic: float,
        p_value: float,
        alpha: float,
        hypothesis_statement: str,
) -> None:
    """Print hypothesis test results."""

    is_significant = p_value <= alpha

    print(f"{test_name}")
    print(
        f"{statistic_label} = {statistic:.3f}, "
        f"p-value {'<=' if is_significant else '>'} {alpha}"
    )
    print(f"Reject null hypothesis: {'yes' if is_significant else 'no'}")
    print(f"{hypothesis_statement}: {'no' if is_significant else 'yes'}")

def verify_sample_size(alpha: float) -> None:
    """Validate sample size by comparing means of two A/A test samples."""

    # Read samples
    df = pd.read_csv("aa_test.csv")
    samples = df["Sample 1"], df["Sample 2"]

    # Calculate statistics
    levene_result = stats.levene(*samples)
    equal_variances = levene_result.pvalue > alpha

    t_result = stats.ttest_ind(*samples, equal_var=equal_variances)

    # Print results
    print_stats_test_result(
        test_name="Levene's test",
        statistic_label="W",
        statistic=levene_result.statistic,
        p_value=levene_result.pvalue,
        alpha=alpha,
        hypothesis_statement="Variances are equal"
    )
    print()
    print_stats_test_result(
        test_name="T-test",
        statistic_label="t",
        statistic=t_result.statistic,
        p_value=t_result.pvalue,
        alpha=alpha,
        hypothesis_statement="Means are equal"
    )
    print()

    if t_result.pvalue <= alpha:
        print("Means are not equal -> larger sample size needed.")
    else:
        print("Means are equal -> sample size is sufficient.")

    wait_for_user()

def estimate_sample_size(
    effect_size: float,
    power: float,
    alpha: float
) -> None:
    """Estimate required sample size using power analysis."""

    # Read and transform data
    df = pd.read_csv("ab_test.csv")
    control = df[df.group == CONTROL]
    experimental = df[df.group == EXPERIMENTAL]

    # Run power analysis
    sample_size = tt_ind_solve_power(
        effect_size=effect_size,
        nobs1=None,
        alpha=alpha,
        power=power,
        ratio=1.0
    )
    sample_size = math.ceil(sample_size / 100) * 100

    # Print results
    print(f"Sample size: {sample_size}")
    print()
    print(f"Control group: {control.shape[0]}")
    print(f"Experimental group: {experimental.shape[0]}")

    wait_for_user()

def plot_sessions_by_day(df: pd.DataFrame) -> None:
    # Extract day from date
    df = df.assign(day=pd.to_datetime(df["date"]).dt.day)
    # Session count by day and group
    grouped = df.groupby(["day", "group"]).size().unstack()
    # Plot grouped bar chart
    grouped.plot(kind="bar")
    plt.title("Number of Sessions per Day by Group")
    plt.xlabel("June")
    plt.ylabel("Number of Sessions")
    plt.show()

def plot_grouped_histogram(
    df: pd.DataFrame,
    column: str,
    group_by: str,
    title: str
) -> None:
    axes = df.hist(column=column, by=group_by, sharex=True, sharey=True)

    # axes is an array (even if 1D), flatten for safety
    for ax in axes.flatten():
        ax.set_xlabel(column.replace("_", " "). capitalize())
        ax.set_ylabel("Frequency")
        ax.yaxis.set_tick_params(labelleft=True)  # restore y tick values
        ax.tick_params(left=True, labelleft=True)  # restore y axis ticks

    plt.suptitle(title)
    plt.show()

def remove_outliers(
    df: pd.DataFrame,
    columns: list[str],
    percentile: float,
) -> pd.DataFrame:
    thresholds = df[columns].quantile(percentile / 100)
    return df[(df[columns] < thresholds).all(axis=1)].copy()

def run_eda() -> None:
    """Perform exploratory data analysis and summarize key metrics."""

    # Read data
    df = pd.read_csv("ab_test.csv")

    # Plots
    plot_sessions_by_day(df)
    plot_grouped_histogram(
        df=df,
        column="order_value",
        group_by="group",
        title="Distribution of Order Value by Group"
    )
    plot_grouped_histogram(
        df=df,
        column="session_duration",
        group_by="group",
        title="Distribution of Session Duration by Group"
    )

    # Key metrics
    filtered = remove_outliers(df, ["order_value", "session_duration"], 99)
    print(f"Mean: {filtered["order_value"].mean():.2f}")
    print(f"Standard deviation: {filtered["order_value"].std(ddof=0):.2f}")
    print(f"Max: {filtered["order_value"].max()}")

    wait_for_user()

def verify_distributions(alpha: float) -> None:
    """
    Perform a Mann-Whitney U test to compare order value distributions
    between control and experimental groups.
    """
    df = pd.read_csv("ab_test.csv")

    filtered = remove_outliers(df, ["order_value", "session_duration"], 99)

    groups = filtered.groupby("group")["order_value"]
    samples = groups.get_group(CONTROL), groups.get_group(EXPERIMENTAL)

    # Non-parametric test used due to non-normal distribution of order values
    results = stats.mannwhitneyu(*samples)

    print_stats_test_result(
        test_name="Mann-Whitney U test",
        statistic_label="U1",
        statistic=results.statistic,
        p_value=results.pvalue,
        alpha=alpha,
        hypothesis_statement="Distributions are same")

    wait_for_user()

def verify_parametric_test(alpha: float) -> None:
    """
    Perform parametric hypothesis testing on log-transformed order values.
    """
    df = pd.read_csv("ab_test.csv")
    filtered = remove_outliers(df, ["order_value", "session_duration"], 99)

    # Log transform to reduce skewness and approximate normality
    filtered = filtered.assign(
        log_order_value=np.log(filtered["order_value"])
    )

    filtered["log_order_value"].hist(bins=30)
    plt.title("Distribution of Log-Transformed Order Value")
    plt.xlabel("ln(Order Value)")
    plt.ylabel("Frequency")
    plt.show()

    groups = filtered.groupby("group")["log_order_value"]
    samples = groups.get_group(CONTROL), groups.get_group(EXPERIMENTAL)
    levene_result = stats.levene(*samples)
    equal_variances = levene_result.pvalue > alpha
    t_result = stats.ttest_ind(*samples, equal_var=equal_variances)

    print_stats_test_result(
        test_name="Levene's test",
        statistic_label="W",
        statistic=levene_result.statistic,
        p_value=levene_result.pvalue,
        alpha=alpha,
        hypothesis_statement="Variances are equal"
    )
    print()
    print_stats_test_result(
        test_name="T-test",
        statistic_label="t",
        statistic=t_result.statistic,
        p_value=t_result.pvalue,
        alpha=alpha,
        hypothesis_statement="Means are equal"
    )
    print()
    print("Interpretation:")
    if t_result.pvalue > alpha:
        print("No statistically significant difference in AOV.")
        return

    control_aov = filtered[filtered["group"] == CONTROL]["order_value"].mean()
    experimental_aov = filtered[filtered["group"] == EXPERIMENTAL]["order_value"].mean()
    print(f"Control AOV: {control_aov:.2f}")
    print(f"Experimental AOV: {experimental_aov:.2f}")

    if control_aov > experimental_aov:
        print("Statistically significant decrease in AOV observed in experimental group.")
    else:
        print("Statistically significant increase in AOV observed in experimental group.")


def main():
    print("=== Stage 1: Sample Size Verification ===")
    verify_sample_size(ALPHA)

    print("\n=== Stage 2: Sample Size Estimation ===")
    estimate_sample_size(effect_size=0.2, power=0.8, alpha=ALPHA)

    print("\n=== Stage 3: Exploratory Data Analysis ===")
    run_eda()

    print("\n=== Stage 4: Non-parametric Test ===")
    verify_distributions(ALPHA)

    print("\n=== Stage 5: Parametric Test ===")
    verify_parametric_test(ALPHA)

if __name__ == "__main__":
    main()