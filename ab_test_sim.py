import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# ---------- Generate Data ----------
def generate_data(n_A, n_B, p_A, p_B):
    group_A = np.random.binomial(1, p_A, n_A)
    group_B = np.random.binomial(1, p_B, n_B)

    df = pd.DataFrame({
        'group': ['A'] * n_A + ['B'] * n_B,
        'converted': np.concatenate([group_A, group_B])
    })

    return df

# ---------- Run A/B Test ----------
def run_ab_test(df):
    conv_counts = df.groupby('group')['converted'].sum().values
    total_counts = df.groupby('group')['converted'].count().values

    stat, pval = proportions_ztest(conv_counts, total_counts)

    rate_A = conv_counts[0] / total_counts[0]
    rate_B = conv_counts[1] / total_counts[1]

    return stat, pval, rate_A, rate_B, total_counts

# ---------- Confidence Interval ----------
def confidence_interval(rate_A, rate_B, n_A, n_B):
    diff = rate_B - rate_A
    se = np.sqrt((rate_A*(1-rate_A)/n_A) + (rate_B*(1-rate_B)/n_B))

    lower = diff - 1.96 * se
    upper = diff + 1.96 * se

    return lower, upper

# ---------- Effect Size ----------
def effect_size(rate_A, rate_B):
    if rate_A == 0:
        return 0
    return ((rate_B - rate_A) / rate_A) * 100

# ---------- Interpretation ----------
def interpret(pval, ci_low, ci_high):
    if pval < 0.05:
        if ci_low > 0:
            return "Variant B is significantly better ✅"
        elif ci_high < 0:
            return "Variant A is significantly better ✅"
        else:
            return "Statistically significant but uncertain direction ⚠️"
    else:
        return "No statistically significant difference ❌"

# ---------- Visualization ----------
def visualize(df, pval):
    rates = df.groupby('group')['converted'].mean()

    plt.bar(['A', 'B'], rates)
    plt.ylabel("Conversion Rate")
    plt.title(f"Conversion Rate Comparison (p-value: {pval:.4f})")
    plt.ylim(0, 1)
    plt.show()

# ---------- Sample Size vs P-value ----------
def pvalue_vs_sample(p_A, p_B):
    sample_sizes = np.linspace(50, 5000, 20).astype(int)
    p_values = []

    for size in sample_sizes:
        gA = np.random.binomial(1, p_A, size)
        gB = np.random.binomial(1, p_B, size)

        cA = gA.sum()
        cB = gB.sum()

        _, p = proportions_ztest([cA, cB], [size, size])
        p_values.append(p)

    plt.plot(sample_sizes, p_values, marker='o')
    plt.xlabel("Sample Size")
    plt.ylabel("P-value")
    plt.title("Effect of Sample Size on Statistical Significance")
    plt.show()

# ---------- MAIN ----------
if __name__ == "__main__":
    print("A/B Testing Simulator\n")

    n_A = int(input("Enter sample size for Group A: "))
    n_B = int(input("Enter sample size for Group B: "))
    p_A = float(input("Enter conversion rate for Group A (0-1): "))
    p_B = float(input("Enter conversion rate for Group B (0-1): "))

    if n_A == 0 or n_B == 0 or p_A == 0 or p_B == 0:
        print("Please enter non-zero values.")
    else:
        df = generate_data(n_A, n_B, p_A, p_B)

        stat, pval, rate_A, rate_B, counts = run_ab_test(df)

        lift = effect_size(rate_A, rate_B)
        ci_low, ci_high = confidence_interval(rate_A, rate_B, n_A, n_B)

        print("\n--- RESULTS ---")
        print(f"Conversion A: {rate_A:.4f}")
        print(f"Conversion B: {rate_B:.4f}")
        print(f"Z-statistic: {stat:.4f}")
        print(f"P-value: {pval:.4f}")
        print(f"Lift: {lift:.2f}%")
        print(f"Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")

        print("\n--- INTERPRETATION ---")
        print(interpret(pval, ci_low, ci_high))

        visualize(df, pval)
        pvalue_vs_sample(p_A, p_B)