import streamlit as st
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

st.set_page_config(page_title="A/B Testing Simulator", page_icon="🎯")

st.title("🎯 Improved A/B Testing Simulator")

st.markdown("### Enter Experiment Parameters")

# Inputs
col1, col2 = st.columns(2)

with col1:
    n_A = st.number_input("Sample Size A", min_value=0, value=0)
    p_A = st.slider("Conversion Rate A (%)", 0, 100, 0) / 100

with col2:
    n_B = st.number_input("Sample Size B", min_value=0, value=0)
    p_B = st.slider("Conversion Rate B (%)", 0, 100, 0) / 100

# Validation
if n_A == 0 or n_B == 0 or p_A == 0 or p_B == 0:
    st.warning("⚠️ Please enter non-zero values for all fields.")
else:
    np.random.seed(42)

    # Generate data
    group_A = np.random.binomial(1, p_A, n_A)
    group_B = np.random.binomial(1, p_B, n_B)

    conv_A = group_A.sum()
    conv_B = group_B.sum()

    rate_A = conv_A / n_A
    rate_B = conv_B / n_B

    # Z-test
    stat, pval = proportions_ztest([conv_A, conv_B], [n_A, n_B])

    # Lift
    lift = ((rate_B - rate_A) / rate_A) * 100 if rate_A != 0 else 0

    # Confidence Interval
    se = np.sqrt((rate_A*(1-rate_A)/n_A) + (rate_B*(1-rate_B)/n_B))
    ci_low = (rate_B - rate_A) - 1.96 * se
    ci_high = (rate_B - rate_A) + 1.96 * se

    st.markdown("## 📊 Results Summary")

    st.write(f"Conversion Rate A: {rate_A:.4f}")
    st.write(f"Conversion Rate B: {rate_B:.4f}")

    # ---------------- Z-STAT ----------------
    st.write(f"Z-statistic: {stat:.4f}")
    if abs(stat) > 1.96:
        st.info("Strong statistical difference between groups.")
    else:
        st.info("Weak or negligible statistical difference between groups.")

    # ---------------- P-VALUE ----------------
    st.write(f"P-value: {pval:.4f}")
    if pval < 0.01:
        st.success("Very strong evidence against null hypothesis.")
    elif pval < 0.05:
        st.success("Statistically significant difference detected.")
    elif pval < 0.1:
        st.warning("Weak evidence — results are inconclusive.")
    else:
        st.error("No statistical evidence of difference.")

    # ---------------- LIFT (IMPROVED) ----------------
    st.write(f"Lift: {lift:.2f}%")

    if lift > 20:
        st.success("B shows a strong improvement over A 🚀")
    elif lift > 5:
        st.info("B shows a moderate improvement over A 👍")
    elif lift > 0:
        st.info("B shows a slight improvement over A")
    elif lift < -20:
        st.success("A strongly outperforms B 🚀")
    elif lift < -5:
        st.info("A moderately outperforms B 👍")
    elif lift < 0:
        st.info("A slightly outperforms B")
    else:
        st.info("No meaningful performance difference")

    # ---------------- CI (IMPROVED) ----------------
    st.write(f"Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")

    if ci_low > 0:
        if ci_low > 0.02:
            st.success("Strong confidence that B is better ✅")
        else:
            st.info("Moderate confidence that B is better")
    elif ci_high < 0:
        if ci_high < -0.02:
            st.success("Strong confidence that A is better ✅")
        else:
            st.info("Moderate confidence that A is better")
    else:
        st.warning("Uncertain result — more data needed ⚠️")

    # ---------------- OVERALL ----------------
    st.markdown("## 🧠 Overall Interpretation")

    if pval < 0.05 and ci_low > 0:
        st.success("Clear evidence that B outperforms A. Consider deploying B.")
    elif pval < 0.05 and ci_high < 0:
        st.success("Clear evidence that A outperforms B. Keep A.")
    else:
        st.error("No clear winner. Recommend collecting more data.")

    # ---------------- GRAPH 1 ----------------
    st.markdown("## 📈 Conversion Comparison")

    fig, ax = plt.subplots()
    ax.bar(['A', 'B'], [rate_A, rate_B])
    ax.set_ylabel("Conversion Rate")
    ax.set_title("Conversion Rate Comparison Between Groups")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    st.info(
        "This graph compares conversion rates visually. "
        "Higher bar indicates better performance."
    )

    # ---------------- GRAPH 2 ----------------
    st.markdown("## 📉 P-value vs Sample Size")

    sample_sizes = np.linspace(50, 5000, 20).astype(int)
    p_values = []

    for size in sample_sizes:
        gA = np.random.binomial(1, p_A, size)
        gB = np.random.binomial(1, p_B, size)

        cA = gA.sum()
        cB = gB.sum()

        _, p = proportions_ztest([cA, cB], [size, size])
        p_values.append(p)

    fig2, ax2 = plt.subplots()
    ax2.plot(sample_sizes, p_values, marker='o')
    ax2.set_xlabel("Sample Size")
    ax2.set_ylabel("P-value")
    ax2.set_title("Effect of Sample Size on Statistical Significance")
    st.pyplot(fig2)

    st.info(
        "This graph shows how increasing sample size improves reliability. "
        "Larger samples generally reduce p-values and increase confidence."
    )