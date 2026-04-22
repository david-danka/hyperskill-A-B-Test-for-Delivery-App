# A/B Test Analysis for Delivery App

## Overview

This project simulates a real-world A/B testing workflow for a food delivery application.
The goal is to evaluate whether a new interface increases the average order value (AOV).

## Project Structure

The analysis follows a complete experimental pipeline:

1. **Sample Validation (A/A Test)**

   * Levene’s test for variance equality
   * Independent t-test for mean comparison

2. **Sample Size Estimation**

   * Power analysis to determine required sample size

3. **Exploratory Data Analysis (EDA)**

   * Session trends over time
   * Distribution analysis
   * Outlier detection and removal (99th percentile)

4. **Non-parametric Hypothesis Testing**

   * Mann-Whitney U test (no normality assumption)

5. **Parametric Validation**

   * Log transformation of skewed data
   * Levene’s test + t-test on transformed data

## Key Insights

* Outliers significantly impact AOV and must be handled carefully
* The distribution of order values is skewed, requiring non-parametric methods
* Log transformation allows parametric testing for confirmation

## Technologies Used

* Python
* pandas, NumPy
* SciPy
* statsmodels
* matplotlib

## How to Run

Run the script:

```bash
python main.py
```

All five stages will execute sequentially and display results.

## Notes

This project was completed as part of a data analysis training program, with additional refactoring to improve readability and structure.
