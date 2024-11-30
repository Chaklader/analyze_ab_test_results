# A/B Test Results Analysis

## Description
This project analyzes A/B test results from an e-commerce website to determine whether implementing a new page design would improve conversion rates. The analysis combines rigorous statistical methods to provide data-driven recommendations for the business decision.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Key Components of the Analysis](#key-components-of-the-analysis)
- [Key Findings](#key-findings)
- [Summary](#summary)


## Project Structure
- `ab_test_results.ipynb`: Main analysis notebook with statistical tests and modeling
- `ab_data.csv`: Dataset containing user interactions and conversion data
- `environment.yml`: Conda environment file for setting up the project dependencies

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Chaklader/analyze_ab_test_results.git
   ```
2. Navigate to the project directory:
   ```textmate
   cd analyze_ab_test_results
   ```
3. Create and activate the conda environment:
   ```textmate
   conda env create -f environment.yml
   conda activate statistics
   ```
4. Start the Jupyter notebook:
   ```textmate
   jupyter notebook ab_test_results.ipynb
   ```

### Key Components of the Analysis:
- **Data Analysis**: Analysis of 69,889 user interactions across control and treatment groups
- **Probability Assessment**: Calculation of conversion probabilities and comparative analysis
- **Statistical Testing**: Hypothesis testing with 500 simulations to evaluate statistical significance
- **Regression Modeling**: Logistic regression analysis to assess treatment effects and geographical impact
- **Results Interpretation**: Comprehensive interpretation of findings for business implementation

### Key Findings:
- **Conversion Rates**: The treatment group achieved a conversion rate of 15.53%, significantly higher than the 10.53% observed in the control group.
- **Statistical Significance**: The improvement in conversion rates is statistically significant with a p-value less than 0.05, indicating a strong likelihood that the observed effect is not due to random chance.
- **Geographical Consistency**: The effectiveness of the new page design is consistent across different countries, with no significant variation detected in conversion rates between the US, UK, and CA.
- **Model Insights**: The logistic regression model confirms the positive impact of the treatment, while also demonstrating minimal influence from geographical factors, suggesting that the new design's effectiveness is broadly applicable.

## Summary
This project successfully demonstrates the application of A/B testing in a real-world scenario, providing valuable insights into user behavior and the effectiveness of design changes. The analysis confirmed that the new page design significantly improves conversion rates, offering a strong case for its implementation. The findings are consistent across different geographical regions, indicating a broad applicability of the results. This project serves as a comprehensive guide for data-driven decision-making in e-commerce settings.
