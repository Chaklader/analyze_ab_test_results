WEBSITE: https://www.statisticshowto.com



# CHAPTER 3: PROBABILITY 

This repository contains notes and examples for various statistical concepts used in data analysis.


# Understanding Sensitivity and Specificity in Medical Testing

In medical testing and diagnostic procedures, two crucial metrics help us evaluate the performance of a test: sensitivity and specificity. These concepts are fundamental in understanding how well a test can identify both positive and negative cases.

## Sensitivity (True Positive Rate)

Sensitivity measures the proportion of actual positive cases that are correctly identified. In other words, it answers the question: "If a person has the disease, how likely is the test to detect it?"

**Formula:**
```
Sensitivity = True Positives / (True Positives + False Negatives)
```

### Example of Sensitivity

Let's consider a COVID-19 testing scenario with 100 people who actually have COVID-19:
- 90 people test positive (True Positives)
- 10 people test negative (False Negatives)

Sensitivity = 90 / (90 + 10) = 90%

This means the test correctly identifies 90% of people who actually have COVID-19.

## Specificity (True Negative Rate)

Specificity measures the proportion of actual negative cases that are correctly identified. It answers the question: "If a person doesn't have the disease, how likely is the test to give a negative result?"

**Formula:**
```
Specificity = True Negatives / (True Negatives + False Positives)
```

### Example of Specificity

Consider testing 100 people who don't have COVID-19:
- 95 people test negative (True Negatives)
- 5 people test positive (False Positives)

Specificity = 95 / (95 + 5) = 95%

This means the test correctly identifies 95% of people who don't have COVID-19.

## Real-World Example: Mammogram Screening

Let's look at a real-world example of mammogram screening for breast cancer:

Suppose we have 1000 women:
- 100 have breast cancer
- 900 don't have breast cancer

The mammogram test results show:
- Of the 100 women with cancer:
  - 90 test positive (True Positives)
  - 10 test negative (False Negatives)
- Of the 900 women without cancer:
  - 855 test negative (True Negatives)
  - 45 test positive (False Positives)

**Calculations:**
- Sensitivity = 90/100 = 90%
- Specificity = 855/900 = 95%

## Trade-off Between Sensitivity and Specificity

There's often a trade-off between sensitivity and specificity:
- High sensitivity: Fewer false negatives but more false positives
- High specificity: Fewer false positives but more false negatives

The choice between prioritizing sensitivity or specificity depends on the context:
- For screening tests (like cancer screening), high sensitivity is crucial to not miss cases
- For confirmatory tests, high specificity is important to avoid false alarms

## Summary Table

| Metric | What it Measures | Formula | When it's Important |
|--------|------------------|---------|-------------------|
| Sensitivity | Ability to detect positive cases | TP/(TP+FN) | Screening tests, where missing a case is costly |
| Specificity | Ability to detect negative cases | TN/(TN+FP) | Confirmatory tests, where false alarms are costly |

## Practical Applications

1. **Disease Screening Programs:**
   - High sensitivity tests are used for initial screening
   - High specificity tests are used for confirmation

2. **Quality Control:**
   - Manufacturing defect detection
   - Food safety testing

3. **Security Systems:**
   - Airport security screening
   - Fraud detection systems

Understanding these metrics helps in:
- Choosing appropriate tests for specific situations
- Interpreting test results correctly
- Making informed decisions about testing procedures4


Here's the markdown table based on the cases from the video:

| Case | Probability | Key Term |
|------|------------|----------|
| P(Cancer) | 0.01 | |
| P(¬Cancer) | 0.99 | |
| P(Positive\|Cancer) | 0.9 | Sensitivity |
| P(Negative\|¬Cancer) | 0.9 | Specificity |

This table shows:
1. Prior probability of having cancer (1%)
2. Prior probability of not having cancer (99%)
3. Sensitivity - probability of testing positive given you have cancer (90%)
4. Specificity - probability of testing negative given you don't have cancer (90%)

### Cancer Probabilities

| Case | Probability | Key Term |
|------|------------|----------|
| P(C) | 0.01 | Prior Probability |
| P(Pos\|C) | 0.9 | Sensitivity |
| P(Neg\|¬C) | 0.9 | Specificity |

## Visual Representation


<br>

![Cancer Statistics Visualization](cancer.png)

<br>

Yellow Circle is for the case when the cancer test is positive irrespectively of the cancer status. Menaing it includes the cases when the test is positive but person doesn't have the cancer.


Yellow Circle = P(C&Pos) + P(¬C&Pos) =P(Cancer|Positive) + P(¬Cancer| Positive)


Without calculating it mathematically, let's estimate the answer here.

First, we know that only 1% of people have cancer regardless of the test being positive or negative. Imagine someone has already tested positive, the probability of that person having cancer will be higher than 1%. But how much higher?

Essentially, the quiz asks you to solve the probability of P(C|Pos).

From conditional probability theory, we know that

P(C|Pos) = P(C&Pos) / P(Pos) = P(C&Pos) / [P(C&Pos) + P(
¬
¬C&Pos)],

which is the ratio of people who have cancer and also test positive to people who test positive whether they have cancer or not. This ratio will be quite small because the majority (99%) of people don't have cancer, which means the number of people who test positive and have cancer (P(C&Pos)) is relatively small compared to the number of people who test positive but don't have cancer (P(
¬
¬C&Pos)). So the ratio could not be as high as 90%.

Based on the estimation and the 3 choices above, we can say that P(C|Pos) is higher than 1% and below 90%. So only 8% makes sense. 


Bayes Theorem

P(C|Pos) = P(Pos|C) * P(C) / P(Pos)

Prior Probability * Test Evidence = Posterior Probability



Prior Probability: This is your initial belief or probability of a hypothesis before seeing new evidence. It represents what you know about an event before collecting additional data.
Test Evidence: This is new information or data that you collect. In Bayesian terms, this is often represented as the likelihood of observing this evidence given your hypothesis.
Posterior Probability: This is your updated belief after combining your prior knowledge with the new evidence. It represents your revised probability estimate after considering both your initial beliefs and the new data.

Example: If you're testing for a medical condition:

   Prior: The general prevalence of the condition in the population (say 1%)
   Test Evidence: The accuracy of your diagnostic test
   Posterior: Your updated estimate of whether the person has the condition after seeing the test results


Prior:
P(C) = 0.01 = 1%
P(Pos|C) = 0.9 = 90%
P(Neg|¬C) = 0.9
P(¬C) = 0.99
P(Pos|¬C) = 0.1

Posterior:
P(C|Pos) = P(C) · P(Pos|C) / [P(C) · P(Pos|C) + P(¬C) · P(Pos|¬C)]
P(¬C|Pos) = P(¬C) · P(Pos|¬C) / [P(C) · P(Pos|C) + P(¬C) · P(Pos|¬C)]

Note on Normalization:
When calculating posterior probabilities for opposite scenarios (like having cancer vs not having cancer given a positive test), we need to normalize the values because:
1. The raw calculations [P(C) · P(Pos|C)] and [P(¬C) · P(Pos|¬C)] give us relative weights, not true probabilities
2. Since these scenarios are mutually exclusive and exhaustive (you either have cancer or you don't), their probabilities must sum to 1
3. Normalization is done by dividing each raw value by their sum:
   - Raw values: 0.009 and 0.099 (sum = 0.108)
   - Normalized: 0.009/0.108 = 8.33% and 0.099/0.108 = 91.67%
   - Now they sum to 100% and represent true probabilities

This shows:
1. Prior probability of cancer (1%)
2. Sensitivity - P(Pos|C) = 90%
3. Specificity - P(Neg|¬C) = 90%
4. Probability of not having cancer (99%)
5. False positive rate - P(Pos|¬C) = 10%

The posterior probabilities are calculated using Bayes' theorem components.



Let's say we get a positive test and we want to look at both cancer and no cancer hypotheses.

1. Cancer Hypothesis = Prior probability x sensitivity
2. No Cancer Hypothesis = Prior probability x (1-sensitivity)
3. Normalizer = Cancer Hypothesis + No Cancer Hypothesis


Sensitivity = P(Pos|C) = Probability of a positive test given that you have cancer
Specificity = P(Neg|¬C) = Probability of a negative test given that you don't have cancer
(1-sensitivity) = P(Neg|C) = Probability of a negative test given that you have cancer (False Negative Rate)
(1-specificity) = P(Pos|¬C) = Probability of a positive test given that you don't have cancer (False Positive Rate)


If these numbers (in steps 1 and 2 above) are added together, the result is the normalizer and it will normally not be 1.

The next step in the process is to normalize the hypotheses using the normalizer. Because the normalizer represents the probability of a positive test, it is independent of cancer diagnosis and therefore can be used to normalize both cases.

P(C|Pos) = Cancer Hypothesis / Normalizer
= [P(C) × P(Pos|C)] / [P(C) × P(Pos|C) + P(¬C) × P(Pos|¬C)]

P(¬C|Pos) = No Cancer Hypothesis / Normalizer
= [P(¬C) × P(Pos|¬C)] / [P(C) × P(Pos|C) + P(¬C) × P(Pos|¬C)]

P(C|Pos) + P(¬C|Pos) = 1

This is the algorithm for Bayes Rule.



For a negative test result (Test = Neg):

P(C|Neg) = [P(C) · P(Neg|C)] / P(Neg)

P(¬C|Neg) = [P(¬C) · P(Neg|¬C)] / P(Neg)

where:
P(Neg) = P(C) · P(Neg|C) + P(¬C) · P(Neg|¬C)


# CHAPTER 4: EXPERIMENTATION  

Finding the most likely number of heads (k) when flipping a fair coin 20 times. 

Given information:
- N = 20 (total number of coin flips)
- p = 0.5 (probability of heads for a fair coin)
- We're using the binomial probability formula: P(k) = [N!/(N-k)!k!] * p^k * (1-p)^(n-k)

The question asks which value of k (1, 3, 10, or 20) maximizes this probability.

For a fair coin with p = 0.5, the most likely outcome is getting heads half the time, because:
1. The coin is unbiased (p = 0.5)
2. Each flip is independent
3. With 20 flips, the expected value is np = 20 * 0.5 = 10 heads

Therefore, k = 10 would maximize the probability. This is because:
- Getting 1 or 3 heads is too few for 20 flips of a fair coin
- Getting 20 heads (all heads) is extremely unlikely
- 10 heads represents the most balanced and likely outcome

The answer would be 10 from the given options.



Binomial Probability formula P(k) = [N!/(N-k)!k!] * p^k * (1-p)^(n-k) step by step:

1. The formula has three main parts:
   * [N!/(N-k)!k!] - This is the combination part (how many ways to choose k items from N items)
   * p^k - Probability of success raised to number of successes
   * (1-p)^(n-k) - Probability of failure raised to number of failures

2. Using the coin flip example:
   * N = total number of flips (20)
   * k = number of heads we want
   * p = probability of heads (0.5)
   * (1-p) = probability of tails (0.5)

3. Let's understand each part:
   * [N!/(N-k)!k!] calculates how many different ways you can get k heads in N flips
   * p^k represents the probability of getting all those k heads
   * (1-p)^(n-k) represents the probability of getting all those (n-k) tails

4. Example with k=3, N=20:
   * Ways to choose 3 heads from 20 flips: [20!/(20-3)!3!]
   * Probability of those 3 heads: (0.5)^3
   * Probability of 17 tails: (0.5)^17

When multiplied together, this gives the total probability of getting exactly 3 heads in 20 flips.



The bell curve is often called a normal distribution. A normal distribution has two variables: mean μ and variance σ².

For any outcome x, the quadratic difference between the value x and μ can be written as:

   (x−μ)²

   
### Normal Distribution (Bell Curve)

The normal distribution is a symmetrical, bell-shaped curve where:
- Most observations cluster around the central peak (mean)
- The further an observation is from the mean, the rarer it becomes
- Data is equally likely to fall above or below the mean
- The curve never touches the x-axis (asymptotic)


### Key Measures
1. **Mean (μ)**
   - The center point of the distribution
   - Where the peak of the bell curve occurs
   - Represents the average value

2. **Variance (σ²)**
   - Measures how spread out the numbers are from the mean
   - Calculated as the average of squared differences from the mean
   - Formula: σ² = Σ(x - μ)²/N
   - Always positive due to squaring

3. **Standard Deviation (σ)**
   - Square root of variance: σ = √(σ²)
   - More practical measure as it's in the same units as the data
   - In a normal distribution:
     * 68% of data falls within ±1σ of the mean
     * 95% falls within ±2σ
     * 99.7% falls within ±3σ

### Relationship
- The larger the variance/standard deviation, the wider and flatter the bell curve
  * This means the data is more spread out and variable
  * Example: Students' heights in a university (more variation) vs. in a 1st-grade class
  * More extreme values are more common
  * Less predictable data

- The smaller the variance/standard deviation, the narrower and taller the bell curve
  * This means the data points are clustered closely around the mean
  * Example: Manufacturing parts with tight quality control
  * Extreme values are rare
  * More consistent and predictable data

### Real-World Implications
- High variance might indicate:
  * Less control over a process
  * More diversity in the data
  * Potential quality control issues in manufacturing
  * Higher risk in financial data

- Low variance might indicate:
  * Good process control
  * Consistency in measurements
  * Reliable manufacturing process
  * More stable and predictable outcomes


This teaches about the Normal Distribution (also called Gaussian Distribution). 

Normal Distribution:
1. Defined by two parameters:
   - μ (mu) = mean (center of distribution)
   - σ² (sigma squared) = variance (spread of distribution)

2. The formula shown is for standardized score:
   f(x) = (x-μ)²/σ²
   - Measures how far any value x is from the mean
   - In standard deviations

3. Key characteristics shown in graphs:
   - Bell-shaped curve
   - Symmetric around mean
   - Different variances (σ²=1, σ²=4) show different spreads
   - Mean (μ) determines center location

This distribution is fundamental in statistics for modeling naturally occurring phenomena that cluster 
around a central value.

The formula f(x) = (x-μ)²/σ² shown in the image is actually not the complete formula for the normal 
distribution, and it's not the standard deviation.

The standard deviation is σ (sigma), which is the square root of the variance (σ²).

The complete formula for the normal distribution probability density function (PDF) is:

f(x) = (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²))

Where:
- e is Euler's number (≈ 2.71828)
- π (pi) is approximately 3.14159
- μ is the mean
- σ is the standard deviation
- σ² is the variance

The part shown in the image, (x-μ)²/σ², is just one component of the full formula. It represents the 
squared z-score, which measures how many standard deviations an observation is from the mean.

Let me solve this step by step:

1) The function given is:
   f(x) = e^(-1/2 * ((x-μ)²/σ²))

2) This is part of the normal distribution formula, where:
   - μ is the mean
   - σ² is the variance
   - x is any outcome

3) To minimize f(x), we need to minimize the exponent (since e raised to a smaller power gives a smaller result)

4) Looking at the exponent: -1/2 * ((x-μ)²/σ²)
   - The negative in front means that when (x-μ)²/σ² gets larger, the whole exponent gets more negative
   - As the exponent gets more negative, e raised to that power gets closer to zero

5) Therefore, to minimize f(x), we need to make (x-μ)²/σ² as large as possible
   - This happens when |x-μ| is as large as possible
   - This occurs when x approaches either +∞ or -∞

Therefore, both -∞ and +∞ minimize f(x). The answer is: Select both -∞ and +∞


### Understanding the Normal Distribution PDF

The formula f(x) = (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²)) is the probability density function (PDF) of the normal distribution. Here's what each part means:

### Components Breakdown:

1/√(2πσ²): This is the normalization constant that ensures the total area under the curve equals 1
e: Euler's number (≈ 2.71828), the base of natural logarithms
-(x-μ)²/(2σ²): The exponential term that creates the bell shape

### What It Tells Us:

   The function gives the relative likelihood of a random variable X taking on a specific value x
   Higher f(x) values mean that x is more likely to occur
   The curve is highest at x = μ (the mean)
   The curve is symmetric around μ
   Properties:
   Always positive (never below zero)
   Total area under the curve = 1
   Approximately 68% of values lie within ±1σ of μ
   The shape is determined by σ (standard deviation)
   The location is determined by μ (mean)

### Why It's Important:

   Models many natural phenomena (heights, measurement errors, etc.)
   Foundation for statistical inference
   Central Limit Theorem: means of large samples tend toward normal distribution
   Basis for many statistical tests and confidence intervals


### Normal Distribution Explained Simply 🎈

Imagine you and your friends are playing a game where everyone throws a ball at a target:

1. **The Middle (Mean, μ)**
   - Most balls land near the middle of the target
   - This is like the bullseye! 🎯

2. **Spreading Out (Standard Deviation, σ)**
   - Some balls land a little bit away from the middle
   - Few balls land very far from the middle
   - Like ripples in a pond! 🌊

3. **The Bell Shape**
   - If you counted where all the balls landed and made a mountain with blocks:
     * The mountain would be highest in the middle
     * It would slope down evenly on both sides
     * It would look like a bell! 🔔

4. **Real-Life Examples**
   - Heights of people in your class
     * Most are close to average height
     * Few are very tall or very short
   - Shoe sizes in a store
     * Lots of middle sizes
     * Fewer very small or very large sizes
   - Test scores
     * Many students score near the average
     * Fewer get very high or very low scores

The normal distribution can be written as:

N(x; μ, σ²) = 1/√(2πσ²) * e^(-1/2 * (x-μ)²/σ²)

where μ is the mean and σ² is the variance of the normal distribution.

The quadratic term penalizes any deviations from the expectation μ and the exponential squeezes the shape back into the curves.

The shape of the normal distribution is shown below. The y-axis is the probability p(x) and x is the possible mean values of the experiments. There is some information we can extract from the curve:

1. The probability is maximized when x = μ
2. The probability decreases when x is deviating from μ
3. The probability is approaching 0 when x is very far from the mean


Here's a markdown explanation of the Central Limit Theorem with respect to coin flips:

# Central Limit Theorem for Coin Flips

| Number of Flips | Probability Distribution (P) | What We See |
|----------------|----------------------------|-------------|
| Single (n=1) | P(heads) = p, P(tails) = 1-p | Just a binary outcome (heads/tails) |
| A Few (n≈5-10) | [N!/(N-k)!k!] * p^k * (1-p)^(N-k) | Binomial distribution (discrete) |
| Many (n→∞) | (1/√2πσ) * exp(-1/2 * (x-p)²/σ²) | Normal distribution (continuous) |

## Explanation:

1. **Single Flip**:
   - Just one outcome: heads (p) or tails (1-p)
   - Most basic probability scenario
   - No distribution pattern yet

2. **A Few Flips**:
   - Follows binomial distribution
   - Discrete probability outcomes
   - Shows rough pattern but still choppy
   - Example: Getting 3 heads in 5 flips

3. **Many Flips**:
   - Approaches normal distribution
   - Smooth, bell-shaped curve
   - Centers around true probability
   - Standard deviation = √(np(1-p))

This demonstrates how as the number of trials increases, any binomial distribution (like coin flips) will approximate a normal distribution, which is the essence of the Central Limit Theorem.

### Central Limit Theorem Explained 🎲

Imagine you're doing an experiment over and over again (like flipping coins or measuring heights). The Central Limit Theorem (CLT) tells us something amazing:

#### What It Says
- When you take lots of samples and calculate their averages
- These averages will follow a normal distribution (bell curve)
- This happens regardless of the original distribution's shape!

#### Simple Example: Rolling Dice 🎲
1. **Single Roll**
   - Just get one number (1-6)
   - Distribution is uniform (equal chance for each number)

2. **Average of 2 Rolls**
   - Start seeing a pattern
   - More likely to get middle values
   - Less likely to get extreme values

3. **Average of Many Rolls**
   - Beautiful bell curve emerges
   - Most averages cluster around 3.5 (true mean)
   - Very rare to get averages near 1 or 6

#### Why It's Important 🌟
1. **Makes Predictions Possible**
   - Helps us understand large datasets
   - Can predict future behavior
   - Works for almost any type of data

2. **Real-World Applications**
   - Quality control in manufacturing
   - Opinion polls and surveys
   - Medical research
   - Financial market analysis

#### Key Points to Remember
- Needs a large enough sample size (usually n > 30)
- The larger the sample size, the more "normal" it becomes
- Works even if original data is not normal
- The mean of sample means equals the population mean
- The standard deviation of sample means = population standard deviation/√n

Think of it as nature's way of bringing order to chaos - no matter how messy your original data is, when you take enough samples and average them, they organize themselves into a beautiful bell curve! 🎯


## Inferential Statistics

Inferential statistics allows us to draw conclusions about a population based on sample data. Here are key concepts:

### Hypothesis Testing
- **Null Hypothesis (H₀)**: Initial assumption of no effect or no difference
- **Alternative Hypothesis (H₁)**: Contradicts the null hypothesis
- **p-value**: Probability of obtaining results at least as extreme as observed, assuming H₀ is true
- **Significance Level (α)**: Threshold for rejecting H₀ (typically 0.05)

### Confidence Intervals
CI = x̄ ± (z * (σ/√n))
- x̄: Sample mean
- z: z-score for desired confidence level
- σ: Population standard deviation
- n: Sample size

### Common Statistical Tests and Their Applications

#### 1. T-Test (One Sample)
Think of this as checking if your sample is significantly different from a known value. For example, testing if the average height of students in your class differs from the national average.

**Equation**: t = (x̄ - μ₀)/(s/√n)
- x̄: Sample mean
- μ₀: Hypothesized population mean
- s: Sample standard deviation
- n: Sample size

#### 2. Two-Sample T-Test (Independent)
Imagine comparing two groups that are completely separate, like test scores of two different classes. This test tells you if their averages are truly different.

**Equation**: t = (x̄₁ - x̄₂)/√(s₁²/n₁ + s₂²/n₂)
- x̄₁, x̄₂: Means of two samples
- s₁², s₂²: Variances of two samples
- n₁, n₂: Sample sizes

#### 3. Paired T-Test
Perfect for "before and after" scenarios, like measuring weight loss or improvement in test scores for the same group of people.

**Equation**: t = d̄/(sd/√n)
- d̄: Mean difference between pairs
- sd: Standard deviation of differences
- n: Number of pairs

#### 4. Z-Test
Used when you know the population standard deviation - rare in real life but common in theory. Like a t-test but for large samples or known population variance.

**Equation**: z = (x̄ - μ)/(σ/√n)
- x̄: Sample mean
- μ: Population mean
- σ: Population standard deviation
- n: Sample size

#### 5. Chi-Squared Test
Perfect for categorical data, like testing if there's a relationship between favorite color and gender, or if dice rolls are fair.

**Equation**: χ² = Σ((O - E)²/E)
- O: Observed frequencies
- E: Expected frequencies
- Σ: Sum over all categories

#### 6. F-Test
Compares variability between groups. Useful when you want to know if two or more groups have similar spread in their data.

**Equation**: F = s₁²/s₂²
- s₁²: Variance of first sample
- s₂²: Variance of second sample
- Larger variance goes in numerator

Use StatsModels' CompareMeans to calculate the confidence interval for the difference between means:


```python
import numpy as np, statsmodels.stats.api as sms

X1, X2 = np.arange(10,21), np.arange(20,26.5,.5)

cm = sms.CompareMeans(sms.DescrStatsW(X1), sms.DescrStatsW(X2))
print cm.tconfint_diff(usevar='unequal')
```

Output is

```textmate
(-10.414599391793885, -5.5854006082061138)
```

### Understanding Hypothesis Testing

Hypothesis testing is a fundamental tool in statistical analysis that helps us make decisions about populations based on sample data. Here's a comprehensive overview:

#### Core Components

1. **Null Hypothesis (H₀)**
   - The default position or "no effect" claim
   - Example: "There is no difference between treatments"
   - What we assume is true until proven otherwise

2. **Alternative Hypothesis (H₁)**
   - The claim we want to support
   - Example: "The new treatment is more effective"
   - What we need evidence to accept

#### Types of Errors

| | H₀ is True | H₀ is False |
|---|---|---|
| Reject H₀ | Type I Error (α) | Correct Decision |
| Fail to Reject H₀ | Correct Decision | Type II Error (β) |

#### Key Concepts

1. **Significance Level (α)**
   - Probability of Type I error
   - Usually set at 0.05 (5%)
   - Lower α means stronger evidence needed

2. **Power (1 - β)**
   - Probability of correctly rejecting false H₀
   - Increases with:
     * Larger sample size
     * Larger effect size
     * Lower variability

3. **P-value**
   - Probability of getting results as extreme as observed
   - If p < α: Reject H₀
   - If p ≥ α: Fail to reject H₀

#### Common Misconceptions

1. **About P-values**
   - ❌ P-value is NOT probability H₀ is true
   - ✓ It's probability of data, given H₀ is true

2. **About "Failing to Reject"**
   - ❌ Not rejecting ≠ proving H₀
   - ✓ Just insufficient evidence against H₀

3. **About Significance**
   - ❌ Statistical ≠ practical significance
   - ✓ Consider effect size and context

#### Best Practices

1. **Before Testing**
   - Define hypotheses clearly
   - Choose appropriate test
   - Set α level
   - Determine sample size needed

2. **During Analysis**
   - Check assumptions
   - Use appropriate test statistic
   - Calculate effect size

3. **After Testing**
   - Report exact p-values
   - Consider practical significance
   - Document all decisions

### Relationship Between Confidence Intervals and Hypothesis Testing

Confidence intervals and hypothesis testing are two sides of the same inferential statistics coin. Here's how they relate:

1. **Complementary Information**
   - Hypothesis tests tell us whether there's a significant difference
   - Confidence intervals tell us the range of plausible values for that difference

2. **Decision Alignment**
   - If a 95% confidence interval doesn't contain the null hypothesis value:
     * The corresponding two-tailed hypothesis test will reject H₀ at α = 0.05
   - If it does contain the null value:
     * The test will fail to reject H₀ at α = 0.05

3. **Practical Advantages**
   - Confidence intervals provide more information than hypothesis tests alone
   - They show both statistical significance AND practical importance
   - Example: A difference might be statistically significant (p < 0.05) but practically tiny

4. **Mathematical Connection**
   ```
   CI = Point Estimate ± (Critical Value × Standard Error)
   Test Statistic = (Point Estimate - Null Value) / Standard Error
   ```

5. **Visual Interpretation**
   - If CI contains null value → Cannot reject H₀
   - If CI excludes null value → Reject H₀
   
This dual approach (using both CIs and hypothesis tests) often provides better insights than either method alone.



–––––––––––––––––––––––––––––––––––––––

<br>

![Alt Text](cancer.png)

<br>

### 1. Descriptive Statistics
Descriptive statistics is about describing our collected data using the measures discussed throughout this lesson: measures of center, measures of spread, the shape of our distribution, and outliers. We can also use plots of our data to gain a better understanding.

### 2. Inferential Statistics
Inferential Statistics is about using our collected data to draw conclusions to a larger population. Performing inferential statistics well requires that we take a sample that accurately represents our population of interest.

A common way to collect data is via a survey. However, surveys may be extremely biased depending on the types of questions that are asked, and the way the questions are asked. This is a topic you should think about when tackling a project.

We looked at specific examples that allowed us to identify the

   Population - our entire group of interest.
   Parameter - numeric summary about a population
   Sample - a subset of the population
   Statistic numeric summary about a sample



A sampling distribution is the distribution of a statistic. Here we looked at the distribution of the proportion for samples of 5 students. 



### Law of Large Numbers (LLN)

The Law of Large Numbers states that as the sample size increases, the sample mean converges to the true population mean. There are two versions:

1. **Weak Law (WLLN)**:
   - States that for a large n, the sample mean (x̄) is likely to be close to μ
   - P(|x̄ - μ| < ε) → 1 as n → ∞, for any ε > 0

2. **Strong Law (SLLN)**:
   - States that x̄ converges to μ with probability 1
   - P(lim n→∞ x̄ = μ) = 1

### Central Limit Theorem (CLT)

The Central Limit Theorem is a fundamental principle stating that when independent random variables are added, their properly normalized sum tends toward a normal distribution, regardless of the underlying distribution. Key points:

1. **Requirements**:
   - Independent and identically distributed variables
   - Sample size sufficiently large (usually n ≥ 30)

2. **Mathematical Expression**:
   For sample mean x̄:
   (x̄ - μ)/(σ/√n) → N(0,1)

3. **Practical Implications**:
   - Sample means follow normal distribution
   - Standard error = σ/√n
   - Enables statistical inference and hypothesis testing
   - Forms basis for confidence intervals

The Central Limit Theorem states that with a large enough sample size the sampling distribution of the mean will be normally distributed.

The Central Limit Theorem actually applies to these well-known statistics:

- Sample means (x̄)
- Sample proportions (p)
- Difference in sample means (x̄₁ - x̄₂)
- Difference in sample proportions (p₁ - p₂)

And it applies to additional statistics, but it doesn't apply to all statistics! You will see more on this towards the end of this lesson.

### Where Central Limit Theorem Doesn't Apply

The CLT has important limitations and doesn't work in these cases:

1. **Small Sample Sizes**:
   - When n < 30, especially for highly skewed distributions
   - For extremely non-normal data, might need n > 100

2. **Non-Independent Samples**:
   - Time series data with autocorrelation
   - Clustered or hierarchical data
   - Samples with strong dependencies

3. **Infinite Variance**:
   - Distributions with undefined variance (e.g., Cauchy distribution)
   - Heavy-tailed distributions where variance doesn't exist

4. **Sample Maximum/Minimum**:
   - Extreme value statistics follow different distributions
   - Maximum/minimum values follow extreme value distributions, not normal

5. **Ratio Statistics**:
   - Ratios of random variables often have undefined moments
   - Can lead to heavy-tailed distributions

### Parameter Estimation Methods

#### 1. Maximum Likelihood Estimation (MLE)
- Finds parameter values that maximize the likelihood of observing the given data
- Mathematical formulation: θ̂ = argmax L(θ|x)
- Properties:
  - Consistent: Converges to true value as sample size increases
  - Asymptotically efficient: Achieves minimum variance
  - Example: For normal distribution, MLE of μ is sample mean x̄

#### 2. Method of Moments Estimation (MME)
- Equates sample moments with theoretical population moments
- Steps:
  1. Calculate sample moments (mean, variance, etc.)
  2. Set equal to theoretical moments
  3. Solve for parameters
- Example: For normal distribution
  - First moment (mean): x̄ = μ
  - Second moment (variance): s² = σ²

#### 3. Bayesian Estimation
- Combines prior knowledge with observed data
- Uses Bayes' theorem: P(θ|x) ∝ P(x|θ)P(θ)
- Components:
  - Prior distribution P(θ): Initial beliefs
  - Likelihood P(x|θ): Data evidence
  - Posterior distribution P(θ|x): Updated beliefs
- Advantages:
  - Incorporates prior knowledge
  - Provides full probability distribution
  - Handles uncertainty naturally



Bootstrapping is sampling with replacement. Using random.choice in Python actually samples in this way. Where the probability of any number in our set stays the same regardless of how many times it has been chosen. Flipping a coin and rolling a die is like bootstrap sampling as well, as rolling a 6 in one scenario doesn't mean that 6 is less likely later.



### Bootstrapping Made Simple

Think of bootstrapping like this:

Imagine you have a bag of 10 marbles, and you want to understand the typical color distribution in a much larger population of marbles from the same factory. Here's what bootstrapping does:

1. You reach into the bag and pick a marble
2. Write down its color
3. Put it back in the bag
4. Repeat many times (say 1000 times)

Even though you only have 10 real marbles, by putting each marble back before picking again (sampling with replacement), you can create many different samples. This helps you:
- Estimate how much your statistics might vary
- Build confidence intervals
- Make predictions about the larger population

It's like making the most of your limited data by reusing it in a smart way. The key is that each time you pick, every marble has an equal chance of being chosen, even if it was just picked before.

Real-world example: If you want to understand average customer spending but only have data from 100 customers, bootstrapping lets you create 1000s of different combinations of these 100 customers to better understand the possible range of average spending.


### Bootstrapping in Statistics

Bootstrapping is a resampling technique that involves sampling with replacement from the original dataset. Key aspects include:

#### Core Concepts
- **Sampling with Replacement**: Each observation can be selected multiple times
- **Random Selection**: Each observation has equal probability of being chosen each time
- **Large Number of Resamples**: Typically 1000+ bootstrap samples

#### Applications
1. **Estimating Standard Errors**:
   - Calculate statistic for each bootstrap sample
   - Standard deviation of bootstrap statistics estimates standard error

2. **Confidence Intervals**:
   - Percentile method: Use quantiles of bootstrap distribution
   - BCa (Bias-Corrected and accelerated) method for better accuracy

3. **Hypothesis Testing**:
   - Generate null distribution through bootstrapping
   - Compare observed statistic to bootstrap distribution

#### Advantages
- Non-parametric: No distribution assumptions
- Works with complex statistics
- Effective for small sample sizes
- Provides empirical distribution of statistics

#### Limitations
- Assumes sample represents population well
- Computationally intensive
- May not work well with dependent data
- Cannot extrapolate beyond observed range

#### Python Implementation Example:

```python
# Basic bootstrap sample
np.random.choice(data, size=len(data), replace=True)
```


### Importance of Sampling Distributions

Two popular inferential techniques are confidence intervals and hypothesis testing.

There are many formulas and built-in calculators available to calculate the values for these techniques. However, these formulas often hide their assumptions and potential biases. With sampling distributions and bootstrapping, you'll avoid needing to rely on these formulas and can figure out how to calculate these values without running into these issues.

### Relationship Between Sampling Distributions and Confidence Intervals

Sampling distributions and confidence intervals are closely connected. Here's how they work together:

#### The Connection
1. **Sampling Distribution**:
   - Shows all possible values of a statistic (like mean) from different samples
   - The spread tells us how much sample means typically vary
   - Shape often follows normal distribution (thanks to CLT)

2. **Confidence Interval**:
   - Uses the sampling distribution's properties
   - Usually captures middle 95% (or other %) of sampling distribution
   - Formula: Point Estimate ± (Critical Value × Standard Error)

#### Simple Example
Imagine measuring heights in a class:
- Sample mean = 170cm
- If we know sampling distribution is normal
- And standard error = 2cm
- 95% confidence interval = 170 ± (1.96 × 2) = (166.08cm, 173.92cm)

#### Why This Matters
- Sampling distribution tells us how much sample statistics typically vary
- This variation directly determines confidence interval width
- Larger sample size → Narrower sampling distribution → Tighter confidence intervals
- More variable data → Wider sampling distribution → Wider confidence intervals

### Traditional vs. Bootstrap Confidence Intervals

#### Traditional Methods
1. **Assumptions**:
   - Requires normal distribution
   - Needs known population parameters
   - Uses theoretical formulas

2. **Calculation**:
   - Based on standard error formulas
   - Uses t or z critical values
   - Point estimate ± (critical value × standard error)

#### Bootstrap Methods
1. **Advantages**:
   - No distributional assumptions
   - Works with any statistic
   - Handles complex sampling designs

2. **Process**:
   - Resamples from actual data
   - Creates empirical distribution
   - Uses percentiles of resampled statistics

#### Key Differences
1. **Flexibility**:
   - Traditional: Limited to specific distributions
   - Bootstrap: Works with any distribution

2. **Accuracy**:
   - Traditional: More precise when assumptions met
   - Bootstrap: Better for non-normal/unknown distributions

3. **Computation**:
   - Traditional: Quick, formula-based
   - Bootstrap: Computationally intensive, simulation-based


–––––––––––––––––––––––––––––––––––––––

<br>

![Alt Text](cancer.png)

<br>

### 1. Descriptive Statistics
Descriptive statistics is about describing our collected data using the measures discussed throughout this lesson: measures of center, measures of spread, the shape of our distribution, and outliers. We can also use plots of our data to gain a better understanding.

### 2. Inferential Statistics
Inferential Statistics is about using our collected data to draw conclusions to a larger population. Performing inferential statistics well requires that we take a sample that accurately represents our population of interest.

A common way to collect data is via a survey. However, surveys may be extremely biased depending on the types of questions that are asked, and the way the questions are asked. This is a topic you should think about when tackling a project.

We looked at specific examples that allowed us to identify the

   Population - our entire group of interest.
   Parameter - numeric summary about a population
   Sample - a subset of the population
   Statistic numeric summary about a sample



A sampling distribution is the distribution of a statistic. Here we looked at the distribution of the proportion for samples of 5 students. 



### Law of Large Numbers (LLN)

The Law of Large Numbers states that as the sample size increases, the sample mean converges to the true population mean. There are two versions:

1. **Weak Law (WLLN)**:
   - States that for a large n, the sample mean (x̄) is likely to be close to μ
   - P(|x̄ - μ| < ε) → 1 as n → ∞, for any ε > 0

2. **Strong Law (SLLN)**:
   - States that x̄ converges to μ with probability 1
   - P(lim n→∞ x̄ = μ) = 1

### Central Limit Theorem (CLT)

The Central Limit Theorem is a fundamental principle stating that when independent random variables are added, their properly normalized sum tends toward a normal distribution, regardless of the underlying distribution. Key points:

1. **Requirements**:
   - Independent and identically distributed variables
   - Sample size sufficiently large (usually n ≥ 30)

2. **Mathematical Expression**:
   For sample mean x̄:
   (x̄ - μ)/(σ/√n) → N(0,1)

3. **Practical Implications**:
   - Sample means follow normal distribution
   - Standard error = σ/√n
   - Enables statistical inference and hypothesis testing
   - Forms basis for confidence intervals

The Central Limit Theorem states that with a large enough sample size the sampling distribution of the mean will be normally distributed.

The Central Limit Theorem actually applies to these well-known statistics:

- Sample means (x̄)
- Sample proportions (p)
- Difference in sample means (x̄₁ - x̄₂)
- Difference in sample proportions (p₁ - p₂)

And it applies to additional statistics, but it doesn't apply to all statistics! You will see more on this towards the end of this lesson.

### Where Central Limit Theorem Doesn't Apply

The CLT has important limitations and doesn't work in these cases:

1. **Small Sample Sizes**:
   - When n < 30, especially for highly skewed distributions
   - For extremely non-normal data, might need n > 100

2. **Non-Independent Samples**:
   - Time series data with autocorrelation
   - Clustered or hierarchical data
   - Samples with strong dependencies

3. **Infinite Variance**:
   - Distributions with undefined variance (e.g., Cauchy distribution)
   - Heavy-tailed distributions where variance doesn't exist

4. **Sample Maximum/Minimum**:
   - Extreme value statistics follow different distributions
   - Maximum/minimum values follow extreme value distributions, not normal

5. **Ratio Statistics**:
   - Ratios of random variables often have undefined moments
   - Can lead to heavy-tailed distributions

### Parameter Estimation Methods

#### 1. Maximum Likelihood Estimation (MLE)
- Finds parameter values that maximize the likelihood of observing the given data
- Mathematical formulation: θ̂ = argmax L(θ|x)
- Properties:
  - Consistent: Converges to true value as sample size increases
  - Asymptotically efficient: Achieves minimum variance
  - Example: For normal distribution, MLE of μ is sample mean x̄

#### 2. Method of Moments Estimation (MME)
- Equates sample moments with theoretical population moments
- Steps:
  1. Calculate sample moments (mean, variance, etc.)
  2. Set equal to theoretical moments
  3. Solve for parameters
- Example: For normal distribution
  - First moment (mean): x̄ = μ
  - Second moment (variance): s² = σ²

#### 3. Bayesian Estimation
- Combines prior knowledge with observed data
- Uses Bayes' theorem: P(θ|x) ∝ P(x|θ)P(θ)
- Components:
  - Prior distribution P(θ): Initial beliefs
  - Likelihood P(x|θ): Data evidence
  - Posterior distribution P(θ|x): Updated beliefs
- Advantages:
  - Incorporates prior knowledge
  - Provides full probability distribution
  - Handles uncertainty naturally



Bootstrapping is sampling with replacement. Using random.choice in Python actually samples in this way. Where the probability of any number in our set stays the same regardless of how many times it has been chosen. Flipping a coin and rolling a die is like bootstrap sampling as well, as rolling a 6 in one scenario doesn't mean that 6 is less likely later.



### Bootstrapping Made Simple

Think of bootstrapping like this:

Imagine you have a bag of 10 marbles, and you want to understand the typical color distribution in a much larger population of marbles from the same factory. Here's what bootstrapping does:

1. You reach into the bag and pick a marble
2. Write down its color
3. Put it back in the bag
4. Repeat many times (say 1000 times)

Even though you only have 10 real marbles, by putting each marble back before picking again (sampling with replacement), you can create many different samples. This helps you:
- Estimate how much your statistics might vary
- Build confidence intervals
- Make predictions about the larger population

It's like making the most of your limited data by reusing it in a smart way. The key is that each time you pick, every marble has an equal chance of being chosen, even if it was just picked before.

Real-world example: If you want to understand average customer spending but only have data from 100 customers, bootstrapping lets you create 1000s of different combinations of these 100 customers to better understand the possible range of average spending.


### Bootstrapping in Statistics

Bootstrapping is a resampling technique that involves sampling with replacement from the original dataset. Key aspects include:

#### Core Concepts
- **Sampling with Replacement**: Each observation can be selected multiple times
- **Random Selection**: Each observation has equal probability of being chosen each time
- **Large Number of Resamples**: Typically 1000+ bootstrap samples

#### Applications
1. **Estimating Standard Errors**:
   - Calculate statistic for each bootstrap sample
   - Standard deviation of bootstrap statistics estimates standard error

2. **Confidence Intervals**:
   - Percentile method: Use quantiles of bootstrap distribution
   - BCa (Bias-Corrected and accelerated) method for better accuracy

3. **Hypothesis Testing**:
   - Generate null distribution through bootstrapping
   - Compare observed statistic to bootstrap distribution

#### Advantages
- Non-parametric: No distribution assumptions
- Works with complex statistics
- Effective for small sample sizes
- Provides empirical distribution of statistics

#### Limitations
- Assumes sample represents population well
- Computationally intensive
- May not work well with dependent data
- Cannot extrapolate beyond observed range

#### Python Implementation Example:

```python
# Basic bootstrap sample
np.random.choice(data, size=len(data), replace=True)

```

Follow these instructions to make the following change to my code document.

Instruction: Adding statistical tests explanations before the divider line

Code Edit:
```
{{ ... }}

Use StatsModels' CompareMeans to calculate the confidence interval for the difference between means:


```python
import numpy as np, statsmodels.stats.api as sms

X1, X2 = np.arange(10,21), np.arange(20,26.5,.5)

cm = sms.CompareMeans(sms.DescrStatsW(X1), sms.DescrStatsW(X2))
print cm.tconfint_diff(usevar='unequal')
```

Output is

```textmate
(-10.414599391793885, -5.5854006082061138)
```

### Common Statistical Tests and Their Applications

#### 1. T-Test (One Sample)
Think of this as checking if your sample is significantly different from a known value. For example, testing if the average height of students in your class differs from the national average.

**Equation**: t = (x̄ - μ₀)/(s/√n)
- x̄: Sample mean
- μ₀: Hypothesized population mean
- s: Sample standard deviation
- n: Sample size

#### 2. Two-Sample T-Test (Independent)
Imagine comparing two groups that are completely separate, like test scores of two different classes. This test tells you if their averages are truly different.

**Equation**: t = (x̄₁ - x̄₂)/√(s₁²/n₁ + s₂²/n₂)
- x̄₁, x̄₂: Means of two samples
- s₁², s₂²: Variances of two samples
- n₁, n₂: Sample sizes

#### 3. Paired T-Test
Perfect for "before and after" scenarios, like measuring weight loss or improvement in test scores for the same group of people.

**Equation**: t = d̄/(sd/√n)
- d̄: Mean difference between pairs
- sd: Standard deviation of differences
- n: Number of pairs

#### 4. Z-Test
Used when you know the population standard deviation - rare in real life but common in theory. Like a t-test but for large samples or known population variance.

**Equation**: z = (x̄ - μ)/(σ/√n)
- x̄: Sample mean
- μ: Population mean
- σ: Population standard deviation
- n: Sample size

#### 5. Chi-Squared Test
Perfect for categorical data, like testing if there's a relationship between favorite color and gender, or if dice rolls are fair.

**Equation**: χ² = Σ((O - E)²/E)
- O: Observed frequencies
- E: Expected frequencies
- Σ: Sum over all categories

#### 6. F-Test
Compares variability between groups. Useful when you want to know if two or more groups have similar spread in their data.

**Equation**: F = s₁²/s₂²
- s₁²: Variance of first sample
- s₂²: Variance of second sample
- Larger variance goes in numerator


### Relationship Between Confidence Intervals and Hypothesis Testing

Confidence intervals and hypothesis testing are two sides of the same inferential statistics coin. Here's how they relate:

1. **Complementary Information**
   - Hypothesis tests tell us whether there's a significant difference
   - Confidence intervals tell us the range of plausible values for that difference

2. **Decision Alignment**
   - If a 95% confidence interval doesn't contain the null hypothesis value:
     * The corresponding two-tailed hypothesis test will reject H₀ at α = 0.05
   - If it does contain the null value:
     * The test will fail to reject H₀ at α = 0.05

3. **Practical Advantages**
   - Confidence intervals provide more information than hypothesis tests alone
   - They show both statistical significance AND practical importance
   - Example: A difference might be statistically significant (p < 0.05) but practically tiny

4. **Mathematical Connection**
   ```
   CI = Point Estimate ± (Critical Value × Standard Error)
   Test Statistic = (Point Estimate - Null Value) / Standard Error
   ```

5. **Visual Interpretation**
   - If CI contains null value → Cannot reject H₀
   - If CI excludes null value → Reject H₀
   
This dual approach (using both CIs and hypothesis tests) often provides better insights than either method alone.




### Understanding Hypothesis Testing

Hypothesis testing is a fundamental tool in statistical analysis that helps us make decisions about populations based on sample data. Here's a comprehensive overview:

#### Core Components

1. **Null Hypothesis (H₀)**
   - The default position or "no effect" claim
   - Example: "There is no difference between treatments"
   - What we assume is true until proven otherwise

2. **Alternative Hypothesis (H₁)**
   - The claim we want to support
   - Example: "The new treatment is more effective"
   - What we need evidence to accept

#### Types of Errors

| | H₀ is True | H₀ is False |
|---|---|---|
| Reject H₀ | Type I Error (α) | Correct Decision |
| Fail to Reject H₀ | Correct Decision | Type II Error (β) |

#### Key Concepts

1. **Significance Level (α)**
   - Probability of Type I error
   - Usually set at 0.05 (5%)
   - Lower α means stronger evidence needed

2. **Power (1 - β)**
   - Probability of correctly rejecting false H₀
   - Increases with:
     * Larger sample size
     * Larger effect size
     * Lower variability

3. **P-value**
   - Probability of getting results as extreme as observed
   - If p < α: Reject H₀
   - If p ≥ α: Fail to reject H₀

#### Common Misconceptions

1. **About P-values**
   - ❌ P-value is NOT probability H₀ is true
   - ✓ It's probability of data, given H₀ is true

2. **About "Failing to Reject"**
   - ❌ Not rejecting ≠ proving H₀
   - ✓ Just insufficient evidence against H₀

3. **About Significance**
   - ❌ Statistical ≠ practical significance
   - ✓ Consider effect size and context

#### Best Practices

1. **Before Testing**
   - Define hypotheses clearly
   - Choose appropriate test
   - Set α level
   - Determine sample size needed

2. **During Analysis**
   - Check assumptions
   - Use appropriate test statistic
   - Calculate effect size

3. **After Testing**
   - Report exact p-values
   - Consider practical significance
   - Document all decisions

{{ ... }}

```

Follow these instructions to make the following change to my code document.

Instruction: Cleaning up duplicates and properly organizing the hypothesis testing section

Code Edit:
```
{{ ... }}

### Understanding Hypothesis Testing

Hypothesis testing is a fundamental tool in statistical analysis that helps us make decisions about populations based on sample data. Here's a comprehensive overview:

#### Core Components

1. **Null Hypothesis (H₀)**
   - The default position or "no effect" claim
   - Example: "There is no difference between treatments"
   - What we assume is true until proven otherwise

2. **Alternative Hypothesis (H₁)**
   - The claim we want to support
   - Example: "The new treatment is more effective"
   - What we need evidence to accept

#### Types of Errors

| | H₀ is True | H₀ is False |
|---|---|---|
| Reject H₀ | Type I Error (α) | Correct Decision |
| Fail to Reject H₀ | Correct Decision | Type II Error (β) |

#### Key Concepts

1. **Significance Level (α)**
   - Probability of Type I error
   - Usually set at 0.05 (5%)
   - Lower α means stronger evidence needed

2. **Power (1 - β)**
   - Probability of correctly rejecting false H₀
   - Increases with:
     * Larger sample size
     * Larger effect size
     * Lower variability

3. **P-value**
   - Probability of getting results as extreme as observed
   - If p < α: Reject H₀
   - If p ≥ α: Fail to reject H₀

#### Common Misconceptions

1. **About P-values**
   - ❌ P-value is NOT probability H₀ is true
   - ✓ It's probability of data, given H₀ is true

2. **About "Failing to Reject"**
   - ❌ Not rejecting ≠ proving H₀
   - ✓ Just insufficient evidence against H₀

3. **About Significance**
   - ❌ Statistical ≠ practical significance
   - ✓ Consider effect size and context

#### Best Practices

1. **Before Testing**
   - Define hypotheses clearly
   - Choose appropriate test
   - Set α level
   - Determine sample size needed

2. **During Analysis**
   - Check assumptions
   - Use appropriate test statistic
   - Calculate effect size

3. **After Testing**
   - Report exact p-values
   - Consider practical significance
   - Document all decisions



–––––––––––––––––––––––––––––––––––––––

<br>

![Alt Text](cancer.png)

<br>
