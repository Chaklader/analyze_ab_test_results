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
- Making informed decisions about testing procedures


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

![Cancer Statistics Visualization](images/cancer.png)

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


<br>

![P value](images/graph.png)

<br>

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

### Hypothesis Testing - Testing Population Parameters, Not Statistics

- **Key Concept**: Hypothesis tests are always performed on population parameters, never on sample statistics.
  
- **Reasoning**: 
  - Sample statistics (like sample mean, sample proportion) are values we already have calculated from our data
  - These statistics are known values, so there's no uncertainty to test
  - The purpose of hypothesis testing is to make inferences about unknown population parameters using sample data
  
- **Example**:
  - We test hypotheses about population means (μ), not sample means (x̄)
  - We test hypotheses about population proportions (p), not sample proportions (p̂)
  - We test hypotheses about population standard deviations (σ), not sample standard deviations (s)

- **Remember**: The goal of inferential statistics is to use sample data to make conclusions about unknown population parameters.

Common hypothesis tests include:

1. Testing a population mean (One sample t-test)(opens in a new tab).
2. Testing the difference in means (Two-sample t-test)(opens in a new tab)
3. Testing the difference before and after some treatment on the same individual (Paired t-test)(opens in a new tab)
4. Testing a population proportion (One sample z-test)(opens in a new tab)
5. Testing the difference between population proportions (Two sample z-test)(opens in a new tab)


You can use one of these sites to provide a t-table or z-table to support one of the above approaches:

t-table(opens in a new tab)
t-table or z-table

### Rules for Setting Up Hypotheses
Null Hypothesis (H₀) Properties:
Is assumed true at the start
Usually states "no effect" or "groups are equal"
Contains an equality sign (=, ≤, or ≥)
Like "innocent until proven guilty"
Alternative Hypothesis (H₁) Properties:
What we want to prove
Contains the opposite of H₀ (≠, >, or <)
Cannot overlap with H₀
Like "proving guilty"
Example: Legal System Analogy
H₀: Innocent (default position)
H₁: Guilty (needs to be proven)



### Understanding Type I and Type II Errors in Hypothesis Testing

## Definitions

### Type I Error (α - Alpha)
- Occurs when we **reject a true null hypothesis** (false positive)
- Probability is typically set at 0.05 (5% significance level)
- Example: Concluding a new treatment works when it actually doesn't

### Type II Error (β - Beta)
- Occurs when we **fail to reject a false null hypothesis** (false negative)
- Related to statistical power (1 - β)
- Example: Concluding a treatment doesn't work when it actually does

## Comparison Table
| Aspect | Type I Error | Type II Error |
|--------|--------------|---------------|
| Definition | Rejecting true H₀ | Failing to reject false H₀ |
| Symbol | α (alpha) | β (beta) |
| Common Value | 0.05 | 0.20 |
| Type of Error | False Positive | False Negative |

## Real-World Examples
1. **Medical Testing**
   - Type I: Diagnosing healthy person as sick
   - Type II: Missing actual disease in sick patient

2. **Quality Control**
   - Type I: Rejecting good batch of products
   - Type II: Accepting defective batch

## Relationship Between Errors
- Reducing one type of error typically increases the other
- Sample size increase can reduce both types
- Trade-off depends on relative costs of each error

## Controlling Errors
- Type I controlled by significance level (α)
- Type II reduced by:
  - Increasing sample size
  - Reducing variability
  - Increasing effect size
  - Setting higher significance level

This understanding is crucial for:
- Research design
- Sample size determination
- Statistical power analysis
- Decision-making in hypothesis testing


# Type I Errors

Type I errors have the following features:

1. You should set up your null and alternative hypotheses so that the worse of your errors is the type I error.
2. They are denoted by the symbol α or alpha
3. The definition of a type I error is: Deciding the alternative (H₁) is true when actually (H₀) is true.
4. Type I errors are often called false positives.

# Type II Errors

1. They are denoted by the symbol β or beta
2. The definition of a type II error is: Deciding the null (H₀) is true when actually (H₁) is true.
3. Type II errors are often called false negatives.


In the most extreme case, we can always choose one hypothesis (say always choosing the null) to ensure that a particular error never occurs (never a type I error, assuming we always choose the null). However, more generally, there is a relationship where a single set of data decreasing your chance of one type of error increases the chance of the other error occurring.


# Understanding P-Values


The definition of a p-value is the probability of observing your statistic (or one more extreme in favor of the alternative) if the null hypothesis is true.

In this video, you learned exactly how to calculate this value. The more extreme in favor of the alternative portion of this statement determines the shading associated with your p-value.

Therefore, you have the following cases:

If your parameter is greater than some value in the alternative hypothesis, your shading will look like this to obtain your p-value:


<br>

![P value](images/1.png)

<br>

If your parameter is less than some value in the alternative hypothesis, your shading would look like this to obtain your p-value:


<br>

![P value](images/2.png)

<br>

If your parameter is not equal to some value in the alternative hypothesis, your shading would look like this to obtain your p-value:

<br>

![P value](images/3.png)

<br>

You could integrate the sampling distribution to obtain the area for each of these p-values. Alternatively, you will be simulating to obtain these proportions in the next concepts.




## Definition and Interpretation
- The p-value is the probability of obtaining test results at least as extreme as the observed results, assuming the null hypothesis is true
- Smaller p-values indicate stronger evidence against the null hypothesis
- It answers: "If H₀ were true, how likely would we see data this extreme?"

## Key Properties
1. **Range**: Always between 0 and 1
2. **Threshold**: Compared to significance level (α)
   - If p ≤ α: Reject H₀
   - If p > α: Fail to reject H₀
3. **NOT the probability that H₀ is true**

## Common Misconceptions
- ❌ P-value is NOT the probability of making a mistake
- ❌ P-value is NOT the probability that H₀ is true
- ❌ P-value is NOT the probability that H₁ is true
- ✅ P-value IS the probability of seeing data this extreme if H₀ is true

## Example Interpretation
P-value = 0.03 means:
- If H₀ were true
- And we repeated the experiment many times
- We would see results this extreme only 3% of the time

## Decision Making
- Traditional significance levels:
  - α = 0.05 (5%)
  - α = 0.01 (1%)
- Decision rule:
  - p < α: Strong evidence against H₀
  - p ≥ α: Insufficient evidence against H₀


# Understanding P-values in Hypothesis Testing

## Definition and Interpretation
The p-value represents the probability of observing your test statistic (or a more extreme value) if the null hypothesis is true.

## Key Concepts

### Small P-values (p ≤ α)
- Indicates strong evidence against the null hypothesis
- Suggests data likely came from a different distribution
- Leads to rejecting H₀
- Typically use α = 0.05 as threshold

### Large P-values (p > α)
- Indicates data is consistent with null hypothesis
- Insufficient evidence to reject H₀
- Fail to reject (but don't "accept") H₀

## Decision Rules
```
If p-value ≤ α: Reject H₀
If p-value > α: Fail to reject H₀
```

## Important Notes
1. P-values do not prove hypotheses true or false
2. They quantify the strength of evidence against H₀
3. Small p-values indicate evidence against H₀
4. P-value should be compared to predetermined α level
5. α represents acceptable Type I error rate

## Common Mistakes to Avoid
- Don't interpret p > α as "proving" H₀
- Don't confuse statistical with practical significance
- Remember: failing to reject ≠ accepting H₀

## Example
If α = 0.05 and p = 0.03:
- Since 0.03 < 0.05
- We reject H₀
- Evidence supports alternative hypothesis

Therefore, the wording used in conclusions of hypothesis testing includes: We reject the null hypothesis, or We fail to reject the null hypothesis. This lends itself to the idea that you start with the null hypothesis true by default, and "choosing" the null at the end of the test would have been the choice even if no data were collected.

When performing more than one hypothesis test, your type I error compounds. To correct this, a common technique is called the Bonferroni correction. This correction is very conservative but says that your new type I error rate should be the error rate you actually want to be divided by the number of tests you perform.

Therefore, if you would like to hold a type I error rate of 1% for each of the 20 hypothesis tests, the Bonferroni corrected rate would be 0.01/20 = 0.0005. This would be the new rate you should use to compare the p-value for each of the 20 tests to make your decision.

Other Techniques
Additional techniques to protect against compounding type I errors include:

1. Tukey correction(opens in a new tab)
2. Q-values

# Bonferroni Correction in Multiple Hypothesis Testing

## Purpose
- Controls family-wise error rate (FWER) when performing multiple hypothesis tests
- Reduces probability of Type I errors (false positives)
- More conservative than other multiple testing corrections

## Formula
```
α_adjusted = α / n
```
Where:
- α is original significance level (typically 0.05)
- n is number of independent tests
- α_adjusted is new threshold for each test

## Example
If running 5 tests with α = 0.05:
- α_adjusted = 0.05/5 = 0.01
- Each individual test uses 0.01 as significance threshold
- Must have p < 0.01 to reject null hypothesis

## Advantages
- Simple to calculate and implement
- Guarantees control of family-wise error rate
- Conservative approach to Type I error control

## Disadvantages
- Can be too conservative
- Reduces statistical power
- May increase Type II errors
- Assumes tests are independent

## When to Use
- Multiple independent hypothesis tests
- Strong control of false positives needed
- Small number of comparisons
- Tests are independent

## Alternative Methods
- Holm's sequential Bonferroni
- False Discovery Rate (FDR)
- Benjamini-Hochberg procedure
- Šidák correction

## Practical Example
Testing 3 drug treatments:
- Original α = 0.05
- Adjusted α = 0.05/3 = 0.0167
- Each test must meet p < 0.0167 for significance


# A/B Testing 

A/B tests test changes on a web page by running an experiment where a control group sees the old version while the experiment group sees the new version. A metric is then chosen to measure the level of engagement from users in each group. These results are then used to judge whether one version is more effective than the other. A/B testing is very much like hypothesis testing with the following hypotheses:

   1. Null Hypothesis: The new version is no better, or even worse, than the old version
   2. Alternative Hypothesis: The new version is better than the old version

If we fail to reject the null hypothesis, the results would suggest keeping the old version. If we reject the null hypothesis, the results would suggest launching the change. These tests can be used for a wide variety of changes, from large feature additions to small adjustments in color, to see what change maximizes your metric the most.

A/B testing also has its drawbacks. It can help you compare two options, but it can't tell you about an option you haven’t considered. It can also produce bias results when tested on existing users due to factors like change aversion and novelty effect.

   1. Change Aversion: Existing users may give an unfair advantage to the old version simply because they are unhappy with the change, even if it’s ultimately for the better.
   2. Novelty Effect: Existing users may give an unfair advantage to the new version because they’re excited or drawn to the change, even if it isn’t any better in the long run. You'll learn more about factors like these later.


Let's recap the steps we took to analyze the results of this A/B test.

1. We computed the observed difference between the metric, click-through rate, for the control and experiment groups.
2. We simulated the sampling distribution for the difference in proportions (or difference in click-through rates).
3. We used this sampling distribution to simulate the distribution under the null hypothesis by creating a random normal distribution centered at 0 with the same spread and size.
4. We computed the p-value by finding the proportion of values in the null distribution greater than our observed difference.
5. We used this p-value to determine the statistical significance of our observed difference.




# COURSE - 5: ALGORITHMS 


## Regression 

Scatter plots
Scatter plots are a common visual for comparing two quantitative variables. A common summary statistic that relates to a scatter plot is the correlation coefficient commonly denoted by r.

Though there are a few different ways(opens in a new tab) to measure correlation between two variables, the most common way is with Pearson's correlation coefficient(opens in a new tab). Pearson's correlation coefficient provides the:

1. Strength
2. Direction

of a linear relationship. Spearman's Correlation Coefficient(opens in a new tab) does not measure linear relationships specifically, and it might be more appropriate for certain cases of associating two variables.


# Correlation Coefficients & Regression Analysis

## OCR Result:
Correlation coefficients provide a measure of the **strength** and **direction** of a **linear** relationship.

### Correlation Ranges:
- **Strong**: 0.7 ≤ |r| ≤ 1.0
- **Moderate**: 0.3 ≤ |r| < 0.7
- **Weak**: 0.0 ≤ |r| < 0.3

### Calculation:
r = Σ(x₁-x̄)(y₁-ȳ) / √[Σ(x₁-x̄)²][Σ(y₁-ȳ)²]

In Excel: CORREL(col1, col2)

## Extended Explanation

### Types of Correlation
1. **Pearson's Correlation (r)**
   - Measures linear relationships
   - Most commonly used
   - Values range from -1 to +1
   - Assumes normal distribution

2. **Spearman's Correlation (ρ)**
   - Non-parametric measure
   - Measures monotonic relationships
   - Better for non-linear relationships
   - Uses ranked data

### Interpretation
- **Direction**:
  - Positive: As one variable increases, the other increases
  - Negative: As one variable increases, the other decreases

- **Visualization**:
  - Scatter plots best show relationship
  - Points close to line indicate strong correlation
  - Spread points indicate weak correlation

### Important Notes
1. Correlation ≠ causation
2. Only measures linear relationships
3. Sensitive to outliers
4. Should be used with scatter plots
5. Sample size affects reliability

### Applications
- Market analysis
- Scientific research
- Quality control
- Risk assessment
- Financial modeling


A line is commonly identified by an intercept and a slope.

The intercept is defined as the predicted value of the response when the x-variable is zero.

The slope is defined as the predicted change in the response for every one unit increase in the x-variable.



## Linear Regression Notation

We notate the line in linear regression in the following way:

ŷ = b₀ + b₁x₁

where:

- **ŷ** (y-hat) is the predicted value of the response from the line
- **b₀** is the intercept
- **b₁** is the slope
- **x₁** is the explanatory variable

Note: The hat in ŷ indicates that this is a predicted value from the fitted line, not the real value. We use y (without the hat) to denote the actual response value for a data point in our dataset.


The main algorithm used to find the best fit line is called the least-squares algorithm, which finds the line that minimizes Σ(yi - ŷi)².

There are other ways we might choose a "best" line, but this algorithm tends to do a good job in many scenarios.

It turns out that in order to minimize this function, we have set equations that provide the intercept and slope that should be used.

If you have a set of points like the values in the image here:

X     Y
10    8.04
8     6.95
13    7.58
9     8.81
11    8.33
14    9.96
6     7.24
4     4.26
12    10.84
7     4.82
5     5.68

In order to compute the slope and intercept, we need to compute the following:

   x̄ = (1/n)∑xᵢ 

   ȳ = (1/n)∑yᵢ

   sᵧ = √[(1/(n-1))∑(yᵢ - ȳ)²] (Using the Bessel's Correction formula)

   sₓ = √[(1/(n-1))∑(xᵢ - x̄)²] (Using the Bessel's Correction formula)

   r = ∑(xᵢ-x̄)(yᵢ-ȳ)/√[∑(xᵢ-x̄)²]√[∑(yᵢ-ȳ)²]

   b₁ = r(sᵧ/sₓ)

   b₀ = ȳ - b₁x̄



We can perform hypothesis tests for the coefficients in our linear models using Python (and other software). These tests help us determine if there is a statistically significant linear relationship between a particular variable and the response. The hypothesis test for the intercept isn't useful in most cases.

However, the hypothesis test for each x-variable is a test of if that population slope is equal to zero vs. an alternative where the parameter differs from zero. Therefore, if the slope is different than zero (the alternative is true), we have evidence that the x-variable attached to that coefficient has a statistically significant linear relationship with the response. This in turn suggests that the x-variable should help us in predicting the response (or at least be better than not having it in the model).



The R-squared value is the square of the correlation coefficient.

A common definition for the R-squared variable is that it is the amount of variability in the response variable that can be explained by the x-variable in our model. In general, the closer this value is to 1, the better our model fits the data.

Many feel that R-squared isn't a great measure (which is possibly true), but I would argue that using cross-validation can assist us with validating any measure that helps us understand the fit of a model to our data. Here(opens in a new tab), you can find one such argument explaining why one individual doesn't care for R-squared.


