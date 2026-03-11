import pandas as pd
import numpy as np
from scipy import stats

def get_plot_explanation(plot_type, df, *args):
    """Get educational explanation for a plot"""
    
    explanations = {
        'histogram': f"""
        ### 📚 What This Histogram Shows:
        A histogram displays the **distribution** of a numeric variable. It shows how frequently 
        different ranges of values occur in your data.
        
        ### 🎯 Why It's Useful:
        - **Identify distribution shape**: Is it bell-shaped (normal), skewed, or bimodal?
        - **Spot outliers**: Unusual values appear as isolated bars
        - **Understand data spread**: How concentrated or spread out is the data?
        
        ### 🔍 What to Look For:
        - **Peak/Mode**: The most common value range
        - **Skewness**: Is the distribution leaning left or right?
        - **Spread**: Is data tight or scattered?
        - **Gaps**: Unusual patterns or missing ranges
        
        ### 💡 Example Interpretation:
        If you see a histogram with a peak on the left and a long tail on the right, 
        your data is **right-skewed** (most values are low, with some high outliers).
        """,
        
        'boxplot': f"""
        ### 📚 What This Box Plot Shows:
        A box plot summarizes the **distribution** using quartiles. It shows the median, 
        quartiles, and outliers in a compact format.
        
        ### 🎯 Components:
        - **Line inside box (Median)**: 50% of data is above/below this
        - **Box**: Contains middle 50% of data (Q1 to Q3)
        - **Whiskers**: Lines extending to min/max non-outlier values
        - **Dots**: Individual outliers
        
        ### 🔍 What to Look For:
        - **Box position**: Is median toward top or bottom?
        - **Box size**: Larger box = more variation in middle 50%
        - **Whiskers**: Long whiskers indicate spread data
        - **Outliers**: Dots beyond whiskers are potential anomalies
        
        ### 💡 Comparison Tip:
        Compare multiple box plots side-by-side to see which group has higher/lower values 
        and which has more variability.
        """,
        
        'violin': f"""
        ### 📚 What This Violin Plot Shows:
        A violin plot combines a box plot with a kernel density plot, showing both 
        **distribution shape** and **summary statistics**.
        
        ### 🎯 Why It's Better Than Box Plot:
        - Shows the full **distribution shape**, not just quartiles
        - Reveals **multimodal distributions** (multiple peaks)
        - Better for comparing distributions across groups
        
        ### 🔍 What to Look For:
        - **Width**: Wider sections = more data points at that value
        - **Shape**: Tall and thin = concentrated data; wide = spread out
        - **Multiple peaks**: Shows if data has multiple common values
        
        ### 💡 Example:
        If you see two peaks in a violin plot, your data might have two distinct groups 
        or populations mixed together.
        """,
        
        'scatter': f"""
        ### 📚 What This Scatter Plot Shows:
        A scatter plot displays the **relationship** between two numeric variables. 
        Each point represents one observation.
        
        ### 🎯 Why It's Useful:
        - **Identify correlations**: Do variables move together?
        - **Spot patterns**: Is there a linear, curved, or random relationship?
        - **Find clusters**: Are there distinct groups in your data?
        - **Detect outliers**: Isolated points stand out
        
        ### 🔍 Relationship Types:
        - **Strong positive**: Points trend up-right (both increase together)
        - **Strong negative**: Points trend down-right (one increases as other decreases)
        - **No correlation**: Points scattered randomly
        - **Non-linear**: Points follow a curve, not a straight line
        
        ### 💡 Actionable Insights:
        If you find a strong correlation, you might use one variable to predict the other. 
        If no correlation exists, they might be independent.
        """,
        
        'bar_chart': f"""
        ### 📚 What This Bar Chart Shows:
        A bar chart displays the **frequency** or **count** of each category. 
        The height of each bar represents how common that category is.
        
        ### 🎯 Why It's Useful:
        - **Compare categories**: Which is most/least common?
        - **Identify imbalances**: Is data evenly distributed across categories?
        - **Spot rare categories**: Some categories might be very rare
        
        ### 🔍 What to Look For:
        - **Tallest bar**: Most frequent category
        - **Height differences**: How uneven is the distribution?
        - **Very short bars**: Rare categories (might need special handling)
        - **Missing bars**: Some expected categories might be absent
        
        ### 💡 Preprocessing Insight:
        If you see very small bars, consider grouping rare categories into an "Other" category
        before feeding data to a machine learning model.
        """,
        
        'pie_chart': f"""
        ### 📚 What This Pie Chart Shows:
        A pie chart displays **proportions** or **percentages** of each category as 
        slices of a circle.
        
        ### 🎯 When to Use:
        - Compare a **few categories** (typically 2-5)
        - Show **parts of a whole**
        - Emphasize **percentages**
        
        ### 🔍 What to Look For:
        - **Slice size**: Larger slices = higher proportion
        - **Balance**: Are slices roughly equal or very different?
        - **Dominant category**: Does one category overwhelm others?
        
        ### ⚠️ Limitation:
        Pie charts are harder to read than bar charts for many categories. 
        Use bar charts for more than 5 categories!
        """,
        
        'qq_plot': f"""
        ### 📚 What This Q-Q Plot Shows:
        A Q-Q (Quantile-Quantile) plot compares your data distribution against a 
        **theoretical normal distribution**.
        
        ### 🎯 Why It's Useful:
        - **Test normality**: Is your data normally distributed?
        - **Identify deviations**: Where does data deviate from normal?
        - **Guide transformations**: Do you need to transform your data?
        
        ### 🔍 How to Interpret:
        - **Points on diagonal line**: Data is normally distributed ✓
        - **Curve at start**: Left tail is heavier than normal (left-skewed)
        - **Curve at end**: Right tail is heavier than normal (right-skewed)
        - **S-shape**: Data has different tails than normal distribution
        
        ### 💡 Next Steps:
        If data is not normal, you might:
        - Apply log/square-root transformation
        - Use non-parametric statistical tests
        - Be careful with methods assuming normality
        """,
        
        'heatmap': f"""
        ### 📚 What This Heatmap Shows:
        A correlation heatmap displays the **correlation coefficient** between all pairs 
        of numeric variables as colors.
        
        ### 🎯 Why It's Useful:
        - **Quickly identify relationships**: Red = strong positive, Blue = strong negative
        - **Find multicollinearity**: Highly correlated features might be redundant
        - **Understand feature relationships**: See overall correlation structure
        
        ### 🔍 Color Interpretation:
        - **Dark Red (1.0)**: Perfect positive correlation
        - **Light/White (0)**: No correlation
        - **Dark Blue (-1.0)**: Perfect negative correlation
        
        ### 💡 Action Items:
        If two features are highly correlated (|r| > 0.8):
        - Consider keeping only one for your model
        - They provide redundant information
        - Can improve model interpretability
        """,
        
        'grouped_bar': f"""
        ### 📚 What This Grouped Bar Chart Shows:
        A grouped bar chart compares a **numeric variable** across different **categories**, 
        with multiple groups per category.
        
        ### 🎯 Why It's Useful:
        - **Compare across groups**: Different categories have different distributions?
        - **Identify patterns**: Are relationships consistent across groups?
        - **Spot interactions**: Does one variable's effect depend on another?
        
        ### 🔍 What to Look For:
        - **Height differences within categories**: Variation across groups
        - **Height differences across categories**: Overall trend
        - **Parallel patterns**: Do all groups follow the same trend?
        - **Crossing lines**: Interaction effects (relationship changes by group)
        
        ### 💡 Example:
        If sales by product type differ significantly by region, grouping helps you 
        see this pattern at a glance.
        """,
        
        'line_plot': f"""
        ### 📚 What This Line Plot Shows:
        A line plot shows how a **numeric variable changes** in relation to another variable, 
        typically over time or ordered categories.
        
        ### 🎯 Why It's Useful:
        - **Identify trends**: Is the variable increasing, decreasing, or stable?
        - **Spot seasonality**: Are there regular patterns?
        - **Detect changes**: Where do abrupt changes occur?
        
        ### 🔍 What to Look For:
        - **Upward trend**: Variable is generally increasing
        - **Downward trend**: Variable is generally decreasing
        - **Peaks and valleys**: Cyclic or seasonal patterns
        - **Plateaus**: Periods of stability
        - **Sharp jumps**: Sudden changes worth investigating
        
        ### 💡 Preprocessing Tip:
        If you see a trend, you might need to:
        - Normalize the data
        - Detrend for stationary analysis
        - Include time-based features
        """,
        
        'distribution_fit': f"""
        ### 📚 What This Distribution Fit Shows:
        This plot overlays a **theoretical distribution** (usually normal) on top of your 
        actual **data histogram**.
        
        ### 🎯 Why It's Useful:
        - **Goodness of fit**: How well does the theoretical fit match actual data?
        - **Identify appropriate distributions**: Normal? Exponential? Other?
        - **Guide transformations**: If fit is poor, consider transforming data
        
        ### 🔍 What to Look For:
        - **Close match**: Distribution fits your data well
        - **Tails deviation**: Data tails are heavier/lighter than theoretical
        - **Peak height**: Mode matches or differs from theoretical
        
        ### 💡 When to Transform:
        If the fit is poor, try:
        - Log transformation (for right-skewed data)
        - Square root transformation (moderate skewness)
        - Box-Cox transformation (optimal automatic transformation)
        """
    }
    
    return explanations.get(plot_type, "No explanation available for this plot type.")

def get_preprocessing_explanation(category, column_name, method):
    """Get educational explanations for preprocessing steps"""
    
    explanations = {
        'missing': {
            'Mean': f"""
            ### Filling Missing Values with Mean
            
            **What it does:** Replaces missing values with the **average** of all non-missing values in {column_name}.
            
            **When to use:**
            - Numeric columns with random missing values
            - Small percentage of missing data (<5%)
            - Data is approximately normally distributed
            
            **Pros:**
            - Simple and fast
            - Preserves sample size
            - Maintains mean of the original data
            
            **Cons:**
            - Reduces variance
            - Artificial values may distort relationships
            - Not ideal if data is skewed
            
            **Example:** If {column_name} values are [10, 20, 30, NaN], mean = 20, so NaN → 20
            """,
            
            'Median': f"""
            ### Filling Missing Values with Median
            
            **What it does:** Replaces missing values with the **middle value** of {column_name} when sorted.
            
            **When to use:**
            - Numeric columns with outliers
            - Skewed data
            - Small to moderate missing rates
            
            **Pros:**
            - More robust to outliers than mean
            - Better for skewed distributions
            - Maintains data distribution shape
            
            **Cons:**
            - Slightly less "correct" statistically
            - Still artificial
            
            **Best for:** Data with extreme values or skewed distributions
            """,
            
            'Mode': f"""
            ### Filling Missing Values with Mode
            
            **What it does:** Replaces missing values with the **most frequent value** in {column_name}.
            
            **When to use:**
            - Categorical columns
            - Numeric columns with discrete values
            - When the most common value makes practical sense
            
            **Pros:**
            - Preserves actual data values
            - Good for categories
            - Makes practical sense
            
            **Cons:**
            - May bias toward the most common value
            - Loses information about the missing pattern
            
            **Example:** If {column_name} is [A, B, B, B, C, NaN], mode = B, so NaN → B
            """,
            
            'Forward Fill': f"""
            ### Forward Fill (Propagate Forward)
            
            **What it does:** Fills missing values by copying the **previous value** forward.
            
            **When to use:**
            - Time series data
            - Sequential data (ordered by time or position)
            - Assumes recent past is predictor of future
            
            **Pros:**
            - Perfect for time series
            - Preserves trends
            - Simple logic
            
            **Cons:**
            - Only works if data is properly ordered
            - Can propagate old values far into future
            - Not suitable for random missing data
            
            **Example:** [10, 20, NaN, NaN, 50] → [10, 20, 20, 20, 50]
            """,
            
            'Drop Rows': f"""
            ### Drop Rows with Missing Values
            
            **When to use:**
            - Small amount of data is missing
            - Missing completely at random
            - Can afford to lose observations
            - Missing rate < 5%
            
            **Pros:**
            - No artificial values introduced
            - Clean, honest dataset
            - No assumption made about missing data
            
            **Cons:**
            - Reduces dataset size
            - May lose important information
            - Can bias results if data not missing randomly
            
            **Note:** Generally a last resort. Try other methods first!
            """
        },
        
        'outliers': {
            'IQR Method': f"""
            ### IQR Method for Outliers
            
            **What it does:** Identifies values that fall outside 1.5 × IQR from quartiles.
            
            **Formula:**
            - Q1 = 25th percentile
            - Q3 = 75th percentile
            - IQR = Q3 - Q1
            - Lower bound = Q1 - 1.5 × IQR
            - Upper bound = Q3 + 1.5 × IQR
            
            **When to use:**
            - Robust to distribution shape
            - Non-parametric (no normal distribution assumption)
            - Industry standard
            
            **Action after detection:**
            - Review: Are they true outliers or data entry errors?
            - Cap: Replace with upper/lower bound
            - Remove: Delete if confirmed errors
            - Keep: Leave as is if legitimate extreme values
            """,
            
            'Z-Score': f"""
            ### Z-Score Method for Outliers
            
            **What it does:** Identifies values more than 3 standard deviations from mean.
            
            **Formula:**
            - Z-score = (value - mean) / standard_deviation
            - Usually flag if |Z-score| > 3
            
            **When to use:**
            - Data is approximately normally distributed
            - Want mathematical rigor
            
            **Pros:**
            - Statistically principled
            - Works well for normal distributions
            
            **Cons:**
            - Sensitive to outliers (outliers can inflate std dev)
            - Assumes normality
            """,
            
            'Capping': f"""
            ### Capping/Winsorization
            
            **What it does:** Replaces extreme values with a threshold (usually 95th or 99th percentile).
            
            **Example:** Cap values at 99th percentile
            - Values above 99th percentile → 99th percentile value
            - Values below 1st percentile → 1st percentile value
            
            **Pros:**
            - Preserves all observations
            - Reduces extreme influence without removal
            - Commonly used in practice
            
            **Cons:**
            - Creates artificial identical values at cap
            - May lose information about extreme values
            
            **When to use:**
            - Can't afford to lose data
            - Need to reduce outlier influence but keep observations
            """,
            
            'Keep As Is': f"""
            ### Keep Outliers As Is
            
            **When to use:**
            - Outliers are genuine, legitimate values
            - Need to preserve all information
            - Building robust models
            - Outliers are scientifically interesting
            
            **Note:** Some machine learning algorithms handle outliers well:
            - Tree-based models (Random Forest, XGBoost)
            - Robust regression methods
            - Non-parametric methods
            """
        },
        
        'scaling': {
            'StandardScaler': f"""
            ### StandardScaler (Z-score Normalization)
            
            **What it does:** Transforms data to have mean=0 and standard deviation=1.
            
            **Formula:**
            - X_scaled = (X - mean) / std_deviation
            
            **When to use:**
            - Algorithms sensitive to magnitude (KNN, K-means, SVM)
            - Linear regression, logistic regression
            - Assuming normal distribution is acceptable
            
            **Pros:**
            - Most common scaling method
            - Preserves distribution shape
            - Good for normally distributed data
            
            **Range:** Typically -3 to +3, but unbounded
            
            **Use this when:** You want to standardize to a mean of 0 and std of 1
            """,
            
            'MinMaxScaler': f"""
            ### MinMaxScaler (Normalization to 0-1)
            
            **What it does:** Scales data to a fixed range [0, 1].
            
            **Formula:**
            - X_scaled = (X - min) / (max - min)
            
            **When to use:**
            - Need bounded output (0 to 1)
            - Neural networks
            - Algorithms requiring 0-1 range
            - Data with known min/max
            
            **Pros:**
            - Bounded output
            - Intuitive interpretation
            - Preserves the distribution shape
            
            **Cons:**
            - Sensitive to outliers (they become min/max)
            - Bounded scale may hurt some algorithms
            
            **Range:** Always between 0 and 1
            """,
            
            'RobustScaler': f"""
            ### RobustScaler (Robust to Outliers)
            
            **What it does:** Scales using median and interquartile range.
            
            **Formula:**
            - X_scaled = (X - median) / IQR
            
            **When to use:**
            - Data has significant outliers
            - Can't remove outliers
            - Need robust scaling
            - Distribution not normal
            
            **Pros:**
            - Not affected by outliers
            - Better than StandardScaler for skewed data
            - Robust scaling
            
            **Best for:** Data with outliers you want to keep
            """,
            
            'No Scaling': f"""
            ### When NOT to Scale
            
            **Keep original scale when:**
            - Using tree-based models (Random Forest, XGBoost, Decision Trees)
            - Interpretability is critical
            - Features already on similar scales
            - Statistical tests assume raw values
            
            **Tree-based models don't care about scale because they use:**
            - Thresholds (value > threshold)
            - Feature ranks, not absolute values
            
            **Note:** When in doubt, scale! Rarely hurts, often helps.
            """
        },
        
        'encoding': {
            'One-Hot Encoding': f"""
            ### One-Hot Encoding
            
            **What it does:** Creates binary (0/1) columns for each category.
            
            **Example:**
            - Original: Color = [Red, Blue, Green, Red]
            - Result:
              - Color_Red = [1, 0, 0, 1]
              - Color_Blue = [0, 1, 0, 0]
              - Color_Green = [0, 0, 1, 0]
            
            **When to use:**
            - Nominal categories (no order)
            - Most machine learning algorithms
            - Linear models that need numeric input
            - Tree-based models often don't need this
            
            **Pros:**
            - Works for all algorithms
            - No ordinal assumption
            - Clear interpretation
            
            **Cons:**
            - Creates many columns (curse of dimensionality)
            - Memory intensive for high cardinality
            
            **Use drop_first=True:** Avoid multicollinearity!
            """,
            
            'Label Encoding': f"""
            ### Label Encoding
            
            **What it does:** Assigns integer labels (0, 1, 2...) to categories.
            
            **Example:**
            - Color = [Red, Blue, Green, Red]
            - Result: [0, 1, 2, 0]
            
            **When to use:**
            - Tree-based models (they handle integers well)
            - Memory constraints (fewer columns than one-hot)
            - Ordinal categories (where order matters)
            
            **Pros:**
            - Memory efficient
            - Simple
            - Works for tree models
            
            **Cons:**
            - Implies ordinal relationship (0 < 1 < 2)
            - Bad for linear models (assumes linear scale)
            - Can mislead algorithms
            
            **⚠️ Be careful:** Only use for ordinal data or tree-based models!
            """,
            
            'Target Encoding': f"""
            ### Target Encoding (Mean Encoding)
            
            **What it does:** Replaces categories with the mean target value for that category.
            
            **Example (Predicting House Price):**
            - City = [New York, LA, NYC, NYC, LA]
            - Target (Price) = [500k, 300k, 450k, 550k, 350k]
            - NYC mean = (500+450+550)/3 = 500
            - LA mean = (300+350)/2 = 325
            - Result: [500, 325, 500, 500, 325]
            
            **When to use:**
            - High cardinality features (many categories)
            - Want to leverage target information
            - Need to reduce dimensionality
            - Supervised learning scenarios
            
            **Pros:**
            - Captures relationship with target
            - Efficient dimensionality reduction
            - Often improves model performance
            
            **Cons:**
            - Can cause overfitting
            - Requires target variable
            - May not generalize well
            
            **Use regularization:** Smooth with global mean to prevent overfitting!
            """,
            
            'Frequency Encoding': f"""
            ### Frequency Encoding
            
            **What it does:** Replaces categories with their frequency (count/proportion).
            
            **Example:**
            - Color = [Red, Red, Blue, Red, Green]
            - Frequencies: Red=3, Blue=1, Green=1
            - Result: [3/5, 3/5, 1/5, 3/5, 1/5] = [0.6, 0.6, 0.2, 0.6, 0.2]
            
            **When to use:**
            - Frequency information is predictive
            - High cardinality features
            - Quick and simple encoding needed
            - Rare categories might be important
            
            **Pros:**
            - Simple and interpretable
            - Preserves frequency information
            - Works for tree models
            - Handles new categories naturally (frequency = 0)
            
            **Cons:**
            - Loses category identity
            - May group different categories with same frequency
            - Limited information retention
            
            **Good for:** When frequency patterns in data matter
            """
        },
        
        'transform': {
            'Log Transform': f"""
            ### Log Transformation
            
            **What it does:** Applies log() to transform data: X_new = log(X)
            
            **When to use:**
            - Right-skewed data (long tail on right)
            - Positive values only
            - Exponential relationships
            - To stabilize variance
            
            **Pros:**
            - Reduces skewness
            - Compresses large values
            - Makes relationships more linear
            - Stabilizes variance (heteroscedasticity fix)
            
            **Cons:**
            - Only for positive values
            - Loses some information
            - Makes interpretation harder
            - Zero values need special handling
            
            **Inverse transformation:** 10^(X_new) = X_original
            """,
            
            'Square Root Transform': f"""
            ### Square Root Transformation
            
            **What it does:** Applies sqrt() to transform data: X_new = sqrt(X)
            
            **When to use:**
            - Moderate right skewness
            - Positive values only
            - Less extreme than log transform
            - Count data
            
            **Pros:**
            - Less aggressive than log transform
            - Easier to interpret than log
            - Works for count data
            - Still handles skewness well
            
            **Cons:**
            - Only for positive values
            - Moderate effect only
            - Still some transformation loss
            
            **Between log and square root:** If moderate skewness, try this first!
            """,
            
            'Box-Cox Transform': f"""
            ### Box-Cox Transformation
            
            **What it does:** Automatically finds optimal λ for power transformation:
            - X_new = (X^λ - 1) / λ (for λ ≠ 0)
            - X_new = log(X) (for λ = 0)
            
            **When to use:**
            - Want automatic optimal transformation
            - Data is positive
            - Don't know which transform to apply
            - Want mathematically optimal solution
            
            **Pros:**
            - Mathematically optimal
            - Automatic parameter selection
            - Very effective for skewness correction
            - Finds best power transformation
            
            **Cons:**
            - Only for positive values
            - Complex to interpret
            - May overfit if λ chosen on training data
            - Computationally expensive
            
            **Best for:** When you want the best automatic transformation
            """
        },
        
        'rare_categories': {
            'Group as Other': f"""
            ### Group Rare Categories as "Other"
            
            **What it does:** Combines rare categories (< threshold) into single "Other" category.
            
            **Example:**
            - Categories: A (50%), B (30%), C (15%), D (3%), E (2%)
            - Threshold: 5%
            - Result: A (50%), B (30%), C (15%), Other (5%)
            
            **When to use:**
            - Too many categories (high cardinality)
            - Some categories very rare
            - Want to simplify model
            - Improve stability
            
            **Pros:**
            - Reduces dimensionality
            - Fewer one-hot columns
            - More stable predictions
            - Avoids sparse features
            
            **Cons:**
            - Loss of information
            - Obscures rare categories
            - May hurt rare category prediction
            
            **Best for:** When rare categories are noise, not signal
            """,
            
            'Keep Separate': f"""
            ### Keep Rare Categories Separate
            
            **When to use:**
            - Rare categories are important
            - Information loss unacceptable
            - Have enough data for stability
            - Rare categories are meaningful
            
            **Pros:**
            - Preserves all information
            - No information loss
            - Captures rare patterns
            
            **Cons:**
            - Sparse features
            - More model parameters
            - May cause overfitting
            - Unstable predictions for rare categories
            
            **Typical approach:** Use with regularization (L1/L2) to handle sparsity
            """,
            
            'Drop Rows': f"""
            ### Drop Rows with Rare Categories
            
            **When to use:**
            - Very few observations with rare category
            - Can afford data loss
            - Rare categories confirmed as errors
            - High cardinality combined with small dataset
            
            **Example:**
            - Category D has only 2 observations in 1000-row dataset
            - Dropping 2 rows loses <1% of data
            
            **Pros:**
            - Clean data
            - Removes noise
            - Simplifies model
            
            **Cons:**
            - Data loss
            - May be biased
            - Loses real, valid data
            
            **Use as last resort:** Only when truly necessary
            """
        }
    }
    
    category_dict = explanations.get(category, {})
    return category_dict.get(method, f"No explanation available for {method} in {category}")