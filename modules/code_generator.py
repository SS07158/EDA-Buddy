def generate_code_snippets(snippet_type, column_name, method):
    """Generate Python code for preprocessing and visualization"""
    
    code_snippets = {
        'missing_numeric': {
            'Mean': f"""import pandas as pd
import numpy as np

# Fill missing values with mean
df['{column_name}'].fillna(df['{column_name}'].mean(), inplace=True)

# Verify
print(f"Missing values: {{df['{column_name}'].isnull().sum()}}")
""",
            'Median': f"""import pandas as pd
import numpy as np

# Fill missing values with median
df['{column_name}'].fillna(df['{column_name}'].median(), inplace=True)

# Verify
print(f"Missing values: {{df['{column_name}'].isnull().sum()}}")
""",
            'Mode': f"""import pandas as pd
import numpy as np

# Fill missing values with mode
df['{column_name}'].fillna(df['{column_name}'].mode()[0], inplace=True)

# Verify
print(f"Missing values: {{df['{column_name}'].isnull().sum()}}")
""",
            'Forward Fill': f"""import pandas as pd

# Forward fill missing values
df['{column_name}'].fillna(method='ffill', inplace=True)

# Fill any remaining with backward fill
df['{column_name}'].fillna(method='bfill', inplace=True)

# Verify
print(f"Missing values: {{df['{column_name}'].isnull().sum()}}")
""",
            'Drop Rows': f"""import pandas as pd

# Drop rows with missing values in {column_name}
df = df.dropna(subset=['{column_name}'])

# Verify
print(f"Dataset shape: {{df.shape}}")
print(f"Missing values: {{df['{column_name}'].isnull().sum()}}")
"""
        },
        
        'missing_categorical': {
            'Mode': f"""import pandas as pd

# Fill with most frequent category
df['{column_name}'].fillna(df['{column_name}'].mode()[0], inplace=True)

# Verify
print(f"Missing values: {{df['{column_name}'].isnull().sum()}}")
""",
            'New Category': f"""import pandas as pd

# Fill with new 'Unknown' category
df['{column_name}'].fillna('Unknown', inplace=True)

# Verify
print(f"Missing values: {{df['{column_name}'].isnull().sum()}}")
print(f"Value counts:")
print(df['{column_name}'].value_counts())
""",
            'Drop Rows': f"""import pandas as pd

# Drop rows with missing values
df = df.dropna(subset=['{column_name}'])

# Verify
print(f"Dataset shape: {{df.shape}}")
print(f"Missing values: {{df['{column_name}'].isnull().sum()}}")
"""
        },
        
        'outliers_numeric': {
            'IQR Method': f"""import pandas as pd
import numpy as np

# IQR method for outlier detection
Q1 = df['{column_name}'].quantile(0.25)
Q3 = df['{column_name}'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['{column_name}'] < lower_bound) | (df['{column_name}'] > upper_bound)]
print(f"Found {{len(outliers)}} outliers")

# Remove outliers
df = df[(df['{column_name}'] >= lower_bound) & (df['{column_name}'] <= upper_bound)]
print(f"Dataset shape after removing outliers: {{df.shape}}")
""",
            'Z-Score': f"""import pandas as pd
import numpy as np
from scipy import stats

# Z-score method
z_scores = np.abs(stats.zscore(df['{column_name}'].dropna()))

# Identify outliers (|Z| > 3)
outliers_mask = z_scores > 3

# Remove outliers
df = df[~df.index.isin(df[outliers_mask].index)]
print(f"Removed {{outliers_mask.sum()}} outliers")
print(f"Dataset shape: {{df.shape}}")
""",
            'Capping': f"""import pandas as pd

# Cap at 99th and 1st percentiles
lower_cap = df['{column_name}'].quantile(0.01)
upper_cap = df['{column_name}'].quantile(0.99)

df['{column_name}'] = df['{column_name}'].clip(lower_cap, upper_cap)

print(f"Capped values between {{lower_cap}} and {{upper_cap}}")
""",
            'Keep As Is': f"""# Outliers preserved as-is
# Tree-based models handle outliers well:
# - Random Forest, XGBoost, LightGBM
# - They split on thresholds, not magnitudes

# If using linear models:
# - StandardScaler helps reduce outlier influence
# - Consider RobustScaler for more robust handling

print("Outliers kept for robust model handling")
"""
        },
        
        'scaling': {
            'StandardScaler': f"""import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit and transform
df['{column_name}_scaled'] = scaler.fit_transform(df[['{column_name}']])

# View results
print(f"Original - Mean: {{df['{column_name}'].mean():.2f}}, Std: {{df['{column_name}'].std():.2f}}")
print(f"Scaled - Mean: {{df['{column_name}_scaled'].mean():.2f}}, Std: {{df['{column_name}_scaled'].std():.2f}}")

# Save scaler for future use
import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb'))
""",
            'MinMaxScaler': f"""import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scaler = MinMaxScaler()

# Fit and transform
df['{column_name}_scaled'] = scaler.fit_transform(df[['{column_name}']])

# View results
print(f"Original range: [{{df['{column_name}'].min()}}, {{df['{column_name}'].max()}}]")
print(f"Scaled range: [{{df['{column_name}_scaled'].min()}}, {{df['{column_name}_scaled'].max()}}]")

# Save scaler for future use
import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb'))
""",
            'RobustScaler': f"""import pandas as pd
from sklearn.preprocessing import RobustScaler

# Initialize scaler (robust to outliers)
scaler = RobustScaler()

# Fit and transform
df['{column_name}_scaled'] = scaler.fit_transform(df[['{column_name}']])

# View results
print(f"Original median: {{df['{column_name}'].median():.2f}}")
print(f"Scaled median: {{df['{column_name}_scaled'].median():.2f}}")

# Save scaler for future use
import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb'))
""",
            'No Scaling': f"""# No scaling applied
# Suitable for:
# - Tree-based models (Random Forest, XGBoost)
# - When interpretability is critical
# - When features already on similar scales

print("No scaling applied - data remains in original scale")
"""
        },
        
        'encoding': {
            'One-Hot Encoding': f"""import pandas as pd

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['{column_name}'], drop_first=True)

print(f"Original columns: {{df.shape[1]}}")
print(f"After encoding: {{df_encoded.shape[1]}}")
print(f"New columns created:")
print([col for col in df_encoded.columns if '{column_name}' in col])

# Drop original categorical column
df_encoded = df_encoded.drop('{column_name}', axis=1)
""",
            'Label Encoding': f"""import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Label encoding
le = LabelEncoder()
df['{column_name}_encoded'] = le.fit_transform(df['{column_name}'])

# Display mapping
print("Encoding mapping:")
for i, class_ in enumerate(le.classes_):
    print(f"  {{class_}} -> {{i}}")

# Save encoder for future use
import pickle
pickle.dump(le, open('label_encoder.pkl', 'wb'))
""",
            'Target Encoding': f"""import pandas as pd

# Target encoding (requires target variable)
# Example: predicting price by city

# Calculate mean target value per category
target_encoding_map = df.groupby('{column_name}')['target'].mean().to_dict()

# Apply encoding
df['{column_name}_encoded'] = df['{column_name}'].map(target_encoding_map)

# Handle unseen categories
df['{column_name}_encoded'].fillna(df['target'].mean(), inplace=True)

print("Target encoding mapping:")
print(target_encoding_map)

# Save mapping for future use
import pickle
pickle.dump(target_encoding_map, open('target_encoding.pkl', 'wb'))
""",
            'Frequency Encoding': f"""import pandas as pd

# Frequency encoding
frequency_map = df['{column_name}'].value_counts(normalize=True).to_dict()

df['{column_name}_encoded'] = df['{column_name}'].map(frequency_map)

print("Frequency encoding mapping:")
print(frequency_map)

# Save mapping for future use
import pickle
pickle.dump(frequency_map, open('frequency_encoding.pkl', 'wb'))
"""
        },
        
        'transform': {
            'Log Transform': f"""import pandas as pd
import numpy as np

# Log transformation
# Note: Only for positive values

if (df['{column_name}'] <= 0).any():
    print("Warning: Column contains non-positive values!")
    print("Adding small constant before log transformation...")
    df['{column_name}_log'] = np.log(df['{column_name}'] + 1)
else:
    df['{column_name}_log'] = np.log(df['{column_name}'])

# Compare skewness
print(f"Original skewness: {{df['{column_name}'].skew():.3f}}")
print(f"Log-transformed skewness: {{df['{column_name}_log'].skew():.3f}}")

# Inverse transformation (if needed): np.exp(df['{column_name}_log'])
""",
            'Square Root': f"""import pandas as pd
import numpy as np

# Square root transformation
# Note: Only for non-negative values

if (df['{column_name}'] < 0).any():
    print("Warning: Column contains negative values!")
else:
    df['{column_name}_sqrt'] = np.sqrt(df['{column_name}'])

# Compare skewness
print(f"Original skewness: {{df['{column_name}'].skew():.3f}}")
print(f"Sqrt-transformed skewness: {{df['{column_name}_sqrt'].skew():.3f}}")

# Inverse transformation (if needed): df['{column_name}_sqrt'] ** 2
""",
            'Box-Cox': f"""import pandas as pd
import numpy as np
from scipy.stats import boxcox

# Box-Cox transformation
# Note: All values must be positive

if (df['{column_name}'] <= 0).any():
    print("Adding constant to make all values positive...")
    data = df['{column_name}'] + abs(df['{column_name}'].min()) + 1
else:
    data = df['{column_name}']

# Apply Box-Cox
df['{column_name}_boxcox'], lambda_param = boxcox(data)

print(f"Optimal lambda: {{lambda_param:.3f}}")
print(f"Original skewness: {{df['{column_name}'].skew():.3f}}")
print(f"Box-Cox skewness: {{df['{column_name}_boxcox'].skew():.3f}}")
"""
        },
        
        'rare_categories': {
            'Group as Other': f"""import pandas as pd

# Identify rare categories (< 5% threshold)
threshold = 0.05
value_counts = df['{column_name}'].value_counts()
rare_categories = value_counts[value_counts / len(df) < threshold].index.tolist()

print(f"Rare categories: {{rare_categories}}")

# Replace rare with 'Other'
df['{column_name}'] = df['{column_name}'].apply(
    lambda x: 'Other' if x in rare_categories else x
)

print(f"Updated value counts:")
print(df['{column_name}'].value_counts())
""",
            'Keep Separate': f"""import pandas as pd

# Keep all categories
# Use regularization (L1/L2) to handle sparsity in your model

print(f"Unique categories: {{df['{column_name}'].nunique()}}")
print(f"Value counts:")
print(df['{column_name}'].value_counts())

# When encoding, all categories will be preserved
# Consider using:
# - L1 regularization (Lasso) for feature selection
# - L2 regularization (Ridge) to reduce overfitting
""",
            'Drop Rows': f"""import pandas as pd

# Identify rare categories (< 5% threshold)
threshold = 0.05
value_counts = df['{column_name}'].value_counts()
rare_categories = value_counts[value_counts / len(df) < threshold].index.tolist()

# Drop rows with rare categories
df = df[~df['{column_name}'].isin(rare_categories)]

print(f"Removed rows with rare categories")
print(f"Dataset shape: {{df.shape}}")
print(f"Remaining categories:")
print(df['{column_name}'].value_counts())
"""
        }
    }
    
    # Navigate nested dictionary
    if snippet_type in code_snippets:
        return code_snippets[snippet_type].get(method, f"Code not available for {method}")
    else:
        return f"Code snippet for {snippet_type} not available"