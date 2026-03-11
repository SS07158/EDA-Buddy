import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from scipy.stats import boxcox

def suggest_preprocessing(df, column):
    """Suggest preprocessing steps for a column"""
    
    suggestions = []
    
    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric column suggestions
        missing_pct = df[column].isnull().sum() / len(df) * 100
        if missing_pct > 0:
            suggestions.append({
                'type': 'Missing Values',
                'severity': 'high' if missing_pct > 30 else 'medium' if missing_pct > 5 else 'low',
                'recommendation': f"This column has {missing_pct:.1f}% missing values. Consider using mean/median imputation or dropping rows."
            })
        
        # Check skewness
        skewness = df[column].skew()
        if abs(skewness) > 1:
            suggestions.append({
                'type': 'Skewness',
                'severity': 'medium',
                'recommendation': f"Skewness is {skewness:.2f}. Consider log or Box-Cox transformation."
            })
        
        # Check outliers
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))).sum()
        
        if outliers > 0:
            outlier_pct = outliers / len(df) * 100
            suggestions.append({
                'type': 'Outliers',
                'severity': 'high' if outlier_pct > 5 else 'medium',
                'recommendation': f"Found {outliers} outliers ({outlier_pct:.1f}%). Consider IQR method or capping."
            })
        
        # Scaling suggestion
        suggestions.append({
            'type': 'Scaling',
            'severity': 'info',
            'recommendation': "Use StandardScaler for most algorithms, MinMaxScaler if output bounded needed."
        })
    
    else:
        # Categorical column suggestions
        missing_pct = df[column].isnull().sum() / len(df) * 100
        if missing_pct > 0:
            suggestions.append({
                'type': 'Missing Values',
                'severity': 'high' if missing_pct > 30 else 'medium' if missing_pct > 5 else 'low',
                'recommendation': f"This column has {missing_pct:.1f}% missing values. Fill with mode or 'Unknown' category."
            })
        
        # Check cardinality
        unique_count = df[column].nunique()
        if unique_count > 20:
            suggestions.append({
                'type': 'High Cardinality',
                'severity': 'medium',
                'recommendation': f"This column has {unique_count} unique values. Consider target encoding or frequency encoding."
            })
        
        # Check rare categories
        value_counts = df[column].value_counts()
        rare_pct = (value_counts / len(df) * 100)
        rare_cats = (rare_pct < 5).sum()
        
        if rare_cats > 0:
            suggestions.append({
                'type': 'Rare Categories',
                'severity': 'low',
                'recommendation': f"Found {rare_cats} categories with < 5% frequency. Consider grouping as 'Other'."
            })
        
        # Encoding suggestion
        if unique_count <= 2:
            suggestions.append({
                'type': 'Encoding',
                'severity': 'info',
                'recommendation': "Binary column: Use label encoding or one-hot encoding."
            })
        else:
            suggestions.append({
                'type': 'Encoding',
                'severity': 'info',
                'recommendation': "Use one-hot encoding for nominal categories or label encoding for tree-based models."
            })
    
    return suggestions