import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.analysis = {}
    
    def get_column_types(self):
        """Identify column types"""
        types = {}
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                types[col] = 'numeric'
            elif pd.api.types.is_categorical_dtype(self.df[col]):
                types[col] = 'categorical'
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                types[col] = 'datetime'
            else:
                types[col] = 'categorical'
        return types
    
    def get_missing_analysis(self):
        """Analyze missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        return pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        }).sort_values('Missing_Count', ascending=False)
    
    def get_numeric_stats(self, col):
        """Get statistics for numeric column"""
        return {
            'count': self.df[col].count(),
            'mean': self.df[col].mean(),
            'std': self.df[col].std(),
            'min': self.df[col].min(),
            '25%': self.df[col].quantile(0.25),
            '50%': self.df[col].quantile(0.50),
            '75%': self.df[col].quantile(0.75),
            'max': self.df[col].max(),
            'skewness': self.df[col].skew(),
            'kurtosis': self.df[col].kurtosis()
        }
    
    def get_categorical_stats(self, col):
        """Get statistics for categorical column"""
        value_counts = self.df[col].value_counts()
        return {
            'unique': self.df[col].nunique(),
            'most_common': value_counts.index[0],
            'most_common_freq': value_counts.values[0],
            'value_counts': value_counts.to_dict()
        }
    
    def get_correlation_matrix(self):
        """Get correlation matrix for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr()
    
    def get_high_correlations(self, threshold=0.7):
        """Find high correlations"""
        corr = self.get_correlation_matrix()
        high_corr = []
        
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > threshold:
                    high_corr.append({
                        'var1': corr.columns[i],
                        'var2': corr.columns[j],
                        'correlation': corr.iloc[i, j]
                    })
        
        return pd.DataFrame(high_corr) if high_corr else pd.DataFrame()
    
    def get_outliers(self, col):
        """Identify outliers using IQR method"""
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': len(outliers) / len(self.df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_indices': outliers.index.tolist()
        }

def analyze_dataset(df):
    """Wrapper function for analysis"""
    analyzer = DataAnalyzer(df)
    
    return {
        'column_types': analyzer.get_column_types(),
        'missing_analysis': analyzer.get_missing_analysis(),
        'numeric_stats': {col: analyzer.get_numeric_stats(col) 
                         for col in df.select_dtypes(include=[np.number]).columns},
        'categorical_stats': {col: analyzer.get_categorical_stats(col) 
                             for col in df.select_dtypes(include=['object', 'category']).columns},
        'correlations': analyzer.get_correlation_matrix(),
        'high_correlations': analyzer.get_high_correlations()
    }