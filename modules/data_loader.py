import pandas as pd
import numpy as np
from datetime import datetime
import io

class DataLoader:
    def __init__(self):
        self.df = None
        self.file_info = {}
    
    def load_csv(self, file_path):
        """Load CSV file"""
        try:
            self.df = pd.read_csv(file_path)
            self.file_info['source'] = file_path
            self.file_info['format'] = 'csv'
            self.file_info['loaded_at'] = datetime.now()
            return True, "CSV loaded successfully"
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}"
    
    def load_excel(self, file_path):
        """Load Excel file"""
        try:
            self.df = pd.read_excel(file_path)
            self.file_info['source'] = file_path
            self.file_info['format'] = 'excel'
            self.file_info['loaded_at'] = datetime.now()
            return True, "Excel file loaded successfully"
        except Exception as e:
            return False, f"Error loading Excel: {str(e)}"
    
    def get_data_info(self):
        """Get basic information about the dataset"""
        if self.df is None:
            return None
        
        return {
            'rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'column_names': self.df.columns.tolist(),
            'data_types': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,
            'file_info': self.file_info
        }

def load_data(uploaded_file):
    """Wrapper function for Streamlit"""
    if uploaded_file is None:
        return None
    
    loader = DataLoader()
    
    if uploaded_file.name.endswith('.csv'):
        success, message = loader.load_csv(uploaded_file)
    else:
        success, message = loader.load_excel(uploaded_file)
    
    if success:
        return loader.df
    else:
        raise Exception(message)