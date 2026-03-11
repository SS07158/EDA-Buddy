import pandas as pd
import numpy as np
from datetime import datetime
import io
import chardet

class DataLoader:
    """
    Handles file loading with robust encoding detection and error handling
    """
    
    # Common encodings to try in order
    ENCODING_FALLBACKS = [
        'utf-8',
        'utf-8-sig',  # UTF-8 with BOM
        'latin-1',    # ISO-8859-1 (Western European)
        'cp1252',     # Windows Western European
        'iso-8859-1',
        'gb2312',     # Chinese
        'gbk',        # Chinese
        'big5',       # Traditional Chinese
        'shift_jis',  # Japanese
        'euc-kr',     # Korean
        'utf-16',
        'ascii'
    ]
    
    def __init__(self):
        self.df = None
        self.file_info = {}
        self.detected_encoding = None
    
    def detect_encoding(self, file_path_or_bytes):
        """
        Detect file encoding using chardet library
        Returns: encoding name as string
        """
        try:
            # Read sample of file
            if isinstance(file_path_or_bytes, bytes):
                sample = file_path_or_bytes[:10000]  # First 10KB
            else:
                with open(file_path_or_bytes, 'rb') as f:
                    sample = f.read(10000)
            
            # Detect encoding
            detection = chardet.detect(sample)
            encoding = detection.get('encoding')
            confidence = detection.get('confidence', 0)
            
            print(f"🔍 Detected encoding: {encoding} (confidence: {confidence:.2%})")
            
            return encoding if confidence > 0.7 else 'utf-8'
        except Exception as e:
            print(f"⚠️ Encoding detection failed: {e}")
            return 'utf-8'
    
    def load_csv_with_fallback(self, file_path_or_bytes, filename=""):
        """
        Load CSV with automatic encoding detection and fallback
        """
        errors = []
        
        # Try detected encoding first
        detected = self.detect_encoding(file_path_or_bytes)
        encoding_list = [detected] + [enc for enc in self.ENCODING_FALLBACKS if enc != detected]
        
        for encoding in encoding_list:
            try:
                if isinstance(file_path_or_bytes, bytes):
                    self.df = pd.read_csv(
                        io.BytesIO(file_path_or_bytes),
                        encoding=encoding,
                        on_bad_lines='skip',  # Skip lines with errors
                        engine='python'  # More flexible parser
                    )
                else:
                    self.df = pd.read_csv(
                        file_path_or_bytes,
                        encoding=encoding,
                        on_bad_lines='skip',
                        engine='python'
                    )
                
                self.detected_encoding = encoding
                print(f"✅ Successfully loaded with encoding: {encoding}")
                return True, f"CSV loaded with {encoding} encoding"
            
            except UnicodeDecodeError as e:
                errors.append(f"{encoding}: {str(e)[:50]}")
                continue
            except pd.errors.ParserError as e:
                errors.append(f"{encoding}: Parser error")
                continue
            except Exception as e:
                errors.append(f"{encoding}: {str(e)[:50]}")
                continue
        
        # All encodings failed
        return False, f"Could not decode file. Tried: {', '.join(encoding_list[:3])}"
    
    def load_csv_with_encoding_fix(self, file_path_or_bytes, filename=""):
        """
        Load CSV with specific encoding handling for common issues
        """
        try:
            # Try UTF-8 first (most common)
            if isinstance(file_path_or_bytes, bytes):
                self.df = pd.read_csv(
                    io.BytesIO(file_path_or_bytes),
                    encoding='utf-8-sig',  # Handles BOM
                    on_bad_lines='warn'
                )
            else:
                self.df = pd.read_csv(
                    file_path_or_bytes,
                    encoding='utf-8-sig',
                    on_bad_lines='warn'
                )
            
            self.detected_encoding = 'utf-8-sig'
            return True, "CSV loaded successfully (UTF-8)"
        
        except UnicodeDecodeError:
            # Fallback to latin-1 (most robust)
            try:
                if isinstance(file_path_or_bytes, bytes):
                    self.df = pd.read_csv(
                        io.BytesIO(file_path_or_bytes),
                        encoding='latin-1',
                        on_bad_lines='skip'
                    )
                else:
                    self.df = pd.read_csv(
                        file_path_or_bytes,
                        encoding='latin-1',
                        on_bad_lines='skip'
                    )
                
                self.detected_encoding = 'latin-1'
                return True, "CSV loaded with latin-1 encoding (some characters may be replaced)"
            
            except Exception as e:
                return False, f"Failed to load CSV: {str(e)}"
    
    def clean_dataframe_encoding(self):
        """
        Clean dataframe from encoding artifacts
        - Remove BOM characters
        - Fix garbled text
        - Clean column names
        """
        if self.df is None:
            return
        
        try:
            # Fix column names (remove BOM if present)
            self.df.columns = self.df.columns.str.replace('ï»¿', '')  # Common BOM artifact
            self.df.columns = self.df.columns.str.strip()  # Remove whitespace
            
            # Fix string columns
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # Remove BOM from values
                    self.df[col] = self.df[col].apply(
                        lambda x: str(x).replace('ï»¿', '') if pd.notna(x) else x
                    )
                    # Clean whitespace
                    self.df[col] = self.df[col].apply(
                        lambda x: str(x).strip() if pd.notna(x) else x
                    )
            
            print("✅ DataFrame cleaned from encoding artifacts")
        except Exception as e:
            print(f"⚠️ Error cleaning dataframe: {e}")
    
    def load_csv(self, file_path_or_bytes, filename=""):
        """
        Main method to load CSV file with comprehensive error handling
        """
        try:
            # First attempt: Use fallback method
            success, message = self.load_csv_with_fallback(file_path_or_bytes, filename)
            
            if not success:
                return False, message
            
            # Clean encoding artifacts
            self.clean_dataframe_encoding()
            
            # Store file info
            self.file_info = {
                'source': filename or 'uploaded_file',
                'format': 'csv',
                'loaded_at': datetime.now(),
                'encoding': self.detected_encoding,
                'rows': len(self.df),
                'columns': len(self.df.columns)
            }
            
            return True, f"✅ CSV loaded successfully with {self.detected_encoding} encoding"
        
        except Exception as e:
            return False, f"❌ Error loading CSV: {str(e)}"
    
    def load_excel(self, file_path_or_bytes, filename=""):
        """
        Load Excel file with error handling
        Excel files are less prone to encoding issues but handle errors anyway
        """
        try:
            if isinstance(file_path_or_bytes, bytes):
                self.df = pd.read_excel(
                    io.BytesIO(file_path_or_bytes),
                    engine='openpyxl'  # More robust than default
                )
            else:
                self.df = pd.read_excel(
                    file_path_or_bytes,
                    engine='openpyxl'
                )
            
            # Clean any encoding issues
            self.clean_dataframe_encoding()
            
            self.file_info = {
                'source': filename or 'uploaded_file',
                'format': 'excel',
                'loaded_at': datetime.now(),
                'encoding': 'xlsx',
                'rows': len(self.df),
                'columns': len(self.df.columns)
            }
            
            return True, "✅ Excel file loaded successfully"
        
        except Exception as e:
            return False, f"❌ Error loading Excel: {str(e)}"
    
    def validate_dataframe(self):
        """
        Validate dataframe integrity after loading
        """
        if self.df is None:
            return False, "DataFrame is empty"
        
        issues = []
        
        # Check for completely empty columns
        empty_cols = self.df.columns[self.df.isna().all()].tolist()
        if empty_cols:
            issues.append(f"⚠️ Empty columns: {', '.join(empty_cols)}")
            # Remove empty columns
            self.df = self.df.dropna(axis=1, how='all')
        
        # Check for completely empty rows
        empty_rows = len(self.df[self.df.isna().all(axis=1)])
        if empty_rows > 0:
            issues.append(f"⚠️ Found {empty_rows} completely empty rows")
            # Remove empty rows
            self.df = self.df.dropna(how='all')
        
        # Check for duplicate columns
        duplicate_cols = self.df.columns[self.df.columns.duplicated()].tolist()
        if duplicate_cols:
            issues.append(f"⚠️ Duplicate columns found: {set(duplicate_cols)}")
        
        if issues:
            return True, "\n".join(issues)  # Return warnings but still valid
        
        return True, "✅ DataFrame validation passed"
    
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
            'encoding': self.detected_encoding,
            'file_info': self.file_info
        }


def load_data(uploaded_file):
    """
    Wrapper function for Streamlit file uploads
    """
    if uploaded_file is None:
        return None, None
    
    loader = DataLoader()
    
    # Read file bytes
    file_bytes = uploaded_file.read()
    
    # Determine file type
    if uploaded_file.name.endswith('.csv'):
        success, message = loader.load_csv(file_bytes, uploaded_file.name)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        success, message = loader.load_excel(file_bytes, uploaded_file.name)
    else:
        return None, f"❌ Unsupported file format: {uploaded_file.name}"
    
    if not success:
        return None, message
    
    # Validate dataframe
    valid, validation_msg = loader.validate_dataframe()
    
    if valid:
        return loader.df, f"✅ {message}\n{validation_msg}"
    else:
        return loader.df, f"⚠️ {message}\n{validation_msg}"
