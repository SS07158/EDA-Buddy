import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import load_data
from modules.data_analyzer import analyze_dataset
from modules.visualizer import show_visualizations
from modules.explanations import get_plot_explanation
from modules.preprocessor import suggest_preprocessing
from modules.code_generator import generate_code_snippets
import io
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# ============================================================
# PAGE CONFIGURATION & INITIALIZATION
# ============================================================

st.set_page_config(
    page_title="EDA Buddy - Learn Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subheader {
            font-size: 1.5rem;
            color: #667eea;
            margin-top: 20px;
        }
        .info-box {
            background-color: #e7f3ff;
            border-left: 4px solid #667eea;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'selected_column' not in st.session_state:
    st.session_state.selected_column = None

# ============================================================
# VISUALIZATION FUNCTIONS (DEFINED FIRST)
# ============================================================

def show_histogram(df, col):
    """Display histogram with explanation"""
    fig = go.Figure(data=[go.Histogram(x=df[col], nbinsx=30, name=col)])
    fig.update_layout(
        title=f"Histogram of {col}",
        xaxis_title=col,
        yaxis_title="Frequency",
        hovermode='x unified',
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Histogram")
    st.markdown(get_plot_explanation("histogram", df, col))

def show_boxplot(df, col):
    """Display box plot with explanation"""
    fig = go.Figure(data=[go.Box(y=df[col], name=col)])
    fig.update_layout(
        title=f"Box Plot of {col}",
        yaxis_title=col,
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Box Plot")
    st.markdown(get_plot_explanation("boxplot", df, col))

def show_violin(df, col):
    """Display violin plot with explanation"""
    fig = go.Figure(data=[go.Violin(y=df[col], name=col)])
    fig.update_layout(
        title=f"Violin Plot of {col}",
        yaxis_title=col,
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Violin Plot")
    st.markdown(get_plot_explanation("violin", df, col))

def show_qq_plot(df, col):
    """Display Q-Q plot with explanation"""
    data = df[col].dropna()
    sorted_data = np.sort(data)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Data'))
    fig.add_trace(go.Scatter(
        x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
        y=[data.min(), data.max()],
        mode='lines',
        name='Reference Line'
    ))
    
    fig.update_layout(
        title=f"Q-Q Plot of {col}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Q-Q Plot")
    st.markdown(get_plot_explanation("qq_plot", df, col))

def show_distribution_fit(df, col):
    """Display distribution fit with explanation"""
    data = df[col].dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(x=data, nbinsx=30, name='Data', opacity=0.7))
    
    # Add normal distribution fit
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    pdf = stats.norm.pdf(x, mu, sigma)
    max_count = len(data) / 30
    pdf_scaled = pdf * max_count * (data.max() - data.min()) / 30
    
    fig.add_trace(go.Scatter(x=x, y=pdf_scaled, name='Normal Fit', mode='lines'))
    
    fig.update_layout(
        title=f"Distribution Fit of {col}",
        xaxis_title=col,
        yaxis_title="Frequency",
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding Distribution Fitting")
    st.markdown(get_plot_explanation("distribution_fit", df, col))

def show_bar_chart(df, col):
    """Display bar chart with explanation"""
    value_counts = df[col].value_counts()
    
    fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
    fig.update_layout(
        title=f"Bar Chart of {col}",
        xaxis_title=col,
        yaxis_title="Count",
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Bar Chart")
    st.markdown(get_plot_explanation("bar_chart", df, col))

def show_pie_chart(df, col):
    """Display pie chart with explanation"""
    value_counts = df[col].value_counts()
    
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values)])
    fig.update_layout(title=f"Pie Chart of {col}", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Pie Chart")
    st.markdown(get_plot_explanation("pie_chart", df, col))

def show_count_dist(df, col):
    """Display count distribution with explanation"""
    value_counts = df[col].value_counts().sort_values(ascending=True)
    
    fig = go.Figure(data=[go.Barh(y=value_counts.index, x=value_counts.values)])
    fig.update_layout(
        title=f"Count Distribution of {col}",
        xaxis_title="Count",
        yaxis_title=col,
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Count Distribution")
    st.markdown(get_plot_explanation("count_dist", df, col))

def show_scatter_plot(df, col1, col2, color_by):
    """Display scatter plot with explanation"""
    fig = px.scatter(df, x=col1, y=col2, color=color_by, 
                     title=f"Scatter Plot: {col1} vs {col2}",
                     height=600)
    fig.update_layout(hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Scatter Plot")
    st.markdown(get_plot_explanation("scatter", df, col1, col2))

def show_line_plot(df, col1, col2):
    """Display line plot with explanation"""
    df_sorted = df.sort_values(col1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted[col1], y=df_sorted[col2], mode='lines+markers'))
    fig.update_layout(
        title=f"Line Plot: {col1} vs {col2}",
        xaxis_title=col1,
        yaxis_title=col2,
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Line Plot")
    st.markdown(get_plot_explanation("line_plot", df, col1, col2))

def show_grouped_bar(df, numeric_col, cat_col):
    """Display grouped bar chart with explanation"""
    fig = px.bar(df, x=cat_col, y=numeric_col, 
                 title=f"{numeric_col} by {cat_col}",
                 barmode='group', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Grouped Bar Chart")
    st.markdown(get_plot_explanation("grouped_bar", df, numeric_col, cat_col))

def show_correlation_heatmap(df, numeric_cols):
    """Display correlation heatmap with explanation"""
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, color_continuous_scale='RdBu_r', 
                    title="Correlation Heatmap",
                    zmin=-1, zmax=1, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Heatmap")
    st.markdown(get_plot_explanation("heatmap", df))

def show_relationship_plot(df, x_col, y_col, group_col):
    """Display relationship plot with explanation"""
    fig = px.scatter(df, x=x_col, y=y_col, color=group_col,
                     title=f"Relationship: {x_col} vs {y_col}",
                     height=600, trendline="ols")
    fig.update_layout(hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Relationship Plot")
    st.markdown(get_plot_explanation("relationship", df, x_col, y_col))

def show_grouped_distribution(df, col, group_col):
    """Display grouped distribution with explanation"""
    if group_col:
        fig = px.histogram(df, x=col, color=group_col, nbins=30,
                          title=f"Distribution of {col} by {group_col}",
                          barmode='overlay', height=500)
    else:
        fig = px.histogram(df, x=col, nbins=30,
                          title=f"Distribution of {col}",
                          height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Understanding This Distribution")
    st.markdown(get_plot_explanation("histogram", df, col))

# ============================================================
# SINGLE COLUMN VISUALIZATION FUNCTIONS
# ============================================================

def show_single_column_viz(df):
    """Show single column analysis"""
    col = st.selectbox("Select a column:", df.columns)
    
    if col:
        col_type = df[col].dtype
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 📊 Available Plots")
            st.markdown("Choose a visualization to explore this column.")
        
        with col1:
            if pd.api.types.is_numeric_dtype(df[col]):
                st.markdown("**Numeric Column**")
                viz_choice = st.radio(
                    "Select plot:",
                    ["Histogram", "Box Plot", "Violin Plot", "Q-Q Plot", "Distribution Fit"],
                    horizontal=True
                )
                
                if viz_choice == "Histogram":
                    show_histogram(df, col)
                elif viz_choice == "Box Plot":
                    show_boxplot(df, col)
                elif viz_choice == "Violin Plot":
                    show_violin(df, col)
                elif viz_choice == "Q-Q Plot":
                    show_qq_plot(df, col)
                elif viz_choice == "Distribution Fit":
                    show_distribution_fit(df, col)
            
            else:
                st.markdown("**Categorical Column**")
                viz_choice = st.radio(
                    "Select plot:",
                    ["Bar Chart", "Pie Chart", "Count Distribution"],
                    horizontal=True
                )
                
                if viz_choice == "Bar Chart":
                    show_bar_chart(df, col)
                elif viz_choice == "Pie Chart":
                    show_pie_chart(df, col)
                elif viz_choice == "Count Distribution":
                    show_count_dist(df, col)

def show_multi_column_viz(df):
    """Show multi-column analysis"""
    st.markdown("#### Multi-Column Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    all_cols = numeric_cols + cat_cols
    
    if len(all_cols) < 2:
        st.warning("Need at least 2 columns for multi-column analysis.")
        return
    
    viz_type = st.radio(
        "Select visualization type:",
        ["Scatter Plot", "Line Plot", "Grouped Bar Chart", "Heatmap"],
        horizontal=True
    )
    
    if viz_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for scatter plot.")
            return
        
        col1 = st.selectbox("X-axis:", numeric_cols)
        col2 = st.selectbox("Y-axis:", numeric_cols)
        color_by = st.selectbox("Color by (optional):", [None] + all_cols)
        
        if col1 != col2:
            show_scatter_plot(df, col1, col2, color_by)
    
    elif viz_type == "Line Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for line plot.")
            return
        
        col1 = st.selectbox("X-axis:", numeric_cols)
        col2 = st.selectbox("Y-axis:", numeric_cols)
        
        if col1 != col2:
            show_line_plot(df, col1, col2)
    
    elif viz_type == "Grouped Bar Chart":
        if len(numeric_cols) < 1 or len(cat_cols) < 1:
            st.warning("Need at least 1 numeric and 1 categorical column.")
            return
        
        numeric_col = st.selectbox("Numeric column:", numeric_cols)
        cat_col = st.selectbox("Group by:", cat_cols)
        
        show_grouped_bar(df, numeric_col, cat_col)
    
    elif viz_type == "Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for heatmap.")
            return
        
        show_correlation_heatmap(df, numeric_cols)

def show_relationship_viz(df):
    """Show relationship analysis"""
    st.markdown("#### Relationship Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        return
    
    x_col = st.selectbox("X-axis (numeric):", numeric_cols)
    y_col = st.selectbox("Y-axis (numeric):", [c for c in numeric_cols if c != x_col])
    group_col = st.selectbox("Group by (optional):", [None] + cat_cols)
    
    show_relationship_plot(df, x_col, y_col, group_col)

def show_distribution_viz(df):
    """Show distribution analysis"""
    st.markdown("#### Distribution Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not numeric_cols:
        st.warning("Need at least 1 numeric column.")
        return
    
    col = st.selectbox("Select column:", numeric_cols)
    
    if cat_cols:
        group_by = st.selectbox("Group by (optional):", [None] + cat_cols)
        show_grouped_distribution(df, col, group_by)
    else:
        show_histogram(df, col)

# ============================================================
# PAGE FUNCTIONS (DEFINED BEFORE MAIN)
# ============================================================

def show_home_page():
    """Display home page"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to EDA Buddy! 👋
        
        **EDA Buddy** is your interactive learning companion for mastering:
        - 🔍 **Exploratory Data Analysis (EDA)** - Understand your data deeply
        - 🧹 **Data Preprocessing** - Clean and prepare data for analysis
        - 📊 **Visualization** - Create meaningful plots to tell data stories
        - 💡 **Data Interpretation** - Learn what your plots actually mean
        
        #### How It Works:
        1. **Upload** a CSV or Excel file
        2. **Explore** interactive visualizations
        3. **Learn** through detailed explanations
        4. **Preprocess** with guided suggestions
        5. **Copy** ready-to-use Python code
        
        #### Key Features:
        - ✅ Automatic column type detection (numeric, categorical, datetime)
        - ✅ Intelligent visualization suggestions
        - ✅ Educational explanations for every plot
        - ✅ Preprocessing guidance with code examples
        - ✅ Python code generation for reproducibility
        - ✅ Interactive filters and customizations
        """)
    
    with col2:
        st.info("""
        ### 🎯 Quick Start
        
        **New here?**
        1. Go to "📤 Upload Data"
        2. Use sample data or upload yours
        3. Explore "📊 Data Overview"
        4. Check "📈 Visualizations"
        5. Learn from "💻 Code Snippets"
        """)
    
    # Sample datasets info
    st.markdown("---")
    st.markdown("### 📚 Sample Datasets")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Iris Dataset**
        - Flower measurements
        - Perfect for beginners
        - 150 samples, 5 columns
        """)
    with col2:
        st.markdown("""
        **Titanic Dataset**
        - Passenger survival data
        - Mixed data types
        - Preprocessing practice
        """)
    with col3:
        st.markdown("""
        **Student Performance**
        - Academic data
        - Correlations demo
        - Real-world scenario
        """)

def show_upload_page():
    """Display upload data page"""
    st.markdown("<div class='subheader'>📤 Upload Your Data</div>", unsafe_allow_html=True)
    
    # Upload options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Option 1: Load Sample Data")
        sample_choice = st.selectbox(
            "Select a sample dataset:",
            ["None", "Iris", "Titanic", "Student Performance"]
        )
        
        if sample_choice != "None":
            if st.button("📥 Load Sample Data"):
                if sample_choice == "Iris":
                    from sklearn.datasets import load_iris
                    iris = load_iris()
                    st.session_state.df = pd.DataFrame(iris.data, columns=iris.feature_names)
                    st.session_state.df['species'] = iris.target_names[iris.target]
                elif sample_choice == "Titanic":
                    st.session_state.df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv")
                elif sample_choice == "Student Performance":
                    # Create synthetic student data
                    np.random.seed(42)
                    st.session_state.df = pd.DataFrame({
                        'StudentID': range(1, 101),
                        'Math': np.random.normal(75, 15, 100),
                        'Science': np.random.normal(78, 12, 100),
                        'English': np.random.normal(72, 18, 100),
                        'AttendanceRate': np.random.uniform(70, 100, 100),
                        'StudyHours': np.random.exponential(5, 100),
                        'Grade': np.random.choice(['A', 'B', 'C', 'D'], 100)
                    })
                st.success("✅ Sample data loaded successfully!")
                st.rerun()
    
    with col2:
        st.markdown("### Option 2: Upload Your File")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    # Display loaded data info
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### ✅ Data Loaded Successfully!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("Memory", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        
        # Preview
        st.markdown("### 📋 Data Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Column info
        st.markdown("### 📌 Column Information")
        col_info = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Type': [st.session_state.df[col].dtype for col in st.session_state.df.columns],
            'Non-Null': [st.session_state.df[col].notna().sum() for col in st.session_state.df.columns],
            'Null': [st.session_state.df[col].isna().sum() for col in st.session_state.df.columns],
            'Unique': [st.session_state.df[col].nunique() for col in st.session_state.df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

def show_data_overview_page():
    """Display data overview page"""
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first from the '📤 Upload Data' section.")
        return
    
    st.markdown("<div class='subheader'>📊 Data Overview & Statistics</div>", unsafe_allow_html=True)
    
    df = st.session_state.df
    
    # Basic statistics
    st.markdown("### 📈 Basic Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        st.metric("Numeric Columns", numeric_cols)
    with col4:
        categorical_cols = df.select_dtypes(include=['object', 'category']).shape[1]
        st.metric("Categorical Columns", categorical_cols)
    
    # Detailed statistics
    st.markdown("### 🔢 Numeric Columns Statistics")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        st.dataframe(numeric_df.describe().T, use_container_width=True)
    else:
        st.info("No numeric columns found in your dataset.")
    
    # Categorical statistics
    st.markdown("### 📝 Categorical Columns Statistics")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            st.write(f"**{col}**")
            value_counts = df[col].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Values", df[col].nunique())
            with col2:
                st.metric("Most Common", value_counts.index[0])
            st.dataframe(value_counts, use_container_width=True)
    else:
        st.info("No categorical columns found in your dataset.")
    
    # Missing values analysis
    st.markdown("### ❌ Missing Values Analysis")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Percentage': missing_pct.values
    }).sort_values('Missing Count', ascending=False)
    
    if missing_df['Missing Count'].sum() > 0:
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        st.markdown("""
        <div class='warning-box'>
        <b>⚠️ Note:</b> Columns with high missing rates (>30%) may need special handling. Consider:
        - Dropping the column if not critical
        - Imputation (mean, median, mode)
        - Using advanced imputation methods
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ No missing values found!")
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix, 
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1,
                       title='Correlation Heatmap',
                       labels=dict(color="Correlation"))
        st.plotly_chart(fig, use_container_width=True)
        
        # High correlations
        st.markdown("**High Correlations (|r| > 0.7):**")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr:
            st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
            st.markdown("""
            <div class='info-box'>
            <b>💡 Tip:</b> High correlations indicate multicollinearity. One of the correlated variables 
            might be redundant for predictive modeling.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No high correlations found.")
    else:
        st.info("Need at least 2 numeric columns for correlation analysis.")

def show_visualizations_page():
    """Display visualizations page"""
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first from the '📤 Upload Data' section.")
        return
    
    st.markdown("<div class='subheader'>📈 Interactive Visualizations</div>", unsafe_allow_html=True)
    
    df = st.session_state.df
    
    # Visualization type selection
    st.markdown("### 🎨 Choose Visualization Type")
    
    viz_type = st.radio(
        "Select visualization:",
        ["Single Column Analysis", "Multi-Column Comparison", "Relationship Analysis", "Distribution Analysis"],
        horizontal=True
    )
    
    if viz_type == "Single Column Analysis":
        show_single_column_viz(df)
    elif viz_type == "Multi-Column Comparison":
        show_multi_column_viz(df)
    elif viz_type == "Relationship Analysis":
        show_relationship_viz(df)
    elif viz_type == "Distribution Analysis":
        show_distribution_viz(df)

def show_preprocessing_page():
    """Display preprocessing guide page"""
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first from the '📤 Upload Data' section.")
        return
    
    st.markdown("<div class='subheader'>🔧 Preprocessing Guide</div>", unsafe_allow_html=True)
    
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col = st.selectbox("Select a column to preprocess:", df.columns)
    
    if col:
        col_type = df[col].dtype
        
        st.markdown(f"### Column: {col} ({col_type})")
        
        if col in numeric_cols:
            st.markdown("#### Numeric Column Preprocessing")
            
            st.markdown("**1. Handling Missing Values**")
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                st.warning(f"⚠️ This column has {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
                
                strategy = st.radio("Choose imputation strategy:", 
                                   ["Mean", "Median", "Mode", "Forward Fill", "Drop Rows"])
                st.markdown(get_plot_explanation("missing", df, col))
                
                with st.expander("💻 See Python Code"):
                    st.code(generate_code_snippets("missing_numeric", col, strategy), language="python")
            else:
                st.success("✅ No missing values")
            
            st.markdown("**2. Handling Outliers**")
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            
            if outliers > 0:
                st.warning(f"⚠️ This column has {outliers} potential outliers ({outliers/len(df)*100:.1f}%)")
                
                outlier_strategy = st.radio("Choose outlier handling strategy:",
                                           ["IQR Method", "Z-Score", "Capping", "Keep As Is"])
                st.markdown(get_plot_explanation("outliers", df, col))
                
                with st.expander("💻 See Python Code"):
                    st.code(generate_code_snippets("outliers_numeric", col, outlier_strategy), language="python")
            else:
                st.success("✅ No significant outliers")
            
            st.markdown("**3. Scaling/Normalization**")
            scale_method = st.radio("Choose scaling method:",
                                   ["StandardScaler", "MinMaxScaler", "RobustScaler", "No Scaling"])
            st.markdown(get_plot_explanation("scaling", df, col))
            
            with st.expander("💻 See Python Code"):
                st.code(generate_code_snippets("scaling", col, scale_method), language="python")
            
            st.markdown("**4. Feature Engineering**")
            transform = st.radio("Apply transformation?",
                                ["None", "Log Transform", "Square Root", "Box-Cox"])
            if transform != "None":
                st.markdown(get_plot_explanation("transform", df, col))
                with st.expander("💻 See Python Code"):
                    st.code(generate_code_snippets("transform", col, transform), language="python")
        
        else:  # Categorical
            st.markdown("#### Categorical Column Preprocessing")
            
            st.markdown("**1. Handling Missing Values**")
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                st.warning(f"⚠️ This column has {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
                
                cat_missing_strategy = st.radio("Choose imputation strategy:",
                                              ["Mode", "New Category", "Drop Rows"])
                st.markdown(get_plot_explanation("missing", df, col))
                
                with st.expander("💻 See Python Code"):
                    st.code(generate_code_snippets("missing_categorical", col, cat_missing_strategy), language="python")
            else:
                st.success("✅ No missing values")
            
            st.markdown("**2. Encoding Categorical Variables**")
            unique_count = df[col].nunique()
            st.info(f"This column has {unique_count} unique values")
            
            if unique_count <= 2:
                encoding = st.radio("Choose encoding method:",
                                  ["Label Encoding", "One-Hot Encoding"])
            else:
                encoding = st.radio("Choose encoding method:",
                                  ["One-Hot Encoding", "Label Encoding", "Target Encoding", "Frequency Encoding"])
            
            st.markdown(get_plot_explanation("encoding", df, col))
            
            with st.expander("💻 See Python Code"):
                st.code(generate_code_snippets("encoding", col, encoding), language="python")
            
            st.markdown("**3. Handling Rare Categories**")
            value_counts = df[col].value_counts()
            rare_pct = (value_counts / len(df) * 100)
            rare_cats = (rare_pct < 5).sum()
            
            if rare_cats > 0:
                st.warning(f"⚠️ This column has {rare_cats} rare categories (<5% frequency)")
                
                rare_strategy = st.radio("How to handle rare categories?",
                                        ["Group as 'Other'", "Keep Separate", "Drop Rows"])
                st.markdown(get_plot_explanation("rare_categories", df, col))
                
                with st.expander("💻 See Python Code"):
                    st.code(generate_code_snippets("rare_categories", col, rare_strategy), language="python")

def show_code_snippets_page():
    """Display code snippets page"""
    st.markdown("<div class='subheader'>💻 Python Code Snippets</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🔧 Preprocessing", "📈 Analysis", "📥 File Handling"])
    
    with tab1:
        st.markdown("### Visualization Code Snippets")
        
        viz_snippets = {
            "Histogram": """
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['column_name'], bins=30, edgecolor='black')
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.title('Histogram of Column Name')
plt.show()
""",
            "Scatter Plot": """
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['column1'], df['column2'])
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('Scatter Plot')
plt.show()
""",
            "Box Plot": """
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.boxplot(df['column_name'])
plt.ylabel('Values')
plt.title('Box Plot')
plt.show()
""",
            "Correlation Heatmap": """
import matplotlib.pyplot as plt
import seaborn as sns

corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
"""
        }
        
        for name, code in viz_snippets.items():
            with st.expander(f"📊 {name}"):
                st.code(code, language="python")
    
    with tab2:
        st.markdown("### Preprocessing Code Snippets")
        
        preprocess_snippets = {
            "Handle Missing Values (Mean)": """
# Fill missing values with mean
df['column_name'].fillna(df['column_name'].mean(), inplace=True)
""",
            "Handle Missing Values (Median)": """
# Fill missing values with median
df['column_name'].fillna(df['column_name'].median(), inplace=True)
""",
            "StandardScaler": """
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['column_name_scaled'] = scaler.fit_transform(df[['column_name']])
""",
            "MinMaxScaler": """
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['column_name_scaled'] = scaler.fit_transform(df[['column_name']])
""",
            "One-Hot Encoding": """
df_encoded = pd.get_dummies(df, columns=['categorical_column'], drop_first=True)
""",
            "Label Encoding": """
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['column_name_encoded'] = le.fit_transform(df['column_name'])
"""
        }
        
        for name, code in preprocess_snippets.items():
            with st.expander(f"🔧 {name}"):
                st.code(code, language="python")
    
    with tab3:
        st.markdown("### Analysis Code Snippets")
        
        analysis_snippets = {
            "Basic Statistics": """
print(df.describe())
print(df.info())
print(df.isnull().sum())
""",
            "Correlation": """
correlation = df.corr()
print(correlation)
""",
            "Value Counts": """
print(df['column_name'].value_counts())
""",
            "Groupby Analysis": """
grouped = df.groupby('category_column').agg({
    'numeric_column': ['mean', 'sum', 'count']
})
print(grouped)
"""
        }
        
        for name, code in analysis_snippets.items():
            with st.expander(f"📈 {name}"):
                st.code(code, language="python")
    
    with tab4:
        st.markdown("### File Handling Code Snippets")
        
        file_snippets = {
            "Read CSV": """
import pandas as pd

df = pd.read_csv('filename.csv')
""",
            "Read Excel": """
import pandas as pd

df = pd.read_excel('filename.xlsx', sheet_name='Sheet1')
""",
            "Save CSV": """
df.to_csv('output.csv', index=False)
""",
            "Save Excel": """
df.to_excel('output.xlsx', index=False)
"""
        }
        
        for name, code in file_snippets.items():
            with st.expander(f"📥 {name}"):
                st.code(code, language="python")

def show_learning_resources_page():
    """Display learning resources page"""
    st.markdown("<div class='subheader'>❓ Learning Resources</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 EDA Best Practices
        
        **What is EDA?**
        Exploratory Data Analysis (EDA) is the process of investigating data to:
        - Understand data structure and types
        - Identify missing values and outliers
        - Discover patterns and relationships
        - Guide further analysis
        
        **Key Steps:**
        1. Load and inspect data
        2. Check data types and missing values
        3. Calculate summary statistics
        4. Create visualizations
        5. Identify patterns and anomalies
        6. Generate insights
        """)
        
        st.markdown("""
        ### 🎓 Data Types
        
        **Numeric (Quantitative)**
        - Continuous: Any decimal value (height, temperature)
        - Discrete: Integer values (count, age)
        
        **Categorical (Qualitative)**
        - Nominal: No order (color, category)
        - Ordinal: Has order (rating, size)
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Visualization Guide
        
        **For Numeric Data:**
        - Histogram → Distribution shape
        - Box Plot → Outliers and quartiles
        - Scatter → Relationships
        
        **For Categorical Data:**
        - Bar Chart → Frequencies
        - Pie Chart → Proportions
        
        **For Relationships:**
        - Scatter Plot → Correlation
        - Heatmap → Multiple correlations
        """)
        
        st.markdown("""
        ### 🔧 Preprocessing Essentials
        
        **Missing Values:**
        - Mean/Median imputation
        - Mode for categories
        - Drop if too many missing
        
        **Scaling:**
        - StandardScaler (mean=0, std=1)
        - MinMaxScaler (0-1 range)
        
        **Encoding:**
        - One-Hot for categories
        - Label encoding for ordinal
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔗 External Resources
    
    - [Pandas Documentation](https://pandas.pydata.org/docs/)
    - [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
    - [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)
    - [Seaborn Gallery](https://seaborn.pydata.org/examples.html)
    - [Plotly Documentation](https://plotly.com/python/)
    """)

# ============================================================
# MAIN APP FLOW
# ============================================================

def main():
    """Main app flow"""
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 📚 Navigation")
        page = st.radio("Choose a section:", [
            "🏠 Home",
            "📤 Upload Data",
            "📊 Data Overview",
            "📈 Visualizations",
            "🔧 Preprocessing Guide",
            "💻 Code Snippets",
            "❓ Learning Resources"
        ])
    
    # Main content
    st.markdown("<div class='main-header'>📊 EDA Buddy</div>", unsafe_allow_html=True)
    st.markdown("*Your interactive guide to mastering Exploratory Data Analysis & Preprocessing*")
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    
    elif page == "📤 Upload Data":
        show_upload_page()
    
    elif page == "📊 Data Overview":
        show_data_overview_page()
    
    elif page == "📈 Visualizations":
        show_visualizations_page()
    
    elif page == "🔧 Preprocessing Guide":
        show_preprocessing_page()
    
    elif page == "💻 Code Snippets":
        show_code_snippets_page()
    
    elif page == "❓ Learning Resources":
        show_learning_resources_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888; font-size: 0.9rem; margin-top: 20px;'>
            <p>🚀 EDA Buddy v1.0 | Built with Streamlit | Learn • Practice • Master Data Analysis</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()