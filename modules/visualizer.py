import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy import stats

class Visualizer:
    @staticmethod
    def create_histogram(df, col, bins=30):
        """Create histogram"""
        fig = go.Figure(data=[go.Histogram(x=df[col], nbinsx=bins)])
        fig.update_layout(
            title=f"Distribution of {col}",
            xaxis_title=col,
            yaxis_title="Frequency",
            hovermode='x unified',
            height=600,
            template="plotly_white"
        )
        return fig
    
    @staticmethod
    def create_boxplot(df, col):
        """Create box plot"""
        fig = go.Figure(data=[go.Box(y=df[col])])
        fig.update_layout(
            title=f"Box Plot of {col}",
            yaxis_title=col,
            height=600,
            template="plotly_white"
        )
        return fig
    
    @staticmethod
    def create_violin_plot(df, col):
        """Create violin plot"""
        fig = go.Figure(data=[go.Violin(y=df[col])])
        fig.update_layout(
            title=f"Violin Plot of {col}",
            yaxis_title=col,
            height=600,
            template="plotly_white"
        )
        return fig
    
    @staticmethod
    def create_scatter(df, col1, col2, color_by=None):
        """Create scatter plot"""
        fig = px.scatter(df, x=col1, y=col2, color=color_by,
                        title=f"Scatter: {col1} vs {col2}",
                        height=600)
        fig.update_layout(template="plotly_white", hovermode='closest')
        return fig
    
    @staticmethod
    def create_bar_chart(df, col):
        """Create bar chart for categorical data"""
        value_counts = df[col].value_counts().head(20)
        fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
        fig.update_layout(
            title=f"Value Counts of {col}",
            xaxis_title=col,
            yaxis_title="Count",
            height=600,
            template="plotly_white"
        )
        return fig
    
    @staticmethod
    def create_pie_chart(df, col):
        """Create pie chart"""
        value_counts = df[col].value_counts().head(10)
        fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values)])
        fig.update_layout(title=f"Proportions of {col}", height=600)
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df):
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig = px.imshow(corr, color_continuous_scale='RdBu_r',
                       title="Correlation Matrix",
                       zmin=-1, zmax=1, height=700, width=800)
        return fig
    
    @staticmethod
    def create_qq_plot(df, col):
        """Create Q-Q plot"""
        data = df[col].dropna()
        sorted_data = np.sort(data)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, 
                                mode='markers', name='Data'))
        fig.add_trace(go.Scatter(
            x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
            y=[sorted_data.min(), sorted_data.max()],
            mode='lines',
            name='Perfect Fit'
        ))
        
        fig.update_layout(
            title=f"Q-Q Plot of {col}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=600,
            template="plotly_white"
        )
        return fig
    
    @staticmethod
    def create_grouped_bar(df, numeric_col, cat_col):
        """Create grouped bar chart"""
        fig = px.bar(df, x=cat_col, y=numeric_col,
                    title=f"{numeric_col} by {cat_col}",
                    height=600)
        fig.update_layout(template="plotly_white")
        return fig
    
    @staticmethod
    def create_line_plot(df, col1, col2):
        """Create line plot"""
        df_sorted = df.sort_values(col1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sorted[col1], y=df_sorted[col2], 
                                mode='lines+markers'))
        fig.update_layout(
            title=f"Line Plot: {col1} vs {col2}",
            xaxis_title=col1,
            yaxis_title=col2,
            height=600,
            template="plotly_white"
        )
        return fig

def show_visualizations(df, viz_type, **kwargs):
    """Wrapper function for visualizations"""
    viz = Visualizer()
    
    if viz_type == 'histogram':
        return viz.create_histogram(df, kwargs['col'], kwargs.get('bins', 30))
    elif viz_type == 'boxplot':
        return viz.create_boxplot(df, kwargs['col'])
    elif viz_type == 'violin':
        return viz.create_violin_plot(df, kwargs['col'])
    elif viz_type == 'scatter':
        return viz.create_scatter(df, kwargs['col1'], kwargs['col2'], kwargs.get('color_by'))
    elif viz_type == 'bar':
        return viz.create_bar_chart(df, kwargs['col'])
    elif viz_type == 'pie':
        return viz.create_pie_chart(df, kwargs['col'])
    elif viz_type == 'heatmap':
        return viz.create_correlation_heatmap(df)
    elif viz_type == 'qq':
        return viz.create_qq_plot(df, kwargs['col'])
    elif viz_type == 'grouped_bar':
        return viz.create_grouped_bar(df, kwargs['numeric_col'], kwargs['cat_col'])
    elif viz_type == 'line':
        return viz.create_line_plot(df, kwargs['col1'], kwargs['col2'])