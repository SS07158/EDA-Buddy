def show_upload_page():
    """Display upload data page with better encoding handling"""
    st.markdown("<div class='subheader'>📤 Upload Your Data</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Option 1: Load Sample Data")
        sample_choice = st.selectbox(
            "Select a sample dataset:",
            ["None", "Iris", "Titanic", "Student Performance"]
        )
        
        if sample_choice != "None":
            if st.button("📥 Load Sample Data"):
                try:
                    if sample_choice == "Iris":
                        from sklearn.datasets import load_iris
                        iris = load_iris()
                        st.session_state.df = pd.DataFrame(iris.data, columns=iris.feature_names)
                        st.session_state.df['species'] = iris.target_names[iris.target]
                    elif sample_choice == "Titanic":
                        st.session_state.df = pd.read_csv(
                            "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
                        )
                    elif sample_choice == "Student Performance":
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
                except Exception as e:
                    st.error(f"❌ Error loading sample: {e}")
    
    with col2:
        st.markdown("### Option 2: Upload Your File")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            try:
                from modules.data_loader import load_data
                
                # Use improved loader with encoding handling
                df, message = load_data(uploaded_file)
                
                # Display status message
                if df is not None:
                    st.success(message)
                    st.session_state.df = df
                    st.rerun()
                else:
                    st.error(message)
            
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.info("💡 Try these solutions:")
                st.write("""
                1. Save file as UTF-8 encoding in Excel
                2. Try removing special characters from column names
                3. Use CSV format instead of Excel
                4. Report issue with sample file
                """)
    
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
        
        # Column info with encoding display
        st.markdown("### 📌 Column Information")
        col_info = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Type': [st.session_state.df[col].dtype for col in st.session_state.df.columns],
            'Non-Null': [st.session_state.df[col].notna().sum() for col in st.session_state.df.columns],
            'Null': [st.session_state.df[col].isna().sum() for col in st.session_state.df.columns],
            'Unique': [st.session_state.df[col].nunique() for col in st.session_state.df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Show encoding info
        st.info("""
        📝 **Encoding Detection:**
        The system automatically detects and handles different file encodings
        (UTF-8, Latin-1, etc.) to ensure compatibility.
        """)
