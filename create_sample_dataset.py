import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)

sample_data = pd.DataFrame({
    'Age': np.random.randint(18, 65, 100),
    'Income': np.random.normal(50000, 20000, 100),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
    'Experience': np.random.randint(0, 40, 100),
    'Department': np.random.choice(['Sales', 'IT', 'HR', 'Finance', 'Marketing'], 100),
    'Satisfaction': np.random.randint(1, 11, 100),
    'Bonus': np.random.exponential(5000, 100),
    'Attendance': np.random.uniform(70, 100, 100)
})

# Add some missing values
sample_data.loc[np.random.choice(sample_data.index, 5), 'Income'] = np.nan
sample_data.loc[np.random.choice(sample_data.index, 3), 'Bonus'] = np.nan

# Save
sample_data.to_csv('sample_data/sample_dataset.csv', index=False)
print("Sample dataset created!")