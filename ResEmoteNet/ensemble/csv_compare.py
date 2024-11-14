import pandas as pd

# Load the dataset
df = pd.read_csv('fer2013.csv')

# Total number of samples
total_samples = len(df)

# Count the number of samples for each type in 'Usage' column
usage_counts = df['Usage'].value_counts()

# Display results
print("Total samples:", total_samples)
print("Usage counts:")
print(usage_counts)
