import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('unemployment.csv')

# Display basic information
print("First 5 rows:")
print(df.head())
print("\nData Summary:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Rename columns for ease
df.columns = ['States', 'Date', 'Frequency', 'Estimated Unemployment Rate','Estimated Employed', 'Estimated Labour Participation Rate','Region']

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Group data by State and Date
grouped = df.groupby(['States', 'Date'])['Estimated Unemployment Rate'].mean().reset_index()

# Plot unemployment rate over time for selected states
states_to_plot = ['Maharashtra', 'Kerala', 'Delhi', 'Karnataka']
plt.figure(figsize=(12, 6))

for state in states_to_plot:
  state_data = grouped[grouped['States'] == state]
  plt.plot(state_data['Date'], state_data['Estimated Unemployment Rate'], label=state)

plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# Heatmap of unemployment by state and region
pivot = df.pivot_table(values='Estimated Unemployment Rate',index='States', columns='Region', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap='YlGnBu')
plt.title('Average Unemployment Rate by State and Region')
plt.show(block=True)
