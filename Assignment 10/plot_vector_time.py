import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('vector_time.csv')  # Replace with your file path

# Check the first few rows to ensure it's loaded correctly
print(df.head())

# Prepare data for plotting
sizes = sorted(df['Size'].unique())  # Get unique sizes
slices_list = sorted(df['Slices'].unique())  # Get unique slices

# Create a dictionary to store total times for each slice
time_by_slices = {slices: [] for slices in slices_list}
time_by_sizes = {size: [] for size in sizes}

# Populate the time_by_slices and time_by_sizes dictionaries
for size in sizes:
    for slices in slices_list:
        # Filter data for the given size and slices
        result = df[(df['Size'] == size) & (df['Slices'] == slices)]
        if not result.empty:
            total_time = result['Total Time']

            # Append to the dictionary for time_by_slices (for Size vs Time plot)
            time_by_slices[slices].append(total_time)

            # Append to the dictionary for time_by_sizes (for Slice vs Time plot)
            time_by_sizes[size].append(total_time)

# --- Plot 1: Size vs Total Time with logarithmic x-axis ---
plt.figure(figsize=(10, 6))

for slices in slices_list:
    plt.plot(sizes, time_by_slices[slices], label=f'Slices: {slices}', marker='o')

# Labels and title for Size vs Total Time
plt.xlabel('Size')
plt.ylabel('Total Time (s)')
plt.title('Size vs Total Time with Different Slices')

# Set x-axis to logarithmic scale
plt.xscale('log')

# Set the x-ticks to be the actual size values
plt.xticks(sizes, labels=[str(size) for size in sizes])

# Rotate the x-tick labels for better readability if needed
plt.xticks(rotation=45)

# Add grid, legend, and show the plot
plt.legend()
plt.grid(True)

# Show Size vs Time plot
plt.show()

# --- Plot 2: Slice vs Total Time with lines for each size ---
plt.figure(figsize=(10, 6))

for size in sizes:
    plt.plot(slices_list, time_by_sizes[size], label=f'Size: {size}', marker='o')

# Labels and title for Slice vs Total Time
plt.xlabel('Slices')
plt.ylabel('Total Time (s)')
plt.title('Slices vs Total Time with Different Sizes')
plt.legend()
plt.grid(True)

# Show Slice vs Time plot
plt.show()
