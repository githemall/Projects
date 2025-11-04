import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Prepare Data
# Organize the experiment results into a DataFrame.
data = {
    'Version': ['YOLOv8', 'YOLOv8', 'YOLOv8', 'YOLOv11', 'YOLOv11', 'YOLOv11'],
    'Model_size': ['n', 's', 'm', 'n', 's', 'm'],
    'model_name': ['v8n', 'v8s', 'v8m', 'v11n', 'v11s', 'v11m'],
    'gflops': [8.1, 28.5, 78.7, 6.3, 21.3, 67.7],
    'map': [0.298, 0.325, 0.344, 0.320, 0.331, 0.340],
    'params': [3.01, 11.13, 25.85, 2.59, 9.42, 20.04]
}
df = pd.DataFrame(data)

# Sort the data to draw lines correctly (n -> s -> m)
df['Model_size'] = pd.Categorical(df['Model_size'], categories=['n', 's', 'm'], ordered=True)
df = df.sort_values(['Version', 'Model_size'])

# 2. Generate Graph
# Apply Seaborn style for a more refined look.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

# Draw line plots first to connect the points for each model family.
sns.lineplot(
    data=df,
    x='params',
    y='map',
    hue='Version',
    palette={'YOLOv8': 'orangered', 'YOLOv11': 'royalblue'},
    legend=False, # Hide the line plot legend to avoid duplicates
    linewidth=2.5
)

# Draw scatter plot on top of the lines.
# hue='model_family' will color points for YOLOv8 and YOLOv11 differently.
# style='Model_size' will use different markers for n, s, m models.
ax = sns.scatterplot(
    data=df,
    x='params',
    y='map',
    hue='Version',
    style='Model_size',
    s=250,  # Point size
    palette={'YOLOv8': 'orangered', 'YOLOv11': 'royalblue'} # Color scheme
)

# Add model name text next to each point.
for i, point in df.iterrows():
    ax.text(point['params'] + 1, point['map'], str(point['model_name']),
            horizontalalignment='left', size='medium', color='black', weight='semibold')

# 3. Set Graph Title and Axis Labels in English
plt.title('YOLOv11 vs YOLOv8: Accuracy-Efficiency Trade-off', fontsize=20, pad=20)
plt.xlabel('Size (Parameters(M)) - Lower is Better', fontsize=14)
plt.ylabel('Accuracy (mAP@0.5-0.95) - Higher is Better', fontsize=14)
plt.legend(title='', fontsize=12, loc='lower right')
plt.grid(True)

# Save the graph to a file and display it.
plt.savefig('accuracy_vs_efficiency.png', dpi=300)
plt.show()

print("\nGraph saved as 'accuracy_vs_efficiency.png'.")

