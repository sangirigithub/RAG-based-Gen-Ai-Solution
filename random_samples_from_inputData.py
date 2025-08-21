# import pandas as pd

# df = pd.read_csv(r'inputData/train_SQuAD_full.csv')
# print(df.shape)

# titles = ['Computer', 'Idealism', 'Dog', 'Himachal_Pradesh', 'Adolescence', 'Solar_energy']
# # Filter rows where column 'Category' == 'A'
# filtered_df = df[df['title'].isin(titles)]
# print(filtered_df.shape)

# # Randomly select N rows from filtered_df
# random_sample = filtered_df.sample(n=15000, random_state=42)  # random_state for reproducibility
# print(random_sample.shape)

# random_sample.to_csv('train_SQuAD.csv', index=False)

# print('File created!')

### ************************************
import pandas as pd

# Load full CSV
df = pd.read_csv('inputData/train_SQuAD_full.csv')

# Number of random samples desired
num_samples = 10000

# Sample random rows from entire dataset (without filtering)
sampled_df = df.sample(n=min(num_samples, len(df)), random_state=42)

# Extract unique titles from sampled rows
titles_sampled = sampled_df['title'].unique().tolist()

print(f"Number of sampled rows: {len(sampled_df)}")
print(f"Number of unique titles in sample: {len(titles_sampled)}")
print("Sampled titles:", titles_sampled[:20])  # print first 20 titles as example

# Optionally save sampled rows and titles to CSV or text file
sampled_df.to_csv('train_SQuAD.csv', index=False)
with open('inputData/sampled_titles.csv', 'w') as f:
    for title in titles_sampled:
        f.write(title + '\n')
