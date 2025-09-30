import matplotlib.pyplot as plt
import pandas as pd
from datasets import get_dataset_config_names, load_dataset

domains = get_dataset_config_names('subjqa')
subjqa = load_dataset('megagonlabs/subjqa', name='electronics')

dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}

for split, df in dfs.items():
    print(f"Number of question in {split}: {df['id'].nunique()}")

qa_cols = ['title', 'question', 'answers.text', 'answers.answer_start', 'context']
sample_df = dfs['train'][qa_cols].sample(2, random_state=7)
print(sample_df.to_string())

start_idx = sample_df['answers.answer_start'].iloc[0][0]
end_idx = start_idx + len(sample_df['answers.text'].iloc[0][0])
print(sample_df['context'].iloc[0][start_idx:end_idx])

counts = {}
question_types = ['What', 'How', 'Is', 'Does', 'Do', 'Was', 'Where', 'Why']

for q in question_types:
    counts[q] = dfs['train']['question'].str.startswith(q).value_counts()[True]

pd.Series(counts).sort_values().plot.barh()
plt.title('Frequency of Question Types')
plt.show()

for question_type in ['How', 'What', 'Is']:
    for question in (
            dfs['train'][dfs['train'].question.str.startswith(question_type)].sample(n=3, random_state=42)['question']
    ):
        print(question)
