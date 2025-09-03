import os
import pandas as pd

def save_results(model_name, results):
  data = {
      'Model': model_name,
      'Loss': results.get('Loss', None),         # Loss
      'Accuracy': results.get('Accuracy', None),     # Accuracy
      'Precision': results.get('Precision', None),    # Precision
      'Recall': results.get('Recall', None),       # Recall
  }

  df = pd.DataFrame([data])
  df = df.fillna("null")

  # Nome do arquivo CSV
  results_dir = 'results'
  csv_file = os.path.join(results_dir, 'evaluation.csv')
  
  os.makedirs(results_dir, exist_ok=True)

  if os.path.exists(csv_file):
      return df.to_csv(csv_file, mode='a', header=False, index=False)
  df.to_csv(csv_file, mode='w', index=False)