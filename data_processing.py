import pickle
import pandas as pd
import numpy as np

def load_data(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None

def process_data(data):
    try:
        records = []
        for syndrome_id, subjects in data.items():
            for subject_id, images in subjects.items():
                for image_id, embedding in images.items():
                    records.append([syndrome_id, subject_id, image_id, np.array(embedding)])
        
        df = pd.DataFrame(records, columns=["syndrome_id", "subject_id", "image_id", "embedding"])
        
        if df.isnull().values.any():
            print("Aviso: Existem valores ausentes nos dados!")
        
        return df
    except Exception as e:
        print(f"Erro ao processar os dados: {e}")
        return None
