import pickle
import pandas as pd
import numpy as np

def load_data(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Erro ao carregar arquivo: {e}")
        data = None
    return data

def processing_data(data):
    try:
        result = []
        for syndrome_id, subjects in data.items():
            for subject_id, images in subjects.items():
                for image_id, embedding in images.items():
                    result.append([syndrome_id, subject_id, image_id, embedding])
        
        df = pd.DataFrame(result, columns=["syndrome_id", "subject_id", "image_id", "embedding"])
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
        
        if df.isnull().values.any():
            print("Aviso: Há valores ausentes nos dados!")
        
        return df
    except Exception as e:
        print(f"Erro ao processar dados: {e}")
        return None