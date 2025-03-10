import pandas as pd
import numpy as np

def preprocess_data(data):
    """
    Transforma a estrutura hierárquica do dicionário em um DataFrame.
    :param data: Dicionário contendo os embeddings.
    :return: DataFrame com os dados organizados.
    """
    records = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                records.append([syndrome_id, subject_id, image_id, embedding])
    
    df = pd.DataFrame(records, columns=["syndrome_id", "subject_id", "image_id", "embedding"])
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    return df

if __name__ == "__main__":
    from data_loader import load_data
    
    file_path = "mini_gm_public_v0.1.p"  
    data = load_data(file_path)
    df = preprocess_data(data)
    print("Dados processados com sucesso!")
    print(df.head())
