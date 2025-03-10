import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def exploratory_analysis(df):
    print("Resumo dos dados:")
    print(df.head())
    print(f"Total de registros: {len(df)}")
    print(f"Número de síndromes únicas: {df['syndrome_id'].nunique()}")
    print(f"Número de sujeitos únicos: {df['subject_id'].nunique()}")
    print(f"Número de imagens únicas: {df['image_id'].nunique()}")
    
    plt.figure(figsize=(10, 5))
    syndrome_counts = df['syndrome_id'].value_counts()
    sns.barplot(x=syndrome_counts.index, y=syndrome_counts.values, hue=syndrome_counts.index, palette='viridis', legend=False)
    plt.xlabel("ID Síndrome")
    plt.ylabel("Número de Imagens")
    plt.title("Distribuição de Imagens por Síndrome")
    plt.xticks(rotation=45)
    plt.show()
    
    embeddings_matrix = np.stack(df["embedding"].values)
    print("Estatísticas dos Embeddings:")
    print(f"Média: {np.mean(embeddings_matrix, axis=0)[:5]} ...")
    print(f"Variância: {np.var(embeddings_matrix, axis=0)[:5]} ...")
    
    plt.hist(np.linalg.norm(embeddings_matrix, axis=1), bins=30, alpha=0.7, color='blue')
    plt.title("Distribuição das Magnitudes dos Embeddings")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequência")
    plt.show()