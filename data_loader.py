import pickle

def load_data(file_path):
    """
    Carrega os dados do arquivo pickle.
    :param file_path: Caminho do arquivo pickle.
    :return: Dados carregados como um dicionário.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    file_path = "mini_gm_public_v0.1.p"
    data = load_data(file_path)
    print(f"Dados carregados com sucesso! {len(data)} síndromes identificadas.")