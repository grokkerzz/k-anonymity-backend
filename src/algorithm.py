import pandas as pd

def k_anonymity():
    pass


def show_schema(filename):
    df = pd.read_csv(filename)
    return {'schema': list(df.columns)}
