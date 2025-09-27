import pandas as pd
import seaborn as sns
import os

def download_titanic_data():
    """Seaborn'dan Titanic veri setini indir"""
    
    # Veri dizinlerini oluştur
    os.makedirs('data/raw', exist_ok=True)
    
    # Seaborn'dan Titanic veri setini yükle
    df = sns.load_dataset('iris')
    
    # Ham veriyi kaydetme
    df.to_csv('data/raw/iris.csv', index=False)
    
    print("✅ iris veri seti indirildi: data/raw/iris.csv")
    print(f"Veri boyutu: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")
    print(f"Eksik değerler:\n{df.isnull().sum()}")
    
    return df

if __name__ == "__main__":
    download_titanic_data()