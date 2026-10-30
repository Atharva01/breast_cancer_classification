from pathlib import Path
from sklearn.datasets import load_breast_cancer

def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True,exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    output_path = output_dir / "breast_cancer.csv"
    df = dataset['frame']
    df.to_csv(output_path,index=False)
    
    print(f"Dataset saved to:{output_dir}")
    print(f"Shape:{df.shape}")
   
if __name__ == "__main__":
    main()