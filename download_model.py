from transformers import pipeline
import os

def download():
    model_name = "prajjwal1/bert-tiny"
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(current_dir, "model_cache")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    print(f"🚀 Downloading {model_name}...")
    print(f"📂 Target folder: {cache_dir}")
    
    try:
        # This will download the model weights and config into the local model_cache folder
        pipeline("fill-mask", model=model_name, model_kwargs={"cache_dir": cache_dir})
        print("\n✅ Download Complete! The model is now bundled in the 'model_cache' folder.")
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")

if __name__ == "__main__":
    download()
