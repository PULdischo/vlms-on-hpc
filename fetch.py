import typer
import srsly
from huggingface_hub import snapshot_download
from pathlib import Path
from datasets import Dataset

app = typer.Typer()


@app.command()
def model(
        repo_id: str
    ):
    """
    Downloads a model from Hugging Face Hub.
    """
    model_path = snapshot_download(repo_id=repo_id, repo_type="model")
    info_path = Path("model_info.json")
    model_info = srsly.read_json(info_path) if info_path.exists() else {}
    model_info[repo_id] = {"model_path": model_path}
    srsly.write_json(info_path, model_info)
    print(f"Model downloaded to: {model_path}")

@app.command()
def images(
    manifest_url: str,
    output_dir: str = typer.Option("img", "--output-dir", "-o", help="Directory to save images"),
):
    """
    Downloads images from a IIIF manifest.
    """
    from IIIF_download import iiif_tiles_download
    iiif_tiles_download(manifest_url, output_dir)

@app.command()
def to_hub(
    repo_id: str,
    public: bool = typer.Option(
        False, "--public", help="Make the dataset public (default: private)"
    ),
):
    """
    Uploads the results to Hugging Face Hub.
    """
    data = []
    info = srsly.read_json("img/info.json")
    images = Path('img').glob('*.jpg')
    for img_path in images:
        img = {}
        img["name"] = img_path.name
        img['manifest_url'] = info['url']
        img["image_url"] = info.get('images').get(img_path.name)
        md_path = img_path.with_suffix(".md")
        if md_path.exists():
            img["text"] = md_path.read_text()
        data.append(img)
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(repo_id, private=not public)
        

if __name__ == "__main__":
    app()
