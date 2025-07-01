import typer
from huggingface_hub import snapshot_download
from iiif_download import IIIFManifest

app = typer.Typer()


@app.command()
def model(repo_id: str = "nanonets/Nanonets-OCR-s"):
    """
    Downloads a model from Hugging Face Hub.
    """
    snapshot_download(repo_id=repo_id, repo_type="model")


@app.command()
def images(
    manifest_url: str,
):
    """
    Downloads images from a IIIF manifest.
    """
    manifest = IIIFManifest(manifest_url)
    manifest.download()


if __name__ == "__main__":
    app()
