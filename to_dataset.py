import pymupdf
import datasets
import typer 
from tqdm import tqdm
from pathlib import Path



def main(pdf_path: str = typer.Argument(..., help="Path to the PDF files"), dataset_name: str = typer.Argument(..., help="Name of the dataset to be created")):
    """
    Convert PDF files to a dataset with images and text.
    """
    data = []
    if not Path(pdf_path).exists():
        print(f"please provide a valid path to the PDF files")
        return
    
    pdfs = Path(pdf_path).glob('*')
   
    for pdf in pdfs:
        try:
            doc = pymupdf.open(pdf)
            for i, page in tqdm(enumerate(doc)):  # iterate through the pages
                pix = page.get_pixmap(dpi=300)  
                img = pix.pil_image()
                data.append({
                    "image": img,
                    "text": "",
                    "page": i + 1,
                    "pdf_name": pdf.stem
                })
        except Exception as e:
            print(f"Error opening {pdf}: {e}")
            continue
        

    dataset = datasets.Dataset.from_list(data)
    #dataset = dataset.cast_column("image", datasets.Image())
    dataset.push_to_hub(dataset_name)

if __name__ == "__main__":
    typer.run(main)