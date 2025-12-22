from pathlib import Path
from pypdf import PdfReader


def load_documents(data_dir="data"):
    documents = []
    files = [f for f in Path(data_dir).rglob("*") if f.suffix.lower() in [".pdf", ".txt"]]
    total_files = len(files)
    print(f"Found {total_files} files to process.")

    for i, file in enumerate(files, 1):
        print(f"[{i}/{total_files}] Processing {file.name}...", end=" ", flush=True)
        try:
            if file.suffix.lower() == ".pdf":
                reader = PdfReader(file)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                documents.append({"text": text, "path": str(file)})

            elif file.suffix.lower() == ".txt":
                documents.append(
                    {"text": file.read_text(encoding="utf-8"), "path": str(file)}
                )
            print("Done.")
        except Exception as e:
            print(f"Failed: {e}")

    return documents
