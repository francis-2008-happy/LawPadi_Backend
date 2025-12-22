from pathlib import Path


def extract_metadata(file_path: str):
    path = Path(file_path)
    # e.g. 'Constitution_of_the_Federal_Republic_of_Nigeria_1999' -> 'Constitution of the Federal Republic of Nigeria 1999'
    name = path.stem.replace("_", " ")

    metadata = {
        "law_name": name,
        "document_type": path.parent.name,
        "document_type": path.parent.name,  # 'Acts' or 'Cases'
        "court": None,
        "year": None,
        "section_or_case": name,
    }

    # --- Try to find year and court from path ---
    for token in path.stem.split("_"):
        if token.isdigit() and len(token) == 4:
            metadata["year"] = token

    if "supreme" in file_path.lower():
        metadata["court"] = "Supreme Court of Nigeria"
        metadata["court"] = "Supreme Court"
    elif "appeal" in file_path.lower():
        metadata["court"] = "Court of Appeal"

    # --- Construct a display name ---
    display_name = name
    if metadata["document_type"].lower() == "cases" and metadata["court"] and metadata["year"]:
        display_name = f"{name} ({metadata['court']}, {metadata['year']})"

    metadata["section_or_case"] = display_name

    return metadata
