"""Document loaders for ingesting text and markdown files into Simple RAG."""

from pathlib import Path
from typing import List, Union
import json
import os

from simple_rag.models import Document


class DocumentLoader:
    """Loads files or directories into Document objects."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv"}

    @classmethod
    def load_file(cls, file_path: Union[str, Path]) -> Document:
        """Load a single file into a Document."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{path.suffix}'. Supported: {cls.SUPPORTED_EXTENSIONS}"
            )

        content = ""
        metadata = {
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
        }

        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, str):
                    content = data
                else:
                    content = json.dumps(data, indent=2)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        metadata["line_count"] = content.count("\n") + 1
        metadata["char_count"] = len(content)

        return Document(content=content, metadata=metadata)

    @classmethod
    def load_directory(
        cls,
        dir_path: Union[str, Path],
        recursive: bool = True,
    ) -> List[Document]:
        """Recursively loads all supported documents in a directory."""
        path = Path(dir_path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        documents: List[Document] = []
        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                try:
                    doc = cls.load_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return documents
