"""Text splitter with recursive boundary support and sliding window overlap."""

from typing import List
from simple_rag.models import Document, Chunk


class RecursiveCharacterTextSplitter:
    """Splits text by recursively looking at separators to keep chunks semantically cohesive."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text into pieces smaller than chunk_size using available separators."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    other_splits = self._split_text(s, new_separators)
                    good_splits.extend(other_splits)
                else:
                    # Character slice if no more separators exist
                    good_splits.extend(
                        [s[j : j + self.chunk_size] for j in range(0, len(s), self.chunk_size)]
                    )

        # Merge splits up to chunk_size with chunk_overlap
        return self._merge_splits(good_splits, separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for s in splits:
            s_len = len(s)
            extra = len(separator) if current_doc else 0

            if total + s_len + extra > self.chunk_size and current_doc:
                doc = separator.join(current_doc).strip()
                if doc:
                    docs.append(doc)

                # Keep overlap pieces from the end of current_doc
                while total > self.chunk_overlap and len(current_doc) > 1:
                    popped = current_doc.pop(0)
                    total -= len(popped) + (len(separator) if current_doc else 0)

            current_doc.append(s)
            total += s_len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def split_document(self, document: Document) -> List[Chunk]:
        """Split a single document into Chunk objects with metadata."""
        raw_chunks = self._split_text(document.content, self.separators)
        chunks: List[Chunk] = []

        for idx, text in enumerate(raw_chunks):
            chunk_metadata = {
                **document.metadata,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                "char_length": len(text),
            }
            chunk = Chunk(
                content=text,
                document_id=document.id,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split multiple documents."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
