import json
import re
from pathlib import Path
from typing import List, Dict


class ScientificChunker:
    def __init__(self, chunk_size=500, overlap=50):
        """
        chunk_size : max characters per chunk
        overlap    : characters to repeat between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_parsed_paper(self, parsed_json_path: str) -> List[Dict]:
        """
        Takes the JSON output from parser
        Returns a list of chunks with metadata
        """
        with open(parsed_json_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)

        filename = parsed["filename"]
        metadata = parsed["metadata"]
        all_chunks = []
        chunk_id = 0

        for section in parsed["sections"]:
            section_name = section["section"]
            text = section["text"]
            page = section["page"]

            # Skip empty or very short sections
            if len(text.strip()) < 50:
                continue

            # Split section into sentences first
            sentences = self._split_into_sentences(text)

            # Group sentences into chunks
            section_chunks = self._group_sentences(sentences)

            for chunk_text in section_chunks:
                all_chunks.append({
                    "chunk_id": f"{Path(filename).stem}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        "filename": filename,
                        "title": metadata.get("title", "Unknown"),
                        "author": metadata.get("author", "Unknown"),
                        "section": section_name,
                        "page": page
                    }
                })
                chunk_id += 1

        print(f"Created {len(all_chunks)} chunks from {filename}")
        return all_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences
        Handles abbreviations like et al. Fig. Dr. etc.
        """
        # Simple but effective sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Clean up
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences

    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """
        Groups sentences into chunks respecting chunk_size
        Adds overlap between consecutive chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence exceeds limit → save chunk, start new one
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Overlap — keep last few sentences for next chunk
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in overlap_sentences)

            current_chunk.append(sentence)
            current_size += sentence_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        Returns last N sentences for overlap
        Keeps going back until we have enough overlap characters
        """
        overlap_sentences = []
        total = 0

        for sentence in reversed(sentences):
            if total >= self.overlap:
                break
            overlap_sentences.insert(0, sentence)
            total += len(sentence)

        return overlap_sentences


if __name__ == "__main__":
    chunker = ScientificChunker(chunk_size=500, overlap=50)

    # Find all parsed JSONs in outputs/
    json_files = list(Path("outputs").glob("*_parsed.json"))

    if not json_files:
        print("No parsed JSONs found. Run parser.py first!")
    else:
        all_chunks = []
        for json_file in json_files:
            chunks = chunker.chunk_parsed_paper(str(json_file))
            all_chunks.extend(chunks)

        # Save all chunks
        output_path = Path("outputs") / "all_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        print(f"\nTotal chunks: {len(all_chunks)}")
        print(f"Saved to {output_path}")

        # Preview first chunk
        if all_chunks:
            print(f"\nSample chunk:")
            print(f"  ID      : {all_chunks[0]['chunk_id']}")
            print(f"  Section : {all_chunks[0]['metadata']['section']}")
            print(f"  Page    : {all_chunks[0]['metadata']['page']}")
            print(f"  Text    : {all_chunks[0]['text'][:100]}...")