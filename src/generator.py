import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict

load_dotenv()

class AnswerGenerator:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        print("Gemini loaded!")

    def generate(self, query: str, chunks: List[Dict]) -> Dict:
        """
        Takes query + retrieved chunks
        Returns cited answer
        """
        if not chunks:
            return {
                "answer": "No relevant chunks found.",
                "citations": []
            }

        # Build context from chunks
        context = self._build_context(chunks)

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Generate
        response = self.model.generate_content(prompt)
        answer = response.text

        # Extract citations
        citations = self._extract_citations(chunks)

        return {
            "answer": answer,
            "citations": citations,
            "chunks_used": len(chunks)
        }

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Formats chunks into readable context for Gemini
        Each chunk labeled with its source
        """
        context_parts = []

        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            source = f"[Source {i+1}] Paper: {meta['title']} | Section: {meta['section']} | Page: {meta['page']}"
            context_parts.append(f"{source}\n{chunk['text']}")

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are an expert scientific research assistant.

Answer the following question using ONLY the provided context from research papers.
Always cite your sources using [Source N] notation.
If the context doesn't contain enough information, say so clearly.
Be precise, technical, and concise.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER (with citations):"""

    def _extract_citations(self, chunks: List[Dict]) -> List[Dict]:
        citations = []
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            citations.append({
                "source_num": i + 1,
                "title": meta["title"],
                "section": meta["section"],
                "page": meta["page"],
                "filename": meta["filename"]
            })
        return citations


if __name__ == "__main__":
    # Test with mock chunks
    mock_chunks = [
        {
            "text": "Transformers use self-attention mechanisms to process sequential data in parallel rather than recurrently.",
            "metadata": {
                "title": "Attention is All You Need",
                "section": "Introduction",
                "page": 1,
                "filename": "vaswani2017.pdf"
            }
        },
        {
            "text": "The model achieves state of the art results on English to German translation with a BLEU score of 28.4.",
            "metadata": {
                "title": "Attention is All You Need",
                "section": "Results",
                "page": 7,
                "filename": "vaswani2017.pdf"
            }
        }
    ]

    generator = AnswerGenerator()
    result = generator.generate(
        query="How do transformers process sequential data?",
        chunks=mock_chunks
    )

    print("\nAnswer:")
    print(result["answer"])
    print("\nCitations:")
    for c in result["citations"]:
        print(f"  [Source {c['source_num']}] {c['title']} — {c['section']}, Page {c['page']}")