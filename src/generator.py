import os
from dotenv import load_dotenv
from google import genai
from typing import List, Dict

# Load environment variables
load_dotenv()


class AnswerGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-flash-latest"

    def generate(self, query: str, chunks: List[Dict]) -> Dict:
        if not chunks:
            return self._format_response(
                answer="No relevant chunks found.",
                citations=[],
                chunks_used=0
            )

        context = self._build_context(chunks)
        prompt = self._build_prompt(query, context)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            answer = response.text.strip()

        except Exception as e:
            return self._format_response(
                answer=f"LLM request failed: {str(e)}",
                citations=[],
                chunks_used=0
            )

        citations = self._extract_citations(chunks)

        return self._format_response(
            answer=answer,
            citations=citations,
            chunks_used=len(chunks)
        )

    def _build_context(self, chunks: List[Dict]) -> str:
        context_parts = []

        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            source = (
                f"[Source {i+1}] "
                f"{meta['title']} | {meta['section']} | Page {meta['page']}"
            )
            context_parts.append(f"{source}\n{chunk['text']}")

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are an expert scientific research assistant.

STRICT RULES:
- Use ONLY the given context
- Cite using [Source N]
- Do NOT hallucinate
- Be precise and technical

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:"""

    def _extract_citations(self, chunks: List[Dict]) -> List[Dict]:
        return [
            {
                "source": i + 1,
                "title": chunk["metadata"]["title"],
                "section": chunk["metadata"]["section"],
                "page": chunk["metadata"]["page"],
                "file": chunk["metadata"]["filename"]
            }
            for i, chunk in enumerate(chunks)
        ]

    def _format_response(self, answer: str, citations: List[Dict], chunks_used: int) -> Dict:
        return {
            "status": "success" if citations else "no_data",
            "answer": answer,
            "metadata": {
                "chunks_used": chunks_used,
                "num_sources": len(citations)
            },
            "citations": citations
        }


# ---------------- TEST ---------------- #

if __name__ == "__main__":
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

    from pprint import pprint
    pprint(result)