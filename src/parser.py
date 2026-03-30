import fitz  # pymupdf
import os
from pathlib import Path
from PIL import Image
import io
import json
from dotenv import load_dotenv

load_dotenv()

class ScientificPaperParser:
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

    def parse_pdf(self, pdf_path: str) -> dict:
        """
        Extracts text, figures, tables and metadata from a scientific PDF
        Returns a structured dict with all extracted content
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        
        result = {
            "filename": pdf_path.name,
            "total_pages": len(doc),
            "metadata": self._extract_metadata(doc),
            "sections": [],
            "figures": [],
            "tables": []
        }

        print(f"Parsing {pdf_path.name} — {len(doc)} pages")

        for page_num, page in enumerate(doc):
            print(f"  Processing page {page_num + 1}/{len(doc)}...")
            
            # Extract text with structure
            text_blocks = self._extract_text_blocks(page, page_num)
            result["sections"].extend(text_blocks)
            
            # Extract figures
            figures = self._extract_figures(page, page_num, pdf_path.stem)
            result["figures"].extend(figures)

        doc.close()
        
        # Save parsed result as JSON
        output_path = self.output_dir / f"{pdf_path.stem}_parsed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Done! Saved to {output_path}")
        return result

    def _extract_metadata(self, doc) -> dict:
        meta = doc.metadata
        return {
            "title": meta.get("title", "Unknown"),
            "author": meta.get("author", "Unknown"),
            "subject": meta.get("subject", ""),
            "keywords": meta.get("keywords", "")
        }

    def _extract_text_blocks(self, page, page_num: int) -> list:
        blocks = []
        text_dict = page.get_text("dict")
        
        current_section = ""
        current_text = []

        for block in text_dict["blocks"]:
            if block["type"] == 0:  # text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]
                        is_bold = "bold" in span["font"].lower()

                        if not text:
                            continue

                        # Detect section headers (larger font or bold)
                        if font_size > 11 and is_bold and len(text) < 100:
                            # Save previous section
                            if current_text:
                                blocks.append({
                                    "section": current_section,
                                    "text": " ".join(current_text),
                                    "page": page_num + 1,
                                    "type": "text"
                                })
                                current_text = []
                            current_section = text
                        else:
                            current_text.append(text)

        if current_text:
            blocks.append({
                "section": current_section,
                "text": " ".join(current_text),
                "page": page_num + 1,
                "type": "text"
            })

        return blocks

    def _extract_figures(self, page, page_num: int, pdf_stem: str) -> list:
        figures = []
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Skip tiny images (logos, icons)
                img_pil = Image.open(io.BytesIO(image_bytes))
                width, height = img_pil.size
                if width < 100 or height < 100:
                    continue

                # Save figure
                fig_filename = f"{pdf_stem}_page{page_num+1}_fig{img_idx+1}.{image_ext}"
                fig_path = self.output_dir / "figures" / fig_filename
                with open(fig_path, "wb") as f:
                    f.write(image_bytes)

                figures.append({
                    "filename": fig_filename,
                    "page": page_num + 1,
                    "width": width,
                    "height": height,
                    "type": "figure"
                })

            except Exception as e:
                print(f"  Could not extract image {img_idx}: {e}")

        return figures

