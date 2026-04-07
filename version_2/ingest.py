import glob
import logging
import os
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from scripts import llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _extract_text_block(value: str) -> str:
    lines = [_clean_text(line) for line in value.splitlines()]
    return "\n".join([line for line in lines if line])


def _slide_header(filename: str, slide_index: int) -> str:
    return f"Filename: {filename}\nSlide Number: {slide_index}"


def _pick_slide_title(text_blocks: list[str]) -> str:
    for block in text_blocks:
        for line in block.splitlines():
            cleaned_line = _clean_text(line)
            if cleaned_line:
                return cleaned_line
    return ""


def _make_document(
    file_path: str,
    filename: str,
    slide_index: int,
    content_type: str,
    content: str,
    slide_title: str = "",
    element_index: int | None = None,
) -> Document:
    metadata = {
        "source": file_path,
        "filename": filename,
        "page": str(slide_index),
        "content_type": content_type,
        "slide_title": slide_title,
    }
    if element_index is not None:
        metadata["element_index"] = element_index
    return Document(page_content=content, metadata=metadata)


def _build_text_documents(
    file_path: str,
    filename: str,
    slide_index: int,
    slide_title: str,
    text_blocks: list[str],
) -> list[Document]:
    documents: list[Document] = []
    if slide_title:
        title_text = (
            f"{_slide_header(filename, slide_index)}\n"
            "Content Type: Title\n"
            f"Slide Title: {slide_title}\n"
            f"Title:\n{slide_title}"
        )
        documents.append(
            _make_document(
                file_path=file_path,
                filename=filename,
                slide_index=slide_index,
                content_type="title",
                content=title_text,
                slide_title=slide_title,
            )
        )

    for block_index, block in enumerate(text_blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        content_type = "comments" if any("comment" in line.lower() for line in lines) else "text_section"
        block_text = (
            f"{_slide_header(filename, slide_index)}\n"
            f"Content Type: {content_type}\n"
            f"Slide Title: {slide_title}\n"
            f"Section Index: {block_index}\n"
            + "\n".join(lines)
        )
        documents.append(
            _make_document(
                file_path=file_path,
                filename=filename,
                slide_index=slide_index,
                content_type=content_type,
                content=block_text,
                slide_title=slide_title,
                element_index=block_index,
            )
        )

    return documents


def _extract_chart_text(shape, filename: str, slide_index: int, chart_index: int) -> str | None:
    try:
        chart = shape.chart
        chart_parts = [
            _slide_header(filename, slide_index),
            "Content Type: Chart",
            f"Chart Index: {chart_index}",
        ]

        chart_title = None
        if chart.has_title and chart.chart_title and chart.chart_title.text_frame:
            chart_title = _clean_text(chart.chart_title.text_frame.text)
        if chart_title:
            chart_parts.append(f"Chart Title: {chart_title}")

        category_labels = []
        try:
            for category in chart.plots[0].categories:
                label = _clean_text(str(category.label))
                if label:
                    category_labels.append(label)
        except Exception:
            category_labels = []

        if category_labels:
            chart_parts.append("Categories: " + " | ".join(category_labels))

        series_lines = []
        for series in chart.series:
            series_name = _clean_text(getattr(series, "name", "") or "Unnamed Series")
            values = list(series.values)
            points = []
            for idx, value in enumerate(values):
                label = category_labels[idx] if idx < len(category_labels) else f"Point {idx + 1}"
                points.append(f"{label}: {value}")
            series_lines.append(f"{series_name} -> " + "; ".join(points))

        if series_lines:
            chart_parts.append("Series Data:\n" + "\n".join(series_lines))

        return "\n".join(chart_parts) if len(chart_parts) > 3 else None
    except Exception as exc:
        logging.warning(f"Chart extraction failed on slide {slide_index}: {exc}")
        return None


def build_vector_database():
    target_folder = "presentations"
    os.makedirs(target_folder, exist_ok=True)
    ppt_files = glob.glob(os.path.join(target_folder, "*.pptx"))

    if not ppt_files:
        logging.warning("No .pptx files found. Please add some and run again.")
        return

    raw_documents: list[Document] = []

    for file_path in ppt_files:
        actual_filename = os.path.basename(file_path)
        logging.info(f"Extracting: {actual_filename}")

        try:
            prs = Presentation(file_path)
            for slide_index, slide in enumerate(prs.slides, start=1):
                text_blocks = []
                table_blocks = []
                chart_blocks = []
                table_counter = 0
                chart_counter = 0

                for shape in slide.shapes:
                    if shape.has_text_frame and shape.text.strip():
                        cleaned_text = _extract_text_block(shape.text)
                        if cleaned_text:
                            text_blocks.append(cleaned_text)
                        continue

                    slide_title = _pick_slide_title(text_blocks)

                    if shape.has_table:
                        table_rows = []
                        for row in shape.table.rows:
                            row_data = [_clean_text(cell.text) for cell in row.cells]
                            table_rows.append(" | ".join(row_data))
                        if table_rows:
                            table_counter += 1
                            table_text = (
                                f"{_slide_header(actual_filename, slide_index)}\n"
                                "Content Type: Table\n"
                                f"Slide Title: {slide_title}\n"
                                f"Table Index: {table_counter}\n"
                                "Table Data:\n"
                                + "\n".join(table_rows)
                            )
                            table_blocks.append(table_text)
                            raw_documents.append(
                                _make_document(
                                    file_path=file_path,
                                    filename=actual_filename,
                                    slide_index=slide_index,
                                    content_type="table",
                                    content=table_text,
                                    slide_title=slide_title,
                                    element_index=table_counter,
                                )
                            )
                        continue

                    if getattr(shape, "has_chart", False):
                        chart_counter += 1
                        chart_text = _extract_chart_text(
                            shape=shape,
                            filename=actual_filename,
                            slide_index=slide_index,
                            chart_index=chart_counter,
                        )
                        if chart_text:
                            chart_blocks.append(chart_text)
                            raw_documents.append(
                                _make_document(
                                    file_path=file_path,
                                    filename=actual_filename,
                                    slide_index=slide_index,
                                    content_type="chart",
                                    content=chart_text,
                                    slide_title=slide_title,
                                    element_index=chart_counter,
                                )
                            )

                slide_title = _pick_slide_title(text_blocks)
                raw_documents.extend(
                    _build_text_documents(
                        file_path=file_path,
                        filename=actual_filename,
                        slide_index=slide_index,
                        slide_title=slide_title,
                        text_blocks=text_blocks,
                    )
                )

                slide_sections = []
                if text_blocks:
                    slide_sections.append("Slide Text:\n" + "\n".join(text_blocks))
                if table_blocks:
                    slide_sections.append("Tables:\n" + "\n\n".join(table_blocks))
                if chart_blocks:
                    slide_sections.append("Charts:\n" + "\n\n".join(chart_blocks))

                if slide_sections:
                    slide_summary = (
                        f"{_slide_header(actual_filename, slide_index)}\n"
                        "Content Type: Slide Summary\n"
                        f"Slide Title: {slide_title}\n\n"
                        + "\n\n".join(slide_sections)
                    )
                    raw_documents.append(
                        _make_document(
                            file_path=file_path,
                            filename=actual_filename,
                            slide_index=slide_index,
                            content_type="slide_summary",
                            content=slide_summary,
                            slide_title=slide_title,
                        )
                    )

        except Exception as exc:
            logging.error(f"Failed to process {file_path}: {exc}")

    logging.info(f"Extracted {len(raw_documents)} total retrievable items. Starting chunking...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    chunked_docs = text_splitter.split_documents(raw_documents)
    logging.info(f"Created {len(chunked_docs)} chunks.")

    db_directory = "./chroma_db_simple"
    logging.info("Saving to Chroma DB...")
    if os.path.isdir(db_directory):
        shutil.rmtree(db_directory)

    Chroma.from_documents(
        documents=chunked_docs,
        embedding=llm.embeddings,
        persist_directory=db_directory,
    )

    logging.info("Simple database built successfully!")


if __name__ == "__main__":
    build_vector_database()
