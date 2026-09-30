from pathlib import Path
import re


FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|$)", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
INCLUDE_RE = re.compile(r"{%\s*include\s+([^\s%]+)\s*%}")
HEADING_ATTRIBUTE_RE = re.compile(r"^\{:\s*[^}]+\}$")


def load_markdown_files(folder_path):
    """Load all Markdown files in a folder."""
    folder = Path(folder_path)
    documents = []

    for file_path in sorted(folder.glob("*.md")):
        documents.append(
            {
                "source": file_path.name,
                "text": file_path.read_text(encoding="utf-8"),
            }
        )

    return documents


def resolve_includes(document, include_dir):
    """Replace Jekyll include tags with Markdown from the include directory."""
    include_dir = Path(include_dir)

    def replace_include(match):
        include_name = match.group(1)
        include_path = include_dir / Path(include_name).name

        if not include_path.exists():
            print(
                f"Warning: include file not found for {include_name} "
                f"in {document['source']}"
            )
            return match.group(0)

        return include_path.read_text(encoding="utf-8").strip()

    return {
        **document,
        "text": INCLUDE_RE.sub(replace_include, document["text"]),
    }


def clean_document(document):
    """Remove YAML front matter while preserving Markdown content."""
    cleaned_text = FRONT_MATTER_RE.sub("", document["text"]).strip()

    return {
        **document,
        "text": cleaned_text,
    }


def _clean_heading_text(heading_text):
    return heading_text.strip().strip("#").strip()


def _section_name(heading_stack):
    section_parts = [
        heading_stack[level]
        for level in (2, 3, 4)
        if heading_stack.get(level)
    ]

    if section_parts:
        return " > ".join(section_parts)
    if heading_stack.get(1):
        return heading_stack[1]
    return "Overview"


def _has_meaningful_body(lines):
    for line in lines:
        stripped_line = line.strip()
        if (
            stripped_line
            and not HEADING_RE.match(stripped_line)
            and not HEADING_ATTRIBUTE_RE.match(stripped_line)
        ):
            return True

    return False


def _add_chunk(chunks, document, section, lines):
    if not _has_meaningful_body(lines):
        return

    chunks.append(
        {
            "source": document["source"],
            "section": section,
            "text": "\n".join(lines).strip(),
        }
    )


def chunk_document(document):
    """Split a cleaned Markdown document into heading-based chunks."""
    chunks = []
    current_lines = []
    current_section = None
    heading_stack = {}

    for line in document["text"].splitlines():
        heading_match = HEADING_RE.match(line)

        if heading_match:
            level = len(heading_match.group(1))
            heading_text = _clean_heading_text(heading_match.group(2))

            if level == 1:
                heading_stack[1] = heading_text
                for child_level in (2, 3, 4):
                    heading_stack.pop(child_level, None)
            elif level in (2, 3, 4):
                if current_lines:
                    _add_chunk(
                        chunks,
                        document,
                        current_section or _section_name(heading_stack),
                        current_lines,
                    )
                heading_stack[level] = heading_text
                for child_level in range(level + 1, 5):
                    heading_stack.pop(child_level, None)
                current_section = _section_name(heading_stack)
                current_lines = [line]
                continue

        if not current_lines and current_section is None:
            current_section = _section_name(heading_stack)

        current_lines.append(line)

    if current_lines:
        _add_chunk(
            chunks,
            document,
            current_section or _section_name(heading_stack),
            current_lines,
        )

    return [chunk for chunk in chunks if chunk["text"]]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    corpus_path = project_root / "data" / "raw" / "mimic_docs"
    include_path = project_root / "data" / "support" / "mimic_includes"

    documents = load_markdown_files(corpus_path)
    resolved_documents = [
        resolve_includes(document, include_path)
        for document in documents
    ]
    cleaned_documents = [clean_document(document) for document in resolved_documents]
    chunks = [
        chunk
        for document in cleaned_documents
        for chunk in chunk_document(document)
    ]
    chunk_lengths = [len(chunk["text"]) for chunk in chunks]
    sorted_chunks = sorted(chunks, key=lambda chunk: len(chunk["text"]))
    sorted_lengths = sorted(chunk_lengths)
    midpoint = len(sorted_lengths) // 2

    average_length = sum(chunk_lengths) / len(chunk_lengths)
    if len(sorted_lengths) % 2 == 0:
        median_length = (sorted_lengths[midpoint - 1] + sorted_lengths[midpoint]) / 2
    else:
        median_length = sorted_lengths[midpoint]

    def print_chunk_preview(chunk):
        text_preview = chunk["text"][:200].replace("\n", " ")
        print(f"- source={chunk['source']}")
        print(f"  section={chunk['section']}")
        print(f"  text_length={len(chunk['text'])}")
        print(f"  preview={text_preview}")

    print(f"Total number of chunks: {len(chunks)}")
    print(f"Average chunk text length: {average_length:.2f}")
    print(f"Median chunk text length: {median_length:.2f}")

    print("10 shortest chunks:")
    for chunk in sorted_chunks[:10]:
        print_chunk_preview(chunk)

    print("10 longest chunks:")
    for chunk in sorted_chunks[-10:]:
        print_chunk_preview(chunk)

    print('Full chunk whose section contains "Association with UB-04 billing codes":')
    for chunk in chunks:
        if "association with ub-04 billing codes" in chunk["section"].lower():
            print(f"source={chunk['source']}")
            print(f"section={chunk['section']}")
            print(f"text_length={len(chunk['text'])}")
            print(chunk["text"])
