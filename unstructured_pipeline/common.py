import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, Iterator


csv.field_size_limit(10_000_000)

STANDARD_COLUMNS = [
    "Disease Name",
    "Symptom Description",
    "Drug Name",
    "Precautions",
    "Patient Question",
    "Doctor Response",
]

SOURCE_CONFIGS = [
    {
        "input_name": "nhs_conditions.csv",
        "source": "NHS",
        "source_type": "condition",
        "mapper": "map_nhs_conditions",
    },
    {
        "input_name": "nhs_medicines.csv",
        "source": "NHS",
        "source_type": "medicine",
        "mapper": "map_nhs_medicines",
    },
    {
        "input_name": "nhs_symptoms.csv",
        "source": "NHS",
        "source_type": "symptom",
        "mapper": "map_nhs_symptoms",
    },
    {
        "input_name": "nhs_treatments.csv",
        "source": "NHS",
        "source_type": "treatment",
        "mapper": "map_nhs_treatments",
    },
    {
        "input_name": "who_health_topics.csv",
        "source": "WHO",
        "source_type": "health_topic",
        "mapper": "map_who_topics",
    },
]


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_row_keys(row: Dict[str, str]) -> Dict[str, str]:
    return {str(key).lstrip("\ufeff"): value for key, value in row.items()}


def clean_name(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\s*-\s*Other brand names:.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def join_unique(parts: list[str], sep: str = "\n\n") -> str:
    seen = set()
    out = []
    for part in parts:
        part = normalize_text(part)
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return sep.join(out)


def strip_leading_heading(text: str) -> str:
    text = normalize_text(text)
    return re.sub(r"^##+\s*[^\n]+", "", text).strip()


def parse_markdown_sections(content: str) -> Dict[str, str]:
    content = normalize_text(content)
    if not content:
        return {}

    sections = {}
    current_title = "intro"
    buffer = []

    for raw_line in content.split("\n"):
        line = raw_line.strip()
        if not line:
            if buffer and buffer[-1] != "":
                buffer.append("")
            continue
        if line.startswith("#"):
            if buffer:
                sections[current_title] = join_unique(["\n".join(buffer)], sep="\n\n")
            current_title = re.sub(r"^#+\s*", "", line).strip().lower()
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections[current_title] = join_unique(["\n".join(buffer)], sep="\n\n")

    return sections


def select_sections(sections: Dict[str, str], keywords: list[str]) -> str:
    matches = []
    for title, text in sections.items():
        haystack = f"{title}\n{text}".lower()
        if any(keyword in haystack for keyword in keywords):
            matches.append(text)
    return join_unique(matches)


def first_paragraph(text: str) -> str:
    text = strip_leading_heading(text)
    if not text:
        return ""
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph and not paragraph.startswith("#"):
            return paragraph
    return " ".join(split_sentences(text)[:3])


def first_sentences_matching(text: str, keywords: list[str], limit: int = 3) -> str:
    matches = []
    for sentence in split_sentences(text):
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            matches.append(sentence)
        if len(matches) >= limit:
            break
    return " ".join(matches)


def extract_disease_targets_from_medicine(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"used to treat ([^.]+)\.",
        r"used to help treat ([^.]+)\.",
        r"medicine for ([^.]+)\.",
        r"used for ([^.]+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            target = match.group(1)
            target = re.sub(r"\b(certain|some|different|the symptoms of)\b", "", target, flags=re.IGNORECASE)
            return re.sub(r"\s+", " ", target).strip(" ,;:")
    return ""


def derive_condition_name_from_treatment(title: str) -> str:
    title = clean_name(title)
    lowered = title.lower()
    suffixes = [
        " screening",
        " treatment",
        " treatments",
        " surgery",
        " therapies",
        " therapy",
        " test",
        " tests",
        " procedure",
        " procedures",
    ]
    for suffix in suffixes:
        if lowered.endswith(suffix):
            return title[: -len(suffix)].strip()

    match = re.search(r"for ([A-Za-z0-9 ,()'/-]+)$", title, flags=re.IGNORECASE)
    return match.group(1).strip() if match else title


def guess_wikipedia_type(entry: str, category: str, definition: str) -> str:
    text = f"{entry} {category} {definition}".lower()
    drug_keywords = [
        "is a drug",
        "is a medication",
        "is an antiviral",
        "is an antibiotic",
        "is a medicine",
        "used to treat",
        "pharmaceutical",
    ]
    return "drug" if any(keyword in text for keyword in drug_keywords) else "disease"


def base_record() -> Dict[str, str]:
    return {column: "" for column in STANDARD_COLUMNS}


def map_nhs_conditions(row: Dict[str, str]) -> Dict[str, str]:
    sections = parse_markdown_sections(row.get("content", ""))
    record = base_record()
    record["Disease Name"] = clean_name(row.get("name") or row.get("title"))
    record["Symptom Description"] = select_sections(sections, ["symptoms", "signs"])
    record["Precautions"] = join_unique(
        [
            select_sections(
                sections,
                ["how to lower your risk", "prevent", "prevention", "do", "don't", "dont", "advice", "call 999"],
            ),
            first_sentences_matching(
                row.get("content", ""),
                ["see a gp", "call 999", "urgent", "immediate action", "do not smoke", "avoid"],
            ),
        ]
    )
    record["Doctor Response"] = join_unique(
        [
            first_paragraph(row.get("content", "")),
            select_sections(sections, ["treatment", "tests", "causes"]),
        ]
    )
    return record


def map_nhs_medicines(row: Dict[str, str]) -> Dict[str, str]:
    sections = parse_markdown_sections(row.get("content", ""))
    record = base_record()
    record["Drug Name"] = clean_name(row.get("name") or row.get("title"))
    record["Disease Name"] = extract_disease_targets_from_medicine(row.get("content", ""))
    record["Symptom Description"] = select_sections(sections, ["side effects"])
    record["Precautions"] = join_unique(
        [
            select_sections(
                sections,
                [
                    "who can and cannot take",
                    "warnings",
                    "pregnancy",
                    "breastfeeding",
                    "cautions",
                    "interactions",
                    "how and when to take",
                ],
            ),
            first_sentences_matching(row.get("content", ""), ["warning", "avoid", "not suitable", "do not"]),
        ]
    )
    record["Doctor Response"] = join_unique(
        [
            first_paragraph(row.get("content", "")),
            first_sentences_matching(row.get("content", ""), ["used to treat", "used for", "works by", "take"], limit=3),
            select_sections(sections, ["about", "how and when to take"]),
        ]
    )
    return record


def map_nhs_symptoms(row: Dict[str, str]) -> Dict[str, str]:
    sections = parse_markdown_sections(row.get("content", ""))
    record = base_record()
    record["Symptom Description"] = clean_name(row.get("title") or row.get("name"))
    record["Precautions"] = join_unique(
        [
            select_sections(sections, ["see a gp", "call 999", "urgent", "immediate action"]),
            first_sentences_matching(row.get("content", ""), ["see a gp", "call 999", "urgent", "emergency"]),
        ]
    )
    record["Doctor Response"] = join_unique(
        [
            first_paragraph(row.get("content", "")),
            first_sentences_matching(row.get("content", ""), ["caused by", "can be caused", "treatment"], limit=3),
            select_sections(sections, ["causes", "treatment", "what happens"]),
        ]
    )
    return record


def map_nhs_treatments(row: Dict[str, str]) -> Dict[str, str]:
    sections = parse_markdown_sections(row.get("content", ""))
    record = base_record()
    record["Disease Name"] = derive_condition_name_from_treatment(row.get("title") or row.get("name"))
    record["Precautions"] = join_unique(
        [
            select_sections(sections, ["risk", "before", "after", "recovery", "advice", "side effects"]),
            first_sentences_matching(row.get("content", ""), ["risk", "before", "after", "recovery", "urgent"]),
        ]
    )
    record["Doctor Response"] = join_unique(
        [
            first_paragraph(row.get("content", "")),
            first_sentences_matching(row.get("content", ""), ["what it is", "what happens", "used to", "done"], limit=3),
            select_sections(sections, ["treatment", "screening", "procedure", "test", "what happens"]),
        ]
    )
    return record


def map_who_topics(row: Dict[str, str]) -> Dict[str, str]:
    overview = row.get("overview", "")
    record = base_record()
    record["Disease Name"] = clean_name(row.get("topic"))
    record["Symptom Description"] = first_sentences_matching(
        overview, ["symptom", "sign", "condition", "disease", "illness"], limit=2
    )
    record["Precautions"] = first_sentences_matching(
        overview, ["prevent", "prevention", "risk", "avoid", "protect", "vaccin", "hygiene"], limit=3
    )
    record["Doctor Response"] = first_paragraph(overview)
    return record


def map_wikipedia(row: Dict[str, str]) -> Dict[str, str]:
    entry = clean_name(row.get("entry"))
    category = clean_name(row.get("category"))
    definition = normalize_text(row.get("definition"))
    record = base_record()
    if guess_wikipedia_type(entry, category, definition) == "drug":
        record["Drug Name"] = entry
    else:
        record["Disease Name"] = entry

    record["Symptom Description"] = first_sentences_matching(
        definition,
        ["symptom", "sign", "characterized by", "presents with", "manifest", "causes"],
        limit=2,
    )
    record["Precautions"] = first_sentences_matching(
        definition,
        ["prevent", "prevention", "risk", "avoid", "warning", "contraindicated", "seek medical attention"],
        limit=2,
    )
    record["Doctor Response"] = definition
    return record


MAPPERS: Dict[str, Callable[[Dict[str, str]], Dict[str, str]]] = {
    "map_nhs_conditions": map_nhs_conditions,
    "map_nhs_medicines": map_nhs_medicines,
    "map_nhs_symptoms": map_nhs_symptoms,
    "map_nhs_treatments": map_nhs_treatments,
    "map_who_topics": map_who_topics,
}


def build_search_text(record: Dict[str, str], title: str) -> str:
    parts = [f"Title: {title}"] if title else []
    for field in STANDARD_COLUMNS:
        value = normalize_text(record.get(field, ""))
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts).strip()


def build_document(
    source: str,
    source_file: str,
    source_type: str,
    title: str,
    url: str,
    raw_text: str,
    structured_record: Dict[str, str],
    row_index: int,
) -> Dict[str, object]:
    identifier_seed = f"{source_file}|{url}|{title}|{row_index}"
    doc_id = hashlib.md5(identifier_seed.encode("utf-8", errors="ignore")).hexdigest()
    return {
        "doc_id": doc_id,
        "source": source,
        "source_file": source_file,
        "source_type": source_type,
        "title": clean_name(title),
        "url": normalize_text(url),
        "row_index": row_index,
        "language": "en",
        "schema_version": "unstructured_v1",
        "structured_data": {field: normalize_text(structured_record.get(field, "")) for field in STANDARD_COLUMNS},
        "raw_text": normalize_text(raw_text),
        "search_text": build_search_text(structured_record, clean_name(title)),
    }


def iter_unstructured_documents(input_folder: Path) -> Iterator[Dict[str, object]]:
    for config in SOURCE_CONFIGS:
        input_path = input_folder / config["input_name"]
        mapper = MAPPERS[config["mapper"]]
        with input_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row_index, raw_row in enumerate(reader):
                row = clean_row_keys(raw_row)
                record = mapper(row)
                title = row.get("name") or row.get("title") or row.get("topic") or row.get("entry") or ""
                url = row.get("url", "")
                raw_text = row.get("content") or row.get("overview") or row.get("definition") or ""
                yield build_document(
                    source=config["source"],
                    source_file=config["input_name"],
                    source_type=config["source_type"],
                    title=title,
                    url=url,
                    raw_text=raw_text,
                    structured_record=record,
                    row_index=row_index,
                )

    wikipedia_path = input_folder / "wikipedia_medical.csv"
    with wikipedia_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row_index, row in enumerate(reader):
            raw_row = {
                "entry": row[0] if len(row) > 0 else "",
                "url": row[1] if len(row) > 1 else "",
                "category": row[2] if len(row) > 2 else "",
                "definition": row[3] if len(row) > 3 else "",
            }
            record = map_wikipedia(raw_row)
            yield build_document(
                source="Wikipedia",
                source_file="wikipedia_medical.csv",
                source_type=guess_wikipedia_type(raw_row["entry"], raw_row["category"], raw_row["definition"]),
                title=raw_row["entry"],
                url=raw_row["url"],
                raw_text=raw_row["definition"],
                structured_record=record,
                row_index=row_index,
            )


def write_jsonl(path: Path, rows: Iterator[Dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
