import csv
import re
from pathlib import Path


INPUT_FOLDER = Path(__file__).resolve().parents[2] / "Dataset" / "02_Unstructured"
csv.field_size_limit(10_000_000)
STANDARD_COLUMNS = [
    "Disease Name",
    "Symptom Description",
    "Drug Name",
    "Precautions",
    "Patient Question",
    "Doctor Response",
]


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_row_keys(row: dict[str, str]) -> dict[str, str]:
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
    text = re.sub(r"^##+\s*[^\n]+", "", text).strip()
    return text


def parse_markdown_sections(content: str) -> dict[str, str]:
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


def select_sections(sections: dict[str, str], keywords: list[str]) -> str:
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
    sentences = split_sentences(text)
    return " ".join(sentences[:3])


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
            target = re.sub(r"\s+", " ", target).strip(" ,;:")
            return target
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
    if match:
        return match.group(1).strip()

    return title


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
    if any(keyword in text for keyword in drug_keywords):
        return "drug"
    return "disease"


def base_record() -> dict[str, str]:
    return {column: "" for column in STANDARD_COLUMNS}


def map_nhs_conditions(row: dict[str, str]) -> dict[str, str]:
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


def map_nhs_medicines(row: dict[str, str]) -> dict[str, str]:
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


def map_nhs_symptoms(row: dict[str, str]) -> dict[str, str]:
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


def map_nhs_treatments(row: dict[str, str]) -> dict[str, str]:
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


def map_who_topics(row: dict[str, str]) -> dict[str, str]:
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


def map_wikipedia(row: dict[str, str]) -> dict[str, str]:
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


def write_records(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def clean_nhs_like_file(input_name: str, mapper) -> tuple[str, int]:
    input_path = INPUT_FOLDER / input_name
    output_path = INPUT_FOLDER / input_name.replace(".csv", "_cleaned.csv")
    row_count = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as f, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out_f, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        for row in reader:
            writer.writerow(mapper(clean_row_keys(row)))
            row_count += 1

    return output_path.name, row_count


def clean_wikipedia_file() -> tuple[str, int]:
    input_path = INPUT_FOLDER / "wikipedia_medical.csv"
    output_path = INPUT_FOLDER / "wikipedia_medical_cleaned.csv"
    row_count = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as f, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.reader(f)
        writer = csv.DictWriter(out_f, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        next(reader)
        for row in reader:
            writer.writerow(
                map_wikipedia(
                    {
                        "entry": row[0] if len(row) > 0 else "",
                        "url": row[1] if len(row) > 1 else "",
                        "category": row[2] if len(row) > 2 else "",
                        "definition": row[3] if len(row) > 3 else "",
                    }
                )
            )
            row_count += 1

    return output_path.name, row_count


def main() -> None:
    summary = []

    file_mappings = [
        ("nhs_conditions.csv", map_nhs_conditions),
        ("nhs_medicines.csv", map_nhs_medicines),
        ("nhs_symptoms.csv", map_nhs_symptoms),
        ("nhs_treatments.csv", map_nhs_treatments),
        ("who_health_topics.csv", map_who_topics),
    ]

    for input_name, mapper in file_mappings:
        output_name, row_count = clean_nhs_like_file(input_name, mapper)
        summary.append((input_name, output_name, row_count))
        print(f"Cleaned {input_name} -> {output_name} ({row_count} rows)")

    output_name, row_count = clean_wikipedia_file()
    summary.append(("wikipedia_medical.csv", output_name, row_count))
    print(f"Cleaned wikipedia_medical.csv -> {output_name} ({row_count} rows)")

    merged_output = INPUT_FOLDER / "all_unstructured_cleaned.csv"
    merged_count = 0
    with merged_output.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        for _, output_name, _ in summary:
            with (INPUT_FOLDER / output_name).open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    writer.writerow(row)
                    merged_count += 1
    print(f"Merged all cleaned files -> {merged_output.name} ({merged_count} rows)")

    print("\nSummary:")
    for input_name, output_name, row_count in summary:
        print(f"- {input_name}: {output_name}, rows={row_count}")


if __name__ == "__main__":
    main()
