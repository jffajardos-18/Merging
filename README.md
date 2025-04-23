#  Company Records Deduplication and Merging Tool

This Python script automates the process of comparing, deduplicating, and merging two Excel files containing company data. It uses a mix of normalization, fuzzy string matching, phonetic encoding (Soundex), and conflict resolution to ensure a high-quality merged dataset with traceability of duplicates.

---

##  Overview

### Inputs:
- Two Excel files with company data (`file1_path` and `file2_path`).
- The script assumes each file contains columns like **Company name** and **Company website**.

### Outputs:
- A single Excel workbook containing:
  - Merged, deduplicated data
  - Logged duplicates with match reasons
  - Unique entries from each original file
  - Flagged low-confidence clusters for review

---

## ⚙ Features

- **Normalization** of names and websites (removes suffixes, punctuation, etc.)
- **Phonetic encoding** (via Soundex) to group similar-sounding entries
- **Fuzzy matching** using `rapidfuzz` to detect near duplicates
- **Conflict resolution** favoring one file (via `PRIORITY`)
- **Union-Find clustering** to group duplicates
- **Quality Assurance** to flag ambiguous clusters
- **Excel export** using `openpyxl` for easy consumption

---

##  How It Works

1. **Read & preprocess** both files
2. **Normalize** company names and websites
3. **Generate phonetic keys** using `jellyfish.soundex`
4. **Identify duplicates** through exact and fuzzy matching
5. **Log duplicate links** with matching scores and rationale
6. **Cluster entries** using a union-find structure
7. **Flag clusters** where similarity scores drop below a threshold
8. **Merge entries** by cluster, preserving source data and conflicts
9. **Export results** to an Excel file with multiple sheets

---

##  Output Sheets

| Sheet Name            | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| `Clean Data`          | Final merged dataset                                         |
| `Removed Duplicates`  | Pairs identified as duplicates with matching info            |
| `Unique File1`        | Entries unique to the first file                             |
| `Unique File2`        | Entries unique to the second file                            |
| `Flagged Clusters`    | Clusters flagged for low-confidence matches (optional)       |

---

##  Configuration

You can adjust:
- `file1_path`, `file2_path`, `output_path` — paths to input/output files
- `PRIORITY` — which file takes precedence when merging
- `MIN_CLUSTER_SCORE` — score threshold for QA flagging

---

##  Requirements

Make sure the following Python packages are installed:

```bash
pip install pandas numpy rapidfuzz jellyfish openpyxl
```

Optional:
```bash
pip install tldextract
```

---

##  Notes

- The script is designed to **never drop data**—all conflicts are logged.
- If `tldextract` is unavailable, it falls back to `urllib` for domain parsing.
- Designed for **clean, auditable merging** in collaborative data projects.

---
