# hay-museum

A Rust-based tool for identifying Armenian cultural artifacts in large museum collections. Uses a two-stage filtering pipeline combining fast keyword matching with BERT-based semantic classification.

## Overview

This project processes museum datasets (e.g., Russian Goskatalog) to identify items related to Armenian culture and history. The pipeline consists of:

1. **Keyword Filtering** - Fast regex-based filtering using Russian stemming and curated keyword lists
2. **BERT Scoring** - Semantic classification using a fine-tuned rubert-mini-armenian model

## Prerequisites

- Rust toolchain (rustc 1.70+)
- 4GB+ RAM for model loading
- ~4GB disk space for data and models

### Data Setup

Initialize the keywords submodule:

```bash
git submodule update --init --recursive
```

Required data files:
- `data/data.parquet` - Input museum records
- `py/rubert-mini-armenian/final/` - Fine-tuned BERT model

## Building

```bash
# Standard build (CPU only)
cargo build --release

# With NVIDIA GPU support
cargo build --release --features cuda

# With Apple Metal GPU support (macOS)
cargo build --release --features metal
```

## Running

### Main Pipeline

Run the full preprocessing and scoring pipeline:

```bash
cargo run --release
```

This will:
1. Load `data/data.parquet`
2. Filter records using keywords, geonames, and names
3. Output `data/data-preprocessed.parquet`
4. Score filtered records with BERT
5. Write results to `result.json`

With GPU acceleration:

```bash
cargo run --release --features cuda   # NVIDIA
cargo run --release --features metal  # Apple Silicon
```

### Utility Binaries

| Binary | Description |
|--------|-------------|
| `test_bert_classifier` | Test BERT classifier on sample data |
| `compare_models` | Compare different BERT model outputs |
| `check_preprocessed` | Validate preprocessed parquet file |
| `analyze_matches` | Analyze keyword matching patterns |
| `find_collisions` | Find name/keyword collisions |
| `test_penalty` | Test penalty-based filtering |
| `count_records` | Count records in datasets |
| `stem_words` | Test stemming on specific words |
| `trace_match` | Trace why a specific record matched |

Run a specific binary:

```bash
cargo run --release --bin test_bert_classifier
cargo run --release --bin compare_models
```

## Project Structure

```
hay-museum/
├── src/
│   ├── main.rs              # Main pipeline
│   ├── lib.rs               # Library exports
│   ├── bert_classifier.rs   # BERT inference
│   └── bin/                 # Utility binaries
├── data/
│   ├── data.parquet         # Input data
│   ├── data-preprocessed.parquet  # Filtered output
│   └── armenian-keywords/   # Keyword lists (submodule)
│       └── data/ru/
│           ├── keywords.csv
│           ├── geonames.csv
│           ├── names.csv
│           ├── surnames.csv
│           └── midnames.csv
├── py/
│   └── rubert-mini-armenian/  # Fine-tuned BERT model
└── Cargo.toml
```

## Filter Categories

The keyword filtering stage uses several categories from `data/armenian-keywords/data/ru/`:

- **keywords.csv** - Armenian-related terms (армения, урарту, вардапет, etc.)
- **geonames.csv** - Cities, regions, geographic features (Ереван, Карабах, Арарат)
- **names.csv** - Armenian given names (Тигран, Арташес, Месроп)
- **surnames.csv** - Armenian family names (~92K entries)
- **midnames.csv** - Armenian patronymics

## Configuration

Key constants in the code:

| Constant | Value | Description |
|----------|-------|-------------|
| `BERT_THRESHOLD` | 0.25 | Minimum score for acceptance |
| `MIN_STEM_LENGTH` | 4 | Minimum characters for stemmed words |
| `TRUNCATION_LENGTH` | 1536 | Maximum BERT input length |
| `BATCH_SIZE` | 16 | Batch size for BERT inference |

## Output Format

The pipeline outputs `result.json` containing records that pass both filtering stages:

```json
{
  "id": "12345",
  "name": "...",
  "description": "...",
  "armenian_score": 0.87,
  ...
}
```

## Testing

```bash
cargo test
```

## License

See LICENSE file for details.
