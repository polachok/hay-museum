use anyhow::Result;
use polars::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

// Import from parent crate
use hay_museum::bert_classifier::BertClassifier;

fn main() -> Result<()> {
    eprintln!("BERT Classifier Test Tool");
    eprintln!("=========================\n");

    // Load test data (filtered records from keyword filtering)
    let data_path = "test.csv";
    eprintln!("Loading test data from {}...", data_path);

    let file = File::open(data_path)?;
    let reader = BufReader::new(file);

    // Define schema for the columns we need
    let schema = Schema::from_iter(vec![
        Field::new("id".into(), DataType::Int64),
        Field::new("name".into(), DataType::String),
        Field::new("description".into(), DataType::String),
    ]);

    // Command line argument for sample range
    let args: Vec<String> = std::env::args().collect();
    let start_offset = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0)
    } else {
        0
    };

    let df = JsonReader::new(reader)
        .with_json_format(JsonFormat::JsonLines)
        .with_schema(Arc::new(schema))
        .finish()?
        .slice(start_offset as i64, 100);  // Test 100 records starting from offset

    eprintln!("Loaded {} records (sample {}-{})\n", df.height(), start_offset + 1, start_offset + df.height());

    // Initialize BERT classifier
    eprintln!("Initializing BERT classifier...");
    let mut classifier = BertClassifier::load()?;

    eprintln!("\nCreating Armenian prototypes...");
    classifier.create_armenian_prototypes()?;

    // Extract text fields
    eprintln!("\nScoring records...\n");
    eprintln!("{:<6} {:<100} {:<8} {:<10}", "ID", "Text Snippet", "Score", "Decision");
    eprintln!("{}", "-".repeat(130));

    let confidence_threshold = 0.44; // Lowered to capture Armenian cultural figures like Saryan

    let name_col = df.column("name")?.str()?;
    let desc_col = df.column("description")?.str()?;

    let mut kept = 0;
    let mut rejected = 0;
    let mut rejected_records = Vec::new();

    for i in 0..df.height() {
        let name = name_col.get(i).unwrap_or("");
        let desc = desc_col.get(i).unwrap_or("");

        // Combine name and description
        let text = format!("{} {}", name, desc);

        // Score the record
        let score = match classifier.score_armenian_relevance(&text) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error scoring record {}: {}", i, e);
                continue;
            }
        };

        // Make decision
        let decision = if score >= confidence_threshold {
            kept += 1;
            "KEEP"
        } else {
            rejected += 1;
            rejected_records.push((i + 1, text.clone(), score));
            "REJECT"
        };

        // Create text snippet (first 100 chars, char-boundary safe)
        let snippet = if text.chars().count() > 100 {
            let truncated: String = text.chars().take(97).collect();
            format!("{}...", truncated)
        } else {
            text.clone()
        };

        println!("{:<6} {:<100} {:<8.3} {:<10}", i + 1, snippet, score, decision);
    }

    eprintln!("\n{}", "=".repeat(130));
    eprintln!("Summary:");
    eprintln!("  Total records: {}", df.height());
    eprintln!("  Kept: {} ({:.1}%)", kept, (kept as f32 / df.height() as f32) * 100.0);
    eprintln!("  Rejected: {} ({:.1}%)", rejected, (rejected as f32 / df.height() as f32) * 100.0);
    eprintln!("  Threshold: {}", confidence_threshold);

    // Show rejected records for inspection
    eprintln!("\n{}", "=".repeat(130));
    eprintln!("REJECTED RECORDS (for inspection):");
    eprintln!("{}", "=".repeat(130));
    eprintln!("{:<6} {:<100} {:<8}", "ID", "Text Snippet", "Score");
    eprintln!("{}", "-".repeat(130));

    for (id, text, score) in rejected_records.iter().take(20) {
        let snippet = if text.chars().count() > 100 {
            let truncated: String = text.chars().take(97).collect();
            format!("{}...", truncated)
        } else {
            text.clone()
        };
        println!("{:<6} {:<100} {:<8.3}", id, snippet, score);
    }

    if rejected_records.len() > 20 {
        eprintln!("\n... and {} more rejected records", rejected_records.len() - 20);
    }

    Ok(())
}
