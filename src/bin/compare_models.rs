use anyhow::Result;
use hay_museum::bert_classifier::{BertClassifier, ModelType};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<()> {
    println!("Loading test records from test.csv...");

    // Read borderline records (0.44-0.50) from test.csv
    let file = File::open("test.csv")?;
    let reader = BufReader::new(file);

    let mut test_records: Vec<(String, String, f32)> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(record) = serde_json::from_str::<serde_json::Value>(&line) {
            let score = record["armenian_score"].as_f64().unwrap_or(0.0) as f32;

            // Only test borderline records (0.44-0.55)
            if score >= 0.44 && score < 0.55 {
                let name = record["name"].as_str().unwrap_or("").to_string();
                let desc = record["description"].as_str().unwrap_or("").to_string();
                let text = format!("{} {}", name, desc);

                test_records.push((name, text, score));

                if test_records.len() >= 50 {
                    break;
                }
            }
        }
    }

    println!("Testing {} borderline records", test_records.len());
    println!();

    // Load LaBSE model
    println!("{}", "=".repeat(60));
    println!("Loading LaBSE model...");
    println!("{}", "=".repeat(60));
    let mut labse = BertClassifier::load_with_model(ModelType::LaBSE)?;
    labse.create_armenian_prototypes()?;

    // Load Russian BERT model
    println!();
    println!("{}", "=".repeat(60));
    println!("Loading Russian BERT model...");
    println!("{}", "=".repeat(60));
    let mut rubert = BertClassifier::load_with_model(ModelType::RuBERT)?;
    rubert.create_armenian_prototypes()?;

    // Compare scores
    println!();
    println!("{}", "=".repeat(60));
    println!("Comparing model scores on {} records", test_records.len());
    println!("{}", "=".repeat(60));
    println!();

    let mut labse_better = 0;
    let mut rubert_better = 0;
    let mut similar = 0;

    for (i, (name, text, original_score)) in test_records.iter().enumerate() {
        let labse_score = labse.score_armenian_relevance(text)?;
        let rubert_score = rubert.score_armenian_relevance(text)?;

        let diff = (rubert_score - labse_score).abs();

        if diff < 0.05 {
            similar += 1;
        } else if rubert_score > labse_score {
            rubert_better += 1;
        } else {
            labse_better += 1;
        }

        if i < 10 || diff > 0.10 {
            // UTF-8 safe truncation
            let display_name = if name.chars().count() > 40 {
                name.chars().take(40).collect::<String>() + "..."
            } else {
                name.clone()
            };
            println!("Record {}: {}", i + 1, display_name);
            println!("  Original: {:.4}", original_score);
            println!("  LaBSE:    {:.4}", labse_score);
            println!("  RuBERT:   {:.4} {}", rubert_score, if diff > 0.10 { "⚠️  BIG DIFF" } else { "" });
            println!("  Diff:     {:.4}", rubert_score - labse_score);
            println!();
        }
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("Summary:");
    println!("{}", "=".repeat(60));
    println!("Similar scores (diff < 0.05): {}", similar);
    println!("RuBERT scored higher:         {}", rubert_better);
    println!("LaBSE scored higher:          {}", labse_better);
    println!();

    Ok(())
}
