use anyhow::Result;
use hay_museum::bert_classifier::BertClassifier;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<()> {
    println!("Testing improved prototypes on borderline records\n");

    // Read borderline records from test.csv
    let file = File::open("test.csv")?;
    let reader = BufReader::new(file);

    let mut test_records: Vec<(String, String, f32)> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(record) = serde_json::from_str::<serde_json::Value>(&line) {
            let score = record["armenian_score"].as_f64().unwrap_or(0.0) as f32;

            // Test borderline records (0.44-0.55)
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

    println!("Testing {} borderline records (0.44-0.55 range)\n", test_records.len());

    // Load model with new prototypes
    println!("Loading BERT model with improved prototypes...");
    let mut classifier = BertClassifier::load()?;
    classifier.create_armenian_prototypes()?;

    // Test on records
    let mut improved = 0;
    let mut similar = 0;
    let mut worse = 0;
    let mut big_improvements = Vec::new();

    println!("\n{}", "=".repeat(70));
    println!("Score Comparison (Old vs New with Improved Prototypes)");
    println!("{}\n", "=".repeat(70));

    for (i, (name, text, old_score)) in test_records.iter().enumerate() {
        let new_score = classifier.score_armenian_relevance(text)?;
        let diff = new_score - old_score;

        if diff > 0.05 {
            improved += 1;
            if diff > 0.15 {
                let display_name = if name.chars().count() > 45 {
                    name.chars().take(45).collect::<String>() + "..."
                } else {
                    name.clone()
                };
                big_improvements.push((i + 1, display_name, *old_score, new_score, diff));
            }
        } else if diff < -0.05 {
            worse += 1;
        } else {
            similar += 1;
        }
    }

    // Show big improvements
    println!("Top improvements (Δ > 0.15):\n");
    for (idx, name, old, new, diff) in big_improvements.iter().take(15) {
        println!("{}. {}", idx, name);
        println!("   Old: {:.4} → New: {:.4} (Δ {:.4})", old, new, diff);
        println!();
    }

    println!("{}", "=".repeat(70));
    println!("Summary:");
    println!("{}", "=".repeat(70));
    println!("Improved (Δ > +0.05):     {} ({:.1}%)", improved, 100.0 * improved as f32 / test_records.len() as f32);
    println!("Similar (|Δ| ≤ 0.05):     {} ({:.1}%)", similar, 100.0 * similar as f32 / test_records.len() as f32);
    println!("Worse (Δ < -0.05):        {} ({:.1}%)", worse, 100.0 * worse as f32 / test_records.len() as f32);
    println!("Big improvements (>0.15): {}", big_improvements.len());
    println!();

    if improved > worse {
        println!("✅ Net improvement: {} more records scored higher!", improved - worse);
    }

    Ok(())
}
