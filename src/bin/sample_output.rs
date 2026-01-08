use polars::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let sample_size = args.get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);

    println!("Sampling {} random records from test.csv...\n", sample_size);

    let df = LazyFrame::scan_ndjson("test.csv", Default::default())?
        .select([
            col("id"),
            col("name"),
            col("description"),
            col("productionPlace"),
            col("findPlace"),
        ])
        .collect()?;

    // Get random sample
    let sampled = df.sample_n(sample_size, false, true, Some(rand::random()))?;

    for i in 0..sampled.height() {
        println!("{}", "=".repeat(80));
        println!("RECORD #{} (ID: {})", i + 1,
            sampled.column("id")?.i64()?.get(i).unwrap_or(0));
        println!("{}", "=".repeat(80));

        if let Ok(col) = sampled.column("name") {
            if let Ok(str_col) = col.str() {
                if let Some(name) = str_col.get(i) {
                    println!("NAME: {}", name);
                }
            }
        }

        if let Ok(col) = sampled.column("productionPlace") {
            if let Ok(str_col) = col.str() {
                if let Some(place) = str_col.get(i) {
                    if !place.is_empty() {
                        println!("PRODUCTION PLACE: {}", place);
                    }
                }
            }
        }

        if let Ok(col) = sampled.column("findPlace") {
            if let Ok(str_col) = col.str() {
                if let Some(place) = str_col.get(i) {
                    if !place.is_empty() {
                        println!("FIND PLACE: {}", place);
                    }
                }
            }
        }

        if let Ok(col) = sampled.column("description") {
            if let Ok(str_col) = col.str() {
                if let Some(desc) = str_col.get(i) {
                    if !desc.is_empty() {
                        let truncated = if desc.len() > 400 {
                            format!("{}...", &desc[..400])
                        } else {
                            desc.to_string()
                        };
                        println!("DESCRIPTION: {}", truncated);
                    }
                }
            }
        }

        println!();
    }

    Ok(())
}
