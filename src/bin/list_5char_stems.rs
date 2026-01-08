use polars::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashMap;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stemmer = Stemmer::create(Algorithm::Russian);
    let base_path = PathBuf::from("data/armenian-keywords/data/ru/");

    let mut all_5char_stems: HashMap<String, Vec<String>> = HashMap::new();

    for (file_type, filename) in [
        ("Names", "names.csv"),
        ("Surnames", "surnames.csv"),
        ("Keywords", "keywords.csv"),
    ] {
        let path = base_path.join(filename);
        let df = LazyCsvReader::new(PlPathRef::Local(&path).into_owned())
            .with_has_header(true)
            .finish()?
            .collect()?;

        let series = df.column("value")?.str()?;

        for val in series.into_iter().flatten() {
            let lower = val.to_lowercase();
            let stem = stemmer.stem(&lower).into_owned();
            if stem.chars().count() == 5 {
                all_5char_stems
                    .entry(stem.clone())
                    .or_default()
                    .push(format!("{} (from {})", val, file_type));
            }
        }
    }

    println!("=== All Armenian 5-char stems ({} unique) ===\n", all_5char_stems.len());

    // Sort by number of original words that stem to this
    let mut stems_vec: Vec<_> = all_5char_stems.iter().collect();
    stems_vec.sort_by_key(|(_, originals)| std::cmp::Reverse(originals.len()));

    // Show top 50 most common stems
    for (stem, originals) in stems_vec.iter().take(50) {
        println!("'{}' ← {} word(s):", stem, originals.len());
        for orig in originals.iter().take(3) {
            println!("    {}", orig);
        }
        if originals.len() > 3 {
            println!("    ... and {} more", originals.len() - 3);
        }
        println!();
    }

    Ok(())
}
