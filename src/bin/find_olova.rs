use polars::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stemmer = Stemmer::create(Algorithm::Russian);
    let base_path = PathBuf::from("data/armenian-keywords/data/ru/");

    for (file_type, filename) in [
        ("Names", "names.csv"),
        ("Midnames", "midnames.csv"),
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
            if stem == "оловя" {
                println!("FOUND: '{}' → '{}' (from {})", val, stem, file_type);
            }
        }
    }

    Ok(())
}
