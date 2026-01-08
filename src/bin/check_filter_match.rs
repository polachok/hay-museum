use polars::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};
use std::path::PathBuf;
use anyhow::Error;

fn main() -> Result<(), Error> {
    let stemmer = Stemmer::create(Algorithm::Russian);
    let base_path = PathBuf::from("data/armenian-keywords/data/ru/");

    // Load all Armenian filter stems (same logic as main.rs)
    fn read(base: &std::path::Path, filename: &str) -> Result<Series, Error> {
        let path = base.join(filename);
        Ok(LazyCsvReader::new(PlPathRef::Local(&path).into_owned())
            .with_has_header(true)
            .finish()?
            .collect()?
            .column("value")?
            .as_series()
            .ok_or_else(|| anyhow::anyhow!("missing column value"))?
            .str()?
            .to_lowercase()
            .into_series())
    }

    fn stem_series(series: &Series, stemmer: &Stemmer) -> Result<Series, Error> {
        let russian_stopwords = ["потеря", "крестья", "емельян", "емелья"];
        let problematic_stems = ["василь", "грабар", "андрия", "демья", "татья"];

        let rechunked = series.rechunk();
        let stemmed: Vec<String> = rechunked
            .str()?
            .into_iter()
            .filter_map(|opt_val| {
                opt_val.and_then(|val| {
                    let stem = stemmer.stem(val).into_owned();
                    if stem.chars().count() >= 5
                        && !russian_stopwords.contains(&stem.as_str())
                        && !problematic_stems.contains(&stem.as_str()) {
                        Some(stem)
                    } else {
                        None
                    }
                })
            })
            .collect();
        Ok(Series::new(series.name().clone(), stemmed))
    }

    let names = stem_series(&read(&base_path, "names.csv")?, &stemmer)?;
    let midnames = stem_series(&read(&base_path, "midnames.csv")?, &stemmer)?;
    let surnames = stem_series(&read(&base_path, "surnames.csv")?, &stemmer)?;
    let keywords = stem_series(&read(&base_path, "keywords.csv")?, &stemmer)?;

    let mut description_filter = Series::new_empty("desc".into(), &DataType::String);
    description_filter.append(&names)?;
    description_filter.append(&midnames)?;
    description_filter.append(&surnames)?;
    description_filter.append(&keywords)?;
    let description_filter = description_filter.unique()?;

    println!("Total Armenian filter stems: {}\n", description_filter.len());

    // German soldier stems
    let german_stems = vec![
        "антикварн", "барабан", "барабанщик", "герман", "знаменосец",
        "коллекц", "музыкант", "нюринберг", "окраш", "оловя",
        "офицер", "пехот", "постамент", "прусск", "сдела",
        "солдат", "солдатик", "тольк", "фигурк", "хайнрихсс",
        "цветн", "эрнст"
    ];

    println!("Checking German soldier stems against Armenian filter:\n");

    let filter_vec: Vec<_> = description_filter.str()?.into_iter().flatten().collect();

    for stem in &german_stems {
        if filter_vec.contains(stem) {
            println!("✓ MATCH FOUND: {}", stem);
        }
    }

    println!("\nNo matches printed above means ZERO collisions!");

    Ok(())
}
