use polars::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use anyhow::Error;

fn main() -> Result<(), Error> {
    let stemmer = Stemmer::create(Algorithm::Russian);
    let base_path = PathBuf::from("data/armenian-keywords/data/ru/");

    // Load filters
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
        let russian_stopwords = vec!["потеря", "крестья", "емельян", "емелья"];
        let problematic_stems = vec!["василь", "грабар", "андрия", "демья", "татья", "оловя", "сафья"];

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

    // Load filter lists
    let names_orig = read(&base_path, "names.csv")?;
    let midnames_orig = read(&base_path, "midnames.csv")?;
    let surnames_orig = read(&base_path, "surnames.csv")?;
    let keywords_orig = read(&base_path, "keywords.csv")?;
    let geonames = read(&base_path, "geonames.csv")?;

    let names = stem_series(&names_orig, &stemmer)?;
    let midnames = stem_series(&midnames_orig, &stemmer)?;
    let surnames = stem_series(&surnames_orig, &stemmer)?;
    let keywords = stem_series(&keywords_orig, &stemmer)?;

    // Create reverse mapping: stem -> original words
    fn create_reverse_map(orig: &Series, stemmed: &Series, stemmer: &Stemmer) -> HashMap<String, Vec<String>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        if let (Ok(orig_str), Ok(_stem_str)) = (orig.str(), stemmed.str()) {
            for val in orig_str.into_iter().flatten() {
                let stem = stemmer.stem(val).into_owned();
                if stem.chars().count() >= 5 {
                    map.entry(stem).or_insert_with(Vec::new).push(val.to_string());
                }
            }
        }
        map
    }

    let name_map = create_reverse_map(&names_orig, &names, &stemmer);
    let midname_map = create_reverse_map(&midnames_orig, &midnames, &stemmer);
    let surname_map = create_reverse_map(&surnames_orig, &surnames, &stemmer);
    let keyword_map = create_reverse_map(&keywords_orig, &keywords, &stemmer);

    // Combine all stems into one set for fast lookup
    let mut all_stems: HashSet<String> = HashSet::new();
    for stem in names.str()?.into_iter().flatten() {
        all_stems.insert(stem.to_string());
    }
    for stem in midnames.str()?.into_iter().flatten() {
        all_stems.insert(stem.to_string());
    }
    for stem in surnames.str()?.into_iter().flatten() {
        all_stems.insert(stem.to_string());
    }
    for stem in keywords.str()?.into_iter().flatten() {
        all_stems.insert(stem.to_string());
    }

    // Geonames (no stemming)
    let mut geoname_set: HashSet<String> = HashSet::new();
    for geo in geonames.str()?.into_iter().flatten() {
        geoname_set.insert(geo.to_string());
    }

    // Sample and analyze records
    println!("Loading test.csv...\n");
    let df = LazyFrame::scan_ndjson("test.csv", Default::default())?
        .select([
            col("id"),
            col("name"),
            col("description"),
            col("productionPlace"),
            col("findPlace"),
        ])
        .collect()?;

    // Sample 30 random records
    let sampled = df.sample_n(30, false, true, Some(rand::random()))?;

    for i in 0..sampled.height() {
        println!("{}", "=".repeat(80));
        let id = sampled.column("id")?.i64()?.get(i).unwrap_or(0);
        println!("RECORD ID: {}", id);
        println!("{}", "=".repeat(80));

        let mut matched_geonames = Vec::new();
        let mut matched_stems = Vec::new();

        // Check geoname matches
        let check_geo = |text: Option<&str>| {
            if let Some(t) = text {
                let lower = t.to_lowercase();
                for geo in &geoname_set {
                    if lower.contains(geo) {
                        return Some(geo.clone());
                    }
                }
            }
            None
        };

        if let Ok(col) = sampled.column("productionPlace") {
            if let Ok(str_col) = col.str() {
                if let Some(place) = str_col.get(i) {
                    if let Some(geo) = check_geo(Some(place)) {
                        matched_geonames.push(format!("productionPlace: {}", geo));
                    }
                }
            }
        }

        if let Ok(col) = sampled.column("findPlace") {
            if let Ok(str_col) = col.str() {
                if let Some(place) = str_col.get(i) {
                    if let Some(geo) = check_geo(Some(place)) {
                        matched_geonames.push(format!("findPlace: {}", geo));
                    }
                }
            }
        }

        // Check stem matches in name and description
        let check_stems = |text: Option<&str>, field: &str| -> Vec<String> {
            let mut matches = Vec::new();
            if let Some(t) = text {
                let words: Vec<&str> = t.split_whitespace().collect();
                for word in words {
                    let cleaned = word.to_lowercase()
                        .trim_matches(|c: char| !c.is_alphabetic())
                        .to_string();

                    if cleaned.len() < 3 { continue; }

                    let stem = stemmer.stem(&cleaned).into_owned();
                    if stem.chars().count() >= 5 && all_stems.contains(&stem) {
                        let mut sources = Vec::new();
                        if let Some(orig) = name_map.get(&stem) {
                            sources.push(format!("name:{}", orig.join(",")));
                        }
                        if let Some(orig) = midname_map.get(&stem) {
                            sources.push(format!("midname:{}", orig.join(",")));
                        }
                        if let Some(orig) = surname_map.get(&stem) {
                            sources.push(format!("surname:{}", orig.join(",")));
                        }
                        if let Some(orig) = keyword_map.get(&stem) {
                            sources.push(format!("keyword:{}", orig.join(",")));
                        }
                        matches.push(format!("{}: '{}' -> stem:'{}' <- {}",
                            field, cleaned, stem, sources.join(" | ")));
                    }
                }
            }
            matches
        };

        if let Ok(col) = sampled.column("name") {
            if let Ok(str_col) = col.str() {
                if let Some(name) = str_col.get(i) {
                    println!("NAME: {}", name);
                    matched_stems.extend(check_stems(Some(name), "name"));
                }
            }
        }

        if let Ok(col) = sampled.column("description") {
            if let Ok(str_col) = col.str() {
                if let Some(desc) = str_col.get(i) {
                    let truncated = if desc.len() > 300 {
                        format!("{}...", &desc[..300])
                    } else {
                        desc.to_string()
                    };
                    println!("DESCRIPTION: {}", truncated);
                    matched_stems.extend(check_stems(Some(desc), "description"));
                }
            }
        }

        println!("\n>>> MATCH ANALYSIS:");
        if !matched_geonames.is_empty() {
            println!("  GEONAME MATCHES:");
            for m in matched_geonames {
                println!("    ✓ {}", m);
            }
        }
        if !matched_stems.is_empty() {
            println!("  STEM MATCHES:");
            for m in matched_stems {
                println!("    ✓ {}", m);
            }
        }
        if matched_geonames.is_empty() && matched_stems.is_empty() {
            println!("  ⚠ NO MATCHES FOUND (unexpected!)");
        }

        println!();
    }

    Ok(())
}
