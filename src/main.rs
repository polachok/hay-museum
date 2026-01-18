use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Error, anyhow};
use phf::phf_set;
use polars::lazy::dsl::by_name;
use polars::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};

use mimalloc::MiMalloc;

mod bert_classifier;
use bert_classifier::BertClassifier;

use indicatif::ProgressBar;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Exclude stems that match Russian words, names, and patronymics
// Stems to filter out to prevent false positives with Armenian surnames
// These are checked AFTER stemming, so values here should be stems
static RUSSIAN_STEMS: phf::Set<&'static str> = phf_set! {
    "василь", // from Васильян, matches Васильевич
    "грабар", // classical Armenian word, but matches Russian artist И.Э.Грабарь (Igor Grabar)
    "андрия", // Armenian surname Андриян, but matches Russian first name Андриян (e.g. cosmonaut)
    "демья",  // from Демьян (Armenian surname), matches Демьян (Russian first name)
    "татья",  // from Татьян (Armenian surname), matches Татьяна (Russian first name)
    "оловя", // from Оловян (Armenian surname), matches оловянный (Russian: tin/pewter - very common in museum items)
    "сафья", // from Сафьян (Armenian surname), matches сафьян (Russian: morocco leather - common in book bindings)
    "арсень", // from Арсеньев (Russian surname), matches Armenian name Арсен
};

struct Filters {
    names: Series,
    midnames: Series,
    surnames: Series,
    geonames: Series,
    keywords: Series,
}

fn load_filters(path: &str) -> Result<Filters, Error> {
    let base_path = PathBuf::from(path);
    let stemmer = Stemmer::create(Algorithm::Russian);

    fn read(base: &std::path::Path, filename: &str) -> Result<Series, Error> {
        let path = base.join(filename);
        Ok(LazyCsvReader::new(PlPathRef::Local(&path).into_owned())
            .with_has_header(true)
            .finish()?
            .collect()?
            .column("value")?
            .as_series()
            .ok_or_else(|| anyhow!("missing column value"))?
            .str()?
            .to_lowercase()
            .into_series())
    }

    let names = stem_series(&read(&base_path, "names.csv")?, &stemmer)?;
    let midnames = stem_series(&read(&base_path, "midnames.csv")?, &stemmer)?;
    let surnames = stem_series(&read(&base_path, "surnames.csv")?, &stemmer)?;

    let geonames_path = base_path.join("geonames.csv");
    let geonames = LazyCsvReader::new(PlPathRef::Local(&geonames_path).into_owned())
        .with_has_header(true)
        .with_rechunk(true)
        .finish()?
        .drop_nulls(None)
        .filter(
            col("value")
                .is_not_null()
                .and(
                    col("value")
                        .str()
                        .strip_chars(Default::default())
                        .str()
                        .len_bytes()
                        .gt(2),
                )
                .and(col("value").neq(lit("асср"))) // exclude асср
                .and(col("value").neq(lit("армавир"))), // exclude армавир
        )
        .collect()?
        .column("value")?
        .as_series()
        .ok_or_else(|| anyhow!("missing column value"))?
        .str()?
        .to_lowercase()
        .into_series();

    let keywords = stem_series(&read(&base_path, "keywords.csv")?, &stemmer)?;

    Ok(Filters {
        names,
        midnames,
        surnames,
        geonames,
        keywords,
    })
}

fn stem_series(series: &Series, stemmer: &Stemmer) -> Result<Series, Error> {
    let rechunked = series.rechunk();
    let stemmed: Vec<String> = rechunked
        .str()?
        .into_iter()
        .filter_map(|opt_val| {
            opt_val.and_then(|val| {
                let stem = stemmer.stem(val).into_owned();
                // Filter out short stems and Russian stems
                if stem.chars().count() >= 4
                /* && !RUSSIAN_STEMS.contains(stem.as_str()) */
                {
                    Some(stem)
                } else {
                    None
                }
            })
        })
        .collect();
    Ok(Series::new(series.name().clone(), stemmed))
}

fn stem_text_to_words(text: &str, stemmer: &Stemmer) -> impl Iterator<Item = String> {
    // Common Russian words that collide with Armenian name stems
    static RUSSIAN_COMMON_WORDS: phf::Set<&'static str> = phf_set! {
        "торосы",
        "торосов",
        "торосам",
        "торосами",
        "торосах", // ice ridges (plural/oblique cases) - collides with Armenian name Торос
        // Note: "торос" (singular nominative) is NOT blacklisted as it could be the Armenian name
        "тиграи",
        "тиграев",
        "тиграям",
        "тиграями",
        "тиграях", // Tigray people (Ethiopia) - collides with Armenian name Тигран
        "тиграй",
        "тиграйцы",
        "тиграйца",
        "тиграйцев",
        "тиграйцам",
        "бурьян",
        "бурьяна",
        "бурьяну",
        "бурьяном",
        "бурьяне", // Russian word for "weeds/wild grass" - common in descriptions, not Armenian content
        "трунин",
        "трунина",
        "трунину",
        "труниным",
        "трунине", // Russian surname Trunin - collides with Armenian surname Трунян
        // Russian cultural figures (painters, writers, public figures) - checked before stemming
        "головко",     // Арсений Головко - Soviet admiral
        // NOTE: "левитан" removed - Isaac Levitan (И. И. Левитан 1860-1900) was ethnically Armenian
        //       His works appear in Armenian galleries, so filtering would lose legitimate content
        "репин",       // Илья Репин - Russian painter
        "васнецов",    // Виктор Васнецов - Russian painter
        "серов",       // Валентин Серов - Russian painter
        "суриков",     // Василий Суриков - Russian painter
        "шишкин",      // Иван Шишкин - Russian painter
        "крамской",    // Иван Крамской - Russian painter
        "перов",       // Василий Перов - Russian painter
        "саврасов",    // Алексей Саврасов - Russian painter
        "лермонтов",   // Mikhail Lermontov - Russian poet
        "пушкин",      // Alexander Pushkin - Russian poet
        "достоевский", // Fyodor Dostoevsky - Russian writer
        "толстой",     // Leo Tolstoy - Russian writer
        "чехов",       // Anton Chekhov - Russian writer
        "горький",     // Maxim Gorky - Russian writer
        "маяковский",  // Vladimir Mayakovsky - Russian poet
        // Middle Eastern artifacts (not Armenian-specific)
        "кальян",      // hookah/water pipe - Ottoman/Middle Eastern artifact
        "кальяна",
        "кальяну",
        "кальяном",
        "кальяне",
        "кальяны",
        "кальянов",
        "кальянам",
        "кальянами",
        "кальянах",    // hookah (all case forms) - false positive for Armenian content
        // Greek archaeological artifacts (not Armenian-specific)
        "килик",       // kylix - ancient Greek drinking cup
        "килика",
        "килику",
        "киликом",
        "килике",
        "килики",
        "киликов",
        "киликам",
        "киликами",
        "киликах",     // kylix (all case forms) - found in archaeological sites but not Armenian-specific
    };

    text.split(|c: char| !c.is_alphabetic())
        .filter(|word| !word.is_empty())
        .filter_map(|word| {
            let lower = word.to_lowercase();

            // Skip common Russian words that collide with Armenian stems
            if RUSSIAN_COMMON_WORDS.contains(lower.as_str()) {
                return None;
            }

            let stem = stemmer.stem(&lower).into_owned();
            if stem.chars().count() >= 4 && !RUSSIAN_STEMS.contains(stem.as_str()) {
                Some(stem)
            } else {
                None
            }
        })
}

fn preprocess_parquet(input_path: &str, output_path: &str, filters: &Filters) -> Result<(), Error> {
    use std::collections::HashSet;
    use std::sync::Arc;

    eprintln!("Preprocessing parquet file...");

    // Prepare filter series for keyword matching
    //
    let mut filter_set: HashSet<String> = HashSet::new();
    for list in [
        &filters.names,
        &filters.midnames,
        &filters.surnames,
        &filters.keywords,
        &filters.geonames,
    ] {
        for word in list
            .str()
            .iter()
            .flat_map(|chunk| chunk.iter())
            .flat_map(|opt| opt.into_iter())
        {
            filter_set.insert(word.to_lowercase());
        }
    }
    let filter_set = Arc::new(filter_set);
    let filter_set_1 = filter_set.clone();

    eprintln!("Total filter terms: {}", filter_set.len());
    eprintln!("Geonames filter terms: {}", filters.geonames.len());

    let stemmer = Arc::new(Stemmer::create(Algorithm::Russian));

    let stemmer_clone1 = stemmer.clone();
    let stemmer_clone2 = stemmer.clone();

    fn filter_column(
        col: Column,
        output_name: &'static str,
        stemmer: &Stemmer,
        filter: &HashSet<String>,
    ) -> PolarsResult<Column> {
        let s = col.str()?;
        let values: BooleanChunked = s
            .into_iter()
            .map(|opt_val| {
                opt_val
                    .map(|val| {
                        stem_text_to_words(val, &stemmer)
                            .into_iter()
                            .any(|word| filter.contains(&word))
                    })
                    .unwrap_or_default()
            })
            .collect();
        Ok(Column::new(output_name.into(), values.into_series()))
    }

    let lf = LazyFrame::scan_parquet(
        PlPath::from_str(input_path),
        ScanArgsParquet {
            low_memory: true,
            ..Default::default()
        },
    )?
    .with_column(
        col("name")
            .map(
                move |col: Column| {
                    filter_column(col, "matched_name", &stemmer_clone1, &filter_set.clone())
                },
                |_schema, _fields| Ok(Field::new("matched_name".into(), DataType::Boolean)),
            )
            .alias("matched_name"),
    )
    .with_column(
        col("description")
            .map(
                move |col: Column| {
                    filter_column(
                        col,
                        "matched_description",
                        &stemmer_clone2,
                        &filter_set_1.clone(),
                    )
                },
                |_schema, _fields| Ok(Field::new("matched_description".into(), DataType::Boolean)),
            )
            .alias("matched_description"),
    )
    // Add column to track if matched by geonames (reliable, no BERT needed)
    .with_column(
        (col("productionPlace")
            .is_not_null()
            .and(col("productionPlace").str().len_bytes().gt(0))
            .and(col("productionPlace").contains_word_case_insensitive(&filters.geonames)))
        .or(col("findPlace")
            .is_not_null()
            .and(col("findPlace").str().len_bytes().gt(0))
            .and(col("findPlace").contains_word_case_insensitive(&filters.geonames)))
        .alias("matched_geonames"),
    )
    .with_column(
        col("matched_name")
            .or(col("matched_description"))
            .alias("matched_keywords_or_names"),
    )
    // Filter to only records that matched something
    .filter(col("matched_geonames").or(col("matched_keywords_or_names")))
    // Drop stemmed columns - no longer needed after filtering
    .drop(by_name(["matched_name", "matched_description"], false));

    eprintln!("Writing preprocessed (filtered) parquet...");
    lf.sink_parquet(
        SinkTarget::Path(PlPath::from_str(output_path)),
        Default::default(),
        None,
        Default::default(),
    )?
    .collect_with_engine(Engine::Streaming)?;

    eprintln!("Preprocessing complete!");
    Ok(())
}

trait ExprExt {
    fn contains_word_case_insensitive(self, included: &Series) -> Expr;
}

impl ExprExt for Expr {
    fn contains_word_case_insensitive(self, included: &Series) -> Expr {
        let pattern = format!(
            "(?i)\\b({})",
            included
                .iter()
                .map(|any| regex::escape(any.str_value().as_ref()))
                .collect::<Vec<String>>()
                .join("|")
        );
        self.str().contains(lit(pattern), true)
    }
}

fn main() -> Result<(), Error> {
    let data_path = "data/data.parquet";
    let preprocessed_path = "data/data-preprocessed.parquet";
    let filters_path = "data/armenian-keywords/data/ru/";

    // Load filters (needed for preprocessing)
    let filters = load_filters(filters_path)?;

    // Preprocess if needed (now includes keyword/geoname filtering)
    if !std::path::Path::new(preprocessed_path).exists() {
        eprintln!("Preprocessed file not found, creating it...");
        preprocess_parquet(data_path, preprocessed_path, &filters)?;
    }

    // Load preprocessed data (already filtered by keywords/geonames)
    let lf = LazyFrame::scan_parquet(PlPath::from_str(preprocessed_path), Default::default())?;

    let df = lf.clone().select([len()]).collect()?;

    let records_count = df.column("len")?.u32()?.get(0).unwrap();

    println!("Loaded preprocessed parquet. records: {}", records_count);
    // Initialize BERT classifier for semantic scoring
    eprintln!("Initializing BERT classifier...");
    let classifier = BertClassifier::load()?;

    let classifier = Arc::new(classifier);

    const BERT_THRESHOLD: f32 = 0.25;

    // Apply BERT scoring to all records
    // Geoname matches get 1.0, keyword/name matches get BERT scored
    let classifier_clone = classifier.clone();

    let pb = Arc::new(ProgressBar::new(records_count as u64).with_prefix("Scoring"));

    let _ = lf
        .with_column(
            when(col("matched_keywords_or_names").and(col("matched_geonames").not()))
                .then(
                    // Match training data format: use " | " separator and "Автор: " prefix
                    concat_str(
                        [
                            col("name"),
                            col("description"),
                            when(col("authors").list().len().gt(0))
                                .then(concat_str(
                                    [lit("Автор: "), col("authors").list().join(lit(", "), true)],
                                    "",
                                    false,
                                ))
                                .otherwise(lit("")),
                        ],
                        " | ",
                        true,
                    )
                    .map(
                        move |c: Column| {
                            let c = c.str()?;
                            let classifier_clone = classifier_clone.clone();

                            use itertools::Itertools;

                            let pb = pb.clone();

                            let res: Float32Chunked = c
                                .into_iter()
                                .chunks(16)
                                .into_iter()
                                .map(move |chunk| {
                                    let chunk: Vec<&str> = chunk
                                        .into_iter()
                                        .map(|x: Option<&str>| x.unwrap_or_default())
                                        .collect();

                                    let classifier = classifier_clone.clone();

                                    pb.inc(chunk.len() as u64);

                                    classifier
                                        .score_batch(&chunk)
                                        .unwrap()
                                        .into_iter()
                                        .map(Some)
                                })
                                .flatten()
                                .collect();
                            Ok(Column::new("armenian_score".into(), res.into_series()))
                        },
                        |_, _| Ok(Field::new("armenian_score".into(), DataType::Float32)),
                    ),
                )
                .otherwise(lit(1.0f32)) // Geoname matches get perfect score (skip BERT)
                .alias("armenian_score"),
        )
        // Filter: keep all geoname matches, and keyword/name matches above threshold
        .filter(col("matched_geonames").or(col("armenian_score").gt_eq(lit(BERT_THRESHOLD))))
        // Drop temporary columns used for filtering
        .drop(by_name(
            ["matched_geonames", "matched_keywords_or_names"],
            false,
        ))
        .with_new_streaming(true)
        .sink_json(
            SinkTarget::Path(PlPath::from_str("result.json")),
            Default::default(),
            None,
            Default::default(),
        )?
        .collect_with_engine(Engine::Streaming)?;

    Ok(())
}
