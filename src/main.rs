use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Error, anyhow};
use polars::lazy::dsl::by_name;
use polars::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};

use mimalloc::MiMalloc;

mod bert_classifier;
use bert_classifier::BertClassifier;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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
    // Exclude Armenian surnames that are identical to common Russian words
    // and Russian first names incorrectly listed as Armenian surnames
    let russian_stopwords = ["потеря", "крестья", "емельян", "емелья"];

    // Also exclude stems that would match Russian patronymics and Russian proper names
    let problematic_stems = [
        "василь", // from Васильян, matches Васильевич
        "грабар", // classical Armenian word, but matches Russian artist И.Э.Грабарь (Igor Grabar)
        "андрия", // Armenian surname Андриян, but matches Russian first name Андриян (e.g. cosmonaut)
        "демья",  // from Демьян (Armenian surname), matches Демьян (Russian first name)
        "татья",  // from Татьян (Armenian surname), matches Татьяна (Russian first name)
        "оловя", // from Оловян (Armenian surname), matches оловянный (Russian: tin/pewter - very common in museum items)
        "сафья", // from Сафьян (Armenian surname), matches сафьян (Russian: morocco leather - common in book bindings)
    ];

    let rechunked = series.rechunk();
    let stemmed: Vec<String> = rechunked
        .str()?
        .into_iter()
        .filter_map(|opt_val| {
            opt_val.and_then(|val| {
                let stem = stemmer.stem(val).into_owned();
                // Filter out short stems, Russian stopwords, and problematic stems
                if stem.chars().count() >= 5
                    && !russian_stopwords.contains(&stem.as_str())
                    && !problematic_stems.contains(&stem.as_str())
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

fn stem_text_to_words(text: &str, stemmer: &Stemmer) -> Vec<String> {
    // Exclude Armenian surnames that are identical to common Russian words
    // and Russian first names incorrectly listed as Armenian surnames
    let russian_stopwords = [
        "потеря",
        "крестья",
        "емельян",
        "емелья",
        "грабар",
        "андрия",
        "демья",
        "татья",
        "оловя",
        "сафья",
    ];

    // Common Russian patronymics that cause false positives with Armenian surnames
    // Only include those that actually collide (e.g., Васильевич collides with Васильян)
    let russian_patronymics = [
        "васильевич",
        "васильевна", // from Василий, collides with Васильян surname
    ];

    // Common Russian words that collide with Armenian name stems
    let russian_common_words = [
        "торосы",
        "торосов",
        "торосам",
        "торосами",
        "торосах", // ice ridges (plural/oblique cases) - collides with Armenian name Торос
        // Note: "торос" (singular nominative) is NOT blacklisted as it could be the Armenian name
        "григорий",
        "григория",
        "григорию",
        "григорием",
        "григорье", // Russian first name Gregory - collides with Григорян surname
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
    ];

    text.split(|c: char| !c.is_alphabetic())
        .filter(|word| !word.is_empty())
        .filter_map(|word| {
            let lower = word.to_lowercase();

            // Skip specific Russian patronymics that cause false positives
            if russian_patronymics.contains(&lower.as_str()) {
                return None;
            }

            // Skip common Russian words that collide with Armenian stems
            if russian_common_words.contains(&lower.as_str()) {
                return None;
            }

            let stem = stemmer.stem(&lower).into_owned();
            if stem.chars().count() >= 5 && !russian_stopwords.contains(&stem.as_str()) {
                Some(stem)
            } else {
                None
            }
        })
        .collect()
}

fn preprocess_parquet(input_path: &str, output_path: &str) -> Result<(), Error> {
    use std::sync::Arc;

    eprintln!("Preprocessing parquet file...");
    let stemmer = Arc::new(Stemmer::create(Algorithm::Russian));

    let stemmer_clone1 = stemmer.clone();
    let stemmer_clone2 = stemmer.clone();

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
                    let s = col.as_materialized_series();
                    let values: Vec<AnyValue> = s
                        .str()?
                        .into_iter()
                        .map(|opt_val| {
                            let words = opt_val
                                .map(|val| stem_text_to_words(val, &stemmer_clone1))
                                .unwrap_or_default();
                            let word_series = Series::from_iter(words);
                            AnyValue::List(word_series)
                        })
                        .collect();
                    Ok(Series::from_any_values(s.name().clone(), &values, false)?.into())
                },
                |_schema, _fields| {
                    Ok(Field::new(
                        "name_stemmed".into(),
                        DataType::List(Box::new(DataType::String)),
                    ))
                },
            )
            .alias("name_stemmed"),
    )
    .with_column(
        col("description")
            .map(
                move |col: Column| {
                    let s = col.as_materialized_series();
                    let values: Vec<AnyValue> = s
                        .str()?
                        .into_iter()
                        .map(|opt_val| {
                            let words = opt_val
                                .map(|val| stem_text_to_words(val, &stemmer_clone2))
                                .unwrap_or_default();
                            let word_series = Series::from_iter(words);
                            AnyValue::List(word_series)
                        })
                        .collect();
                    Ok(Series::from_any_values(s.name().clone(), &values, false)?.into())
                },
                |_schema, _fields| {
                    Ok(Field::new(
                        "description_stemmed".into(),
                        DataType::List(Box::new(DataType::String)),
                    ))
                },
            )
            .alias("description_stemmed"),
    );

    eprintln!("Writing preprocessed parquet...");
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

    // Preprocess if needed
    if !std::path::Path::new(preprocessed_path).exists() {
        eprintln!("Preprocessed file not found, creating it...");
        preprocess_parquet(data_path, preprocessed_path)?;
    }

    let filters_path = "data/armenian-keywords/data/ru/";
    let filters = load_filters(filters_path)?;

    let mut description_filter = Series::new_empty("desc".into(), &DataType::String);
    description_filter.append(&filters.names)?;
    description_filter.append(&filters.midnames)?;
    description_filter.append(&filters.surnames)?;
    description_filter.append(&filters.keywords)?;
    let description_filter = description_filter.unique()?.rechunk();

    eprintln!("Total filter terms: {}", description_filter.len());
    eprintln!("Geonames filter terms: {}", filters.geonames.len());

    let lf = LazyFrame::scan_parquet(PlPath::from_str(preprocessed_path), Default::default())?
        .filter(
            // geographical (word boundaries - fewer terms)
            col("productionPlace")
                .contains_word_case_insensitive(&filters.geonames)
                .or(col("findPlace").contains_word_case_insensitive(&filters.geonames))
                // name filtering using stemmed word lists with fast is_in()
                .or(col("name_stemmed")
                    .list()
                    .eval(col("").is_in(lit(description_filter.clone()).implode(), false))
                    .list()
                    .sum()
                    .gt(0))
                .or(col("description_stemmed")
                    .list()
                    .eval(col("").is_in(lit(description_filter.clone()).implode(), false))
                    .list()
                    .sum()
                    .gt(0)),
        );

    // Initialize BERT classifier for semantic scoring
    eprintln!("Initializing BERT classifier...");
    let mut classifier = BertClassifier::load()?;
    classifier.create_armenian_prototypes()?;
    let classifier = Arc::new(classifier);

    const BERT_THRESHOLD: f32 = 0.44;

    // Apply BERT scoring and filtering
    let classifier_clone = classifier.clone();
    let _ = lf
        .with_column(
            concat_str([col("name"), col("description")], " ", true)
                .map(
                    move |c: Column| {
                        let c = c.str()?;
                        let classifier_clone = classifier_clone.clone();
                        let res: Float32Chunked = c
                            .into_iter()
                            .map(move |text: Option<&str>| -> Option<f32> {
                                let classifier = classifier_clone.clone();
                                text.and_then(move |t| classifier.score_armenian_relevance(t).ok())
                            })
                            .collect();
                        Ok(Column::new("armenian_score".into(), res.into_series()))
                    },
                    |_, _| Ok(Field::new("armenian_score".into(), DataType::Float32)),
                )
                .alias("armenian_score"),
        )
        .filter(col("armenian_score").gt_eq(lit(BERT_THRESHOLD)))
        .drop(by_name(["name_stemmed", "description_stemmed"], false))
        .with_new_streaming(true)
        .sink_json(
            SinkTarget::Path(PlPath::from_str("test.csv")),
            Default::default(),
            None,
            Default::default(),
        )?
        .collect_with_engine(Engine::Streaming)?;

    Ok(())
}
