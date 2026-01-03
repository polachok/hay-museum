use std::path::PathBuf;

use anyhow::{Error, anyhow};
use polars::prelude::*;

use mimalloc::MiMalloc;

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

    let names = read(&base_path, "names.csv")?;
    let midnames = read(&base_path, "midnames.csv")?;
    let surnames = read(&base_path, "surnames.csv")?;

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

    let keywords = read(&base_path, "keywords.csv")?;

    Ok(Filters {
        names,
        midnames,
        surnames,
        geonames,
        keywords,
    })
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
    let filters_path = "data/armenian-keywords/data/ru/";
    let filters = load_filters(filters_path)?;

    let mut description_filter = Series::new_empty("desc".into(), &DataType::String);
    description_filter.append(&filters.names)?;
    description_filter.append(&filters.midnames)?;
    description_filter.append(&filters.surnames)?;
    description_filter.append(&filters.keywords)?;

    let lf = LazyFrame::scan_parquet(
        PlPath::from_str("data/data.parquet"),
        ScanArgsParquet {
            // parallel: ParallelStrategy::None,
            low_memory: true,
            ..Default::default()
        },
    )?
    .filter(
        // geographical
        col("productionPlace")
            .contains_word_case_insensitive(&filters.geonames)
            .or(col("findPlace").contains_word_case_insensitive(&filters.geonames)), // name
                                                                                     //.or(col("name").contains_word_case_insensitive(&description_filter))
                                                                                     // description
                                                                                     //.or(col("description").contains_word_case_insensitive(&description_filter)),
    );
    let _ = lf //.limit(50)
        .with_new_streaming(true)
        .sink_json(
            SinkTarget::Path(PlPath::from_str("test.csv")),
            Default::default(),
            None,
            Default::default(),
        )?
        .collect_with_engine(Engine::Streaming);
    Ok(())
}
