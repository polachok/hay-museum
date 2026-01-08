use polars::prelude::*;
use anyhow::Error;

fn main() -> Result<(), Error> {
    // Read the preprocessed parquet file
    let df = LazyFrame::scan_parquet(
        PlPath::from_str("data/data-preprocessed.parquet"),
        Default::default()
    )?
    .filter(col("id").eq(lit(15714786)))  // Polovyanny record ID
    .select([
        col("id"),
        col("name"),
        col("productionPlace"),
        col("name_stemmed"),
        col("description_stemmed"),
    ])
    .collect()?;

    println!("Polovyanny Record (ID 15714786):\n");

    let name_stems = df.column("name_stemmed")?.list()?;
    let desc_stems = df.column("description_stemmed")?.list()?;

    println!("Name: {}", df.column("name")?.str()?.get(0).unwrap());
    println!("\nName stems:");
    if let Some(series) = name_stems.get_as_series(0) {
        for val in series.str()?.into_iter().flatten() {
            println!("  {}", val);
        }
    }

    println!("\nDescription stems:");
    if let Some(series) = desc_stems.get_as_series(0) {
        for val in series.str()?.into_iter().flatten() {
            println!("  {}", val);
        }
    }

    Ok(())
}
