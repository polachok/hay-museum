use polars::prelude::*;
use anyhow::Error;

fn main() -> Result<(), Error> {
    let ids = vec![13014585_i64, 13146555_i64, 12747546_i64, 10201531_i64, 14981949_i64];

    let df = LazyFrame::scan_parquet(
        PlPath::from_str("data/data-preprocessed.parquet"),
        Default::default()
    )?
    .filter(col("id").is_in(lit(Series::new("".into(), ids)), false))
    .select([
        col("id"),
        col("name"),
        col("productionPlace"),
        col("name_stemmed"),
        col("description_stemmed"),
    ])
    .collect()?;

    for row_idx in 0..df.height() {
        let id = df.column("id")?.i64()?.get(row_idx).unwrap();
        let name = df.column("name")?.str()?.get(row_idx).unwrap();

        println!("=== ID {} ===", id);
        println!("Name: {}\n", name);

        let name_stems = df.column("name_stemmed")?.list()?;
        let desc_stems = df.column("description_stemmed")?.list()?;

        println!("Name stems:");
        if let Some(series) = name_stems.get_as_series(row_idx) {
            for val in series.str()?.into_iter().flatten() {
                println!("  {}", val);
            }
        }

        println!("\nDescription stems (first 20):");
        if let Some(series) = desc_stems.get_as_series(row_idx) {
            for (i, val) in series.str()?.into_iter().flatten().enumerate() {
                if i >= 20 { break; }
                println!("  {}", val);
            }
        }
        println!("\n");
    }

    Ok(())
}
