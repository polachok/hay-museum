use rust_stemmers::{Algorithm, Stemmer};
use std::env;

fn main() {
    let stemmer = Stemmer::create(Algorithm::Russian);
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        println!("Usage: stem_words <word1> <word2> ...");
        return;
    }

    println!("Testing words:\n");
    for word in args {
        let lower = word.to_lowercase();
        let stem = stemmer.stem(&lower);
        let len = stem.chars().count();
        println!("{:25} → {:15} ({} chars)", word, stem, len);
    }
}
