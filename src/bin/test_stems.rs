use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;
use std::fs;

fn main() {
    let golovko_text = r#"Арсений Григорьевич Головко. Книга "Вместе с флотом." Издание 3-е. Военные мемуары. Москва. "Финансы и статистика".1984 год. 288 страниц в плотной коленкоровой обложке голубого цвета. Мемуары о Северном флоте в годы Великой Отечественной войны. На первом листе экслибрис Николая Петровича Трунина - жителя Мурманска, подарившего книгу из личной библиотеки литературнгому отделу музея в Кобоне."#;

    let stemmer = Stemmer::create(Algorithm::Russian);

    // Load Armenian filter keywords
    let armenian_keywords: HashSet<String> = fs::read_to_string("data/armenian-keywords/data/ru/names.csv")
        .unwrap_or_default()
        .lines()
        .filter_map(|line| line.split(',').next().map(|s| s.to_lowercase()))
        .chain(fs::read_to_string("data/armenian-keywords/data/ru/surnames.csv")
            .unwrap_or_default()
            .lines()
            .filter_map(|line| line.split(',').next().map(|s| s.to_lowercase())))
        .chain(fs::read_to_string("data/armenian-keywords/data/ru/keywords.csv")
            .unwrap_or_default()
            .lines()
            .filter_map(|line| line.split(',').next().map(|s| s.to_lowercase())))
        .collect();

    println!("Testing Golovko text stems:");
    println!("{}", "=".repeat(60));

    let words: Vec<&str> = golovko_text.split(|c: char| !c.is_alphabetic()).filter(|w| !w.is_empty()).collect();

    for word in words {
        let lower = word.to_lowercase();
        let stem = stemmer.stem(&lower);

        if armenian_keywords.contains(stem.as_ref()) {
            println!("✓ MATCH: word='{}' -> stem='{}' (matches Armenian keyword)", word, stem);
        }
    }
}
