use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;

fn main() {
    let stemmer = Stemmer::create(Algorithm::Russian);

    let text = "Коллекция оловянных солдатиков Шагающий солдатик прусской армии с барабаном На постаменте Антикварный оловянный солдатик прусской армии с барабаном На постаменте Шагающий солдатик Окрашен цветной эмалью Сделан в конце 19 века в г Нюринберге Германия в фирме Эрнста Хайнрихссена В коллекцию входят знаменосец 2 пушки 2 конные фигурки офицеры солдаты музыканты барабанщики Турки только шагающая пехота 12 шт прусские всякие разные";

    // Simulate stem_text_to_words logic
    let russian_stopwords = ["потеря", "крестья", "емельян", "емелья", "грабар", "андрия", "демья", "татья"];
    let russian_patronymics = ["васильевич", "васильевна"];
    let russian_common_words = ["торосы", "торосов", "торосам", "торосами", "торосах"];

    let stems: Vec<String> = text
        .split(|c: char| !c.is_alphabetic())
        .filter(|word| !word.is_empty())
        .filter_map(|word| {
            let lower = word.to_lowercase();

            if russian_patronymics.contains(&lower.as_str()) {
                return None;
            }

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
        .collect();

    let unique_stems: HashSet<_> = stems.into_iter().collect();
    let mut sorted_stems: Vec<_> = unique_stems.into_iter().collect();
    sorted_stems.sort();

    println!("5+ char stems extracted from German soldier record:\n");
    for stem in &sorted_stems {
        println!("  {}", stem);
    }

    println!("\nTotal: {} unique 5+ char stems", sorted_stems.len());
}
