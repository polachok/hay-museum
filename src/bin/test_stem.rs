use rust_stemmers::{Algorithm, Stemmer};

fn main() {
    let stemmer = Stemmer::create(Algorithm::Russian);

    let test_words = vec![
        "Татьяна", "Татьяны", "Татьяне",
        "надпись", "надписи",
        "фотография", "фотографии",
        "письмо", "письма",
        "Шагинян", "Шагиняна",
    ];

    println!("Testing suspicious frequent words:\n");

    for word in test_words {
        let lower = word.to_lowercase();
        let stem = stemmer.stem(&lower);
        let len = stem.chars().count();
        println!("{:20} → {:10} ({} chars)", word, stem, len);
    }
}
