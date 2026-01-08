use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;

fn main() {
    let stemmer = Stemmer::create(Algorithm::Russian);

    // Armenian 5-char stems that might collide with Russian words
    let armenian_5char: HashSet<&str> = vec![
        // Already blacklisted
        "демья", "татья",
        // Core Armenian terms (should keep)
        "армен", "армян", "арцах", "ерева", "тигра", "аршак",
        "гагик", "килик", "смбат", "торос", "трдат", "артюш",
        // Suspicious - might match Russian words
        "прося", "болтя", "плотя", "микоя", "касья", "папья",
        "кирья", "сарья", "пашья", "вердя", "гатья", "кастя",
    ].into_iter().collect();

    println!("=== Testing if common Russian words collide with Armenian 5-char stems ===\n");
    println!("Armenian 5-char stems to test: {}\n", armenian_5char.len());

    // Expanded Russian test words (common words that appear in museum descriptions)
    let russian_words = vec![
        // Specific tests for suspicious Armenian stems
        "Демьян", "Демьяна", "Демьянов", "Демьяну",  // Russian first name (collides with 'демья')
        "Татьяна", "Татьяны", "Татьяну",  // Russian first name (collides with 'татья')
        "Тула", "Тульский", "Тульская", "Тульское",  // Russian city (might collide with 'тулья')
        "просо", "просяной", "пшено", "просеять",  // Millet/sift (might collide with 'прося')
        "болтать", "болтал", "болтун", "болтовня",  // To chat (might collide with 'болтя')
        "тащить", "тащил", "тащу",  // To drag (might collide with 'тащия')
        "плотник", "плотный", "плотина", "плотно",  // Carpenter/dense (might collide with 'плотя')
        "каса", "касание", "касаться",  // Touch (might collide with 'касья')
        "папа", "папка", "папирус",  // Father/folder (might collide with 'папья')
        "кирпич", "кирка", "киргиз",  // Brick/pick (might collide with 'кирья')
        "сарай", "сарафан", "Саратов",  // Barn/sundress/city (might collide with 'сарья')
        "паша", "пашня", "пахать",  // Pasha/arable land (might collide with 'пашья')
        "зелень", "зеленый", "вердикт",  // Green/verdict (might collide with 'вердя')
        "каста", "кастрюля",  // Caste/pot (might collide with 'кастя')
        // Common words from museum descriptions
        "орден", "ордена", "медаль", "грамота", "билет", "комбинат",
        "пищевой", "микоян", "миха", "петр", "иван", "банка",
        "песня", "фильм", "путь", "грамм", "пластинка", "кофе",
        // More common Russian words
        "армия", "герой", "народ", "война", "труд", "работа",
        "школа", "завод", "фабрика", "город", "деревня", "село",
        "книга", "письмо", "газета", "журнал", "статья", "текст",
        "рука", "нога", "голова", "сердце", "душа", "тело",
        "день", "ночь", "утро", "вечер", "время", "час",
        "год", "месяц", "неделя", "число", "дата", "период",
        "мир", "земля", "небо", "солнце", "луна", "звезда",
        "вода", "огонь", "воздух", "ветер", "снег", "дождь",
        "дом", "квартира", "комната", "окно", "дверь", "стена",
        "стол", "стул", "кровать", "шкаф", "полка", "ящик",
        "хлеб", "мясо", "рыба", "молоко", "яйцо", "масло",
        "платье", "пальто", "рубашка", "юбка", "брюки", "обувь",
    ];

    let mut found_collisions = Vec::new();

    for word in &russian_words {
        let lower = word.to_lowercase();
        let stem = stemmer.stem(&lower);
        if stem.chars().count() == 5 && armenian_5char.contains(stem.as_ref()) {
            found_collisions.push((word, stem.into_owned()));
        }
    }

    if found_collisions.is_empty() {
        println!("✓ No collisions found with tested Russian words!");
    } else {
        println!("⚠ COLLISIONS FOUND:");
        for (word, stem) in found_collisions {
            println!("  Russian '{}' → '{}' (matches Armenian stem!)", word, stem);
        }
    }
}
