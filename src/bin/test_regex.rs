use regex::Regex;

fn main() {
    // Test if geonames match "Германия"
    let geonames = ["эривань", "герюсы", "ани", "ереван"];

    let pattern = format!(
        "(?i)\\b({})",
        geonames
            .iter()
            .map(|s| regex::escape(s))
            .collect::<Vec<String>>()
            .join("|")
    );

    println!("Pattern: {}\n", pattern);

    let re = Regex::new(&pattern).unwrap();

    let test_strings = vec![
        "Германия",
        "Германия, г. Нюринберг",
        "герюсы",
        "эривань",
    ];

    for test in test_strings {
        let matches = re.is_match(test);
        println!("{:30} -> {}", test, if matches { "MATCH" } else { "no match" });
    }
}
