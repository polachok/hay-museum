use anyhow::Result;
use hay_museum::bert_classifier::BertClassifier;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<()> {
    println!("Loading test records from test.csv...");

    // Read borderline records (0.44-0.50) from test.csv
    let file = File::open("test.csv")?;
    let reader = BufReader::new(file);

    let mut test_records: Vec<(String, String, f32)> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(record) = serde_json::from_str::<serde_json::Value>(&line) {
            let score = record["armenian_score"].as_f64().unwrap_or(0.0) as f32;

            // Only test borderline records (0.44-0.55)
            if score >= 0.44 && score < 0.55 {
                let name = record["name"].as_str().unwrap_or("").to_string();
                let desc = record["description"].as_str().unwrap_or("").to_string();
                let text = format!("{} {}", name, desc);

                test_records.push((name, text, score));

                if test_records.len() >= 30 {
                    break;
                }
            }
        }
    }

    println!("Testing {} borderline records\n", test_records.len());

    // Load model
    println!("Loading BERT model...");
    let mut classifier = BertClassifier::load()?;

    // Test 3 different prototype sets
    let prototype_sets = vec![
        (
            "Current (Generic)",
            vec![
                ("cultural", vec![
                    "армянская культура",
                    "армянское искусство",
                    "армянское наследие",
                ]),
                ("geographic", vec![
                    "Армения",
                    "Ереван",
                    "Карабах",
                    "Закавказье",
                ]),
                ("historical", vec![
                    "история Армении",
                    "армянская история",
                    "армянский народ",
                ]),
                ("linguistic", vec![
                    "армянский язык",
                    "армянский текст",
                    "армянская письменность",
                ]),
            ],
        ),
        (
            "Improved (Museum-focused)",
            vec![
                ("people", vec![
                    "армянский композитор Хачатурян",
                    "армянский художник Айвазовский",
                    "армянский писатель Шагинян",
                    "маршал Баграмян",
                    "Анастас Микоян советский деятель",
                ]),
                ("places", vec![
                    "Ереван столица Армении",
                    "Армянская ССР советская республика",
                    "монастырь в Армении",
                    "армянская картинная галерея",
                    "Карабах исторический регион",
                ]),
                ("culture", vec![
                    "армянский театр имени Сундукяна",
                    "армянская музыка и балет",
                    "килика армянское церковное облачение",
                    "армянская архитектура церкви",
                ]),
                ("artifacts", vec![
                    "экспонат из Армении",
                    "артефакт армянской культуры",
                    "монета республики Армения",
                    "армянский музейный предмет",
                ]),
            ],
        ),
        (
            "Specific (Name/Place-heavy)",
            vec![
                ("names", vec![
                    "Хачатурян Арам композитор",
                    "Айвазовский Иван художник",
                    "Баграмян маршал Советского Союза",
                    "Микоян Анастас государственный деятель",
                    "Шагинян Мариэтта писательница",
                    "Сарьян Мартирос художник",
                ]),
                ("geography", vec![
                    "Ереван город Армения",
                    "Армянская ССР республика",
                    "Тбилиси Закавказье армяне",
                    "Нагорный Карабах регион",
                    "землетрясение в Армении",
                ]),
                ("institutions", vec![
                    "Государственный театр Армении Сундукяна",
                    "Картинная галерея Армении Ереван",
                    "Лазаревский институт армянский",
                    "армянская церковь монастырь храм",
                ]),
                ("surnames", vec![
                    "Петросян армянская фамилия",
                    "Гамбарян армянин",
                    "Мартиросян из Армении",
                    "Арутюнян армянское имя",
                    "Григорян Саркисян Оганян",
                ]),
            ],
        ),
    ];

    for (set_name, prototypes) in &prototype_sets {
        println!("\n{}", "=".repeat(60));
        println!("Testing: {}", set_name);
        println!("{}\n", "=".repeat(60));

        // Create prototypes
        classifier.armenian_prototypes.clear();
        for (category, phrases) in prototypes {
            for phrase in phrases {
                if let Ok(embedding) = classifier.encode_text(phrase) {
                    classifier.armenian_prototypes.push((category.to_string(), embedding));
                }
            }
        }

        println!("Created {} prototype embeddings", classifier.armenian_prototypes.len());

        // Test on records
        let mut above_threshold = 0;
        let mut big_changes = Vec::new();

        for (i, (name, text, original_score)) in test_records.iter().enumerate() {
            let new_score = classifier.score_armenian_relevance(text)?;

            if new_score >= 0.44 {
                above_threshold += 1;
            }

            let diff = new_score - original_score;
            if diff.abs() > 0.10 {
                let display_name = if name.chars().count() > 35 {
                    name.chars().take(35).collect::<String>() + "..."
                } else {
                    name.clone()
                };
                big_changes.push((i + 1, display_name, *original_score, new_score, diff));
            }
        }

        println!("\nResults:");
        println!("  Records above threshold (0.44): {}/{}", above_threshold, test_records.len());
        println!("  Big score changes (>0.10): {}", big_changes.len());

        if !big_changes.is_empty() {
            println!("\nSignificant changes:");
            for (idx, name, old, new, diff) in big_changes.iter().take(10) {
                println!("  #{}: {}", idx, name);
                println!("     Old: {:.4} → New: {:.4} (Δ {:.4})", old, new, diff);
            }
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Comparison complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}
