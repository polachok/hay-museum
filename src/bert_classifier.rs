use anyhow::{Result, anyhow};
use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::ops::softmax;
use candle_nn::{Linear, VarBuilder, linear};
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

/// Model type selection for BERT classifier
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    /// LaBSE: Language-agnostic BERT (109 languages, multilingual)
    LaBSE,
    /// RuBERT-tiny2: cointegrated/rubert-tiny2 (Russian-specific, 12M params, fast)
    #[allow(dead_code)]
    RuBERTTiny2,
    /// Fine-tuned RuBERT-mini for Armenian classification (local model)
    FineTunedArmenian,
}

impl ModelType {
    fn model_id(&self) -> &str {
        match self {
            ModelType::LaBSE => "sentence-transformers/LaBSE",
            ModelType::RuBERTTiny2 => "cointegrated/rubert-tiny2",
            ModelType::FineTunedArmenian => "py/rubert-mini-armenian/final",
        }
    }

    fn is_local(&self) -> bool {
        matches!(self, ModelType::FineTunedArmenian)
    }
}

pub struct BertClassifier {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    #[allow(dead_code)]
    model_type: ModelType,
    pub armenian_prototypes: Vec<(String, Tensor)>,
    pub russian_prototypes: Vec<(String, Tensor)>,
    /// Classification head for fine-tuned models (None for embedding-only models)
    classifier: Option<Linear>,
    pooler: Option<Linear>,
}

impl BertClassifier {
    /// Load BERT model from HuggingFace Hub
    #[allow(dead_code)]
    pub fn load() -> Result<Self> {
        Self::load_with_model(ModelType::LaBSE)
    }

    /// Load BERT model with specific model type
    pub fn load_with_model(model_type: ModelType) -> Result<Self> {
        eprintln!("Loading BERT model ({:?})...", model_type);

        // Auto-detect device: Metal (M1) or CPU
        let device = Self::select_device()?;
        eprintln!("Using device: {:?}", device);

        let model_id = model_type.model_id();

        // Load from local path or HuggingFace
        let (config_path, tokenizer_path, weights_path) = if model_type.is_local() {
            eprintln!("Loading model from local path: {}", model_id);
            let base_path = Path::new(model_id);
            (
                base_path.join("config.json"),
                base_path.join("tokenizer.json"),
                base_path.join("model.safetensors"),
            )
        } else {
            eprintln!("Downloading model files from HuggingFace...");
            let repo = Repo::new(model_id.to_string(), RepoType::Model);
            let api = Api::new()?;
            let repo_api = api.repo(repo);
            (
                repo_api.get("config.json")?,
                repo_api.get("tokenizer.json")?,
                repo_api.get("model.safetensors")?,
            )
        };

        eprintln!("Loading model configuration...");
        let config = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config)?;

        eprintln!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        eprintln!("Loading model weights...");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)?
        };

        // Load BERT base model
        let model = BertModel::load(vb.pp("bert"), &config)?;

        // Load classifier head if it's a fine-tuned model
        let (classifier, pooler) = if matches!(model_type, ModelType::FineTunedArmenian) {
            eprintln!("Loading classification head...");
            let hidden_size = config.hidden_size;
            let num_labels = 2;
            let pooler = linear(hidden_size, hidden_size, vb.pp("bert.pooler.dense"))?;
            let classifier = linear(hidden_size, num_labels, vb.pp("classifier"))?;
            (Some(classifier), Some(pooler))
        } else {
            (None, None)
        };

        eprintln!("Model loaded successfully!");

        // Create Armenian prototypes
        let armenian_prototypes = Vec::new(); // Will be populated lazily
        let russian_prototypes = Vec::new(); // Will be populated lazily

        Ok(Self {
            model,
            tokenizer,
            device,
            model_type,
            armenian_prototypes,
            russian_prototypes,
            classifier,
            pooler,
        })
    }

    /// Select best available device (Metal GPU on M1 Mac, or CPU fallback)
    fn select_device() -> Result<Device> {
        // Note: Metal backend currently missing layer-norm implementation for BERT
        // Falling back to CPU until Candle Metal support is complete
        // TODO: Re-enable Metal when layer-norm is supported
        /*
        #[cfg(target_os = "macos")]
        {
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }
        }
        */
        #[cfg(target_os = "macos")]
        {
            Ok(Device::Cpu)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Device::new_cuda(0)?)
        }
    }

    /// Create category-specific Armenian prototype embeddings
    /// Optimized for museum records with specific names, places, and cultural context
    pub fn create_armenian_prototypes(&mut self) -> Result<()> {
        eprintln!("Creating Armenian prototype embeddings...");

        let prototypes = vec![
            (
                "names",
                vec![
                    "Хачатурян Арам композитор",
                    "Айвазовский Иван художник",
                    "Баграмян маршал Советского Союза",
                    "Микоян Анастас государственный деятель",
                    "Шагинян Мариэтта писательница",
                    "Сарьян Мартирос художник",
                ],
            ),
            (
                "geography",
                vec![
                    "Ереван город Армения",
                    "Армянская ССР республика",
                    "Тбилиси Закавказье армяне",
                    "Нагорный Карабах регион",
                    "землетрясение в Армении",
                ],
            ),
            (
                "institutions",
                vec![
                    "Государственный театр Армении Сундукяна",
                    "Картинная галерея Армении Ереван",
                    "Лазаревский институт армянский",
                    "армянская церковь монастырь храм",
                ],
            ),
            (
                "surnames",
                vec![
                    "Петросян армянская фамилия",
                    "Гамбарян армянин",
                    "Мартиросян из Армении",
                    "Арутюнян армянское имя",
                    "Григорян Саркисян Оганян",
                    "Симонян Акопян армянские фамилии",
                    "Аветисян Казарян Манукян",
                    "Варданян Степанян Амбарцумян",
                ],
            ),
        ];

        self.armenian_prototypes.clear();

        for (category, phrases) in prototypes {
            // Encode all phrases in this category
            let embeddings: Vec<Tensor> = phrases
                .iter()
                .filter_map(|phrase| match self.encode_text(phrase) {
                    Ok(emb) => Some(emb),
                    Err(e) => {
                        eprintln!("  Warning: Failed to encode '{}': {}", phrase, e);
                        None
                    }
                })
                .collect();

            if embeddings.is_empty() {
                return Err(anyhow!(
                    "Failed to create embeddings for category: {}",
                    category
                ));
            }

            // Average the embeddings
            let avg_embedding = Self::average_tensors(&embeddings)?;

            self.armenian_prototypes
                .push((category.to_string(), avg_embedding));
            eprintln!("  Created prototype for '{}' category", category);
        }

        eprintln!("Prototype embeddings created successfully!");
        Ok(())
    }

    /// Create Russian/Soviet negative prototypes to filter out non-Armenian content
    /// These embeddings help distinguish Armenian content from Russian cultural items
    pub fn create_russian_prototypes(&mut self) -> Result<()> {
        eprintln!("Creating Russian negative prototype embeddings...");

        let prototypes = vec![
            (
                "russian_geography",
                vec![
                    "Москва столица Россия",
                    "Санкт-Петербург Ленинград город",
                    "Волга река Россия",
                    "Урал горы Россия",
                    "Сибирь регион Россия",
                ],
            ),
            (
                "russian_place_names",
                vec![
                    "Ясная Поляна усадьба Толстой музей",
                    "Лесная поляна совхоз деревня",
                    "Красная поляна село место Россия",
                    "лесная поляна пейзаж природа Россия",
                    "поляна сказок музей Ялта Крым",
                    "поляна лес природа пейзаж",
                ],
            ),
            (
                "russian_orthodox",
                vec![
                    "русская православная церковь",
                    "икона Божией Матери Россия",
                    "православный храм монастырь Россия",
                    "святой Николай Чудотворец икона",
                    "преображение господне православие",
                ],
            ),
            (
                "russian_crafts",
                vec![
                    "гжель народные промыслы Россия",
                    "хохлома роспись русская",
                    "дымковская игрушка промысел",
                    "палех лаковая миниатюра Россия",
                    "жостово поднос роспись русская",
                ],
            ),
            (
                "non_armenian_artists",
                vec![
                    "Микеланджело Буонарроти итальянский скульптор",
                    "Леонардо да Винчи живопись Италия",
                    "Рафаэль Санти художник Возрождение",
                    "Рембрандт ван Рейн голландский художник",
                    "Тициан Вечеллио венецианская живопись",
                    "Караваджо итальянский барокко живопись",
                    "Клод Моне французский импрессионизм",
                    "Огюст Ренуар французский художник",
                    "Поль Сезанн французская живопись",
                    "Винсент ван Гог нидерландский художник",
                    "Пабло Пикассо испанский художник",
                    "Анри Матисс французский фовизм",
                    "Илья Репин русский художник передвижник",
                    "Валентин Серов русская живопись портрет",
                    "Исаак Левитан русский пейзажист",
                    "Иван Шишкин русский художник пейзажист",
                ],
            ),
            (
                "russian_writers",
                vec![
                    "Лев Толстой русский писатель Ясная Поляна",
                    "Федор Достоевский русский писатель романист",
                    "Антон Чехов русский писатель драматург",
                    "Александр Пушкин русский поэт писатель",
                    "Иван Тургенев русский писатель романист",
                    "Николай Гоголь русский писатель драматург",
                ],
            ),
            (
                "soviet_currency",
                vec![
                    "казначейский билет советский рубль банкнота",
                    "банкнота государственного банка СССР рубль",
                    "монета копейка советская чеканка",
                    "червонец золотая советская монета",
                    "денежный знак СССР государственный билет",
                    "рубль копейка советская валюта",
                ],
            ),
        ];

        self.russian_prototypes.clear();

        for (category, phrases) in prototypes {
            let embeddings: Vec<Tensor> = phrases
                .iter()
                .filter_map(|phrase| match self.encode_text(phrase) {
                    Ok(emb) => Some(emb),
                    Err(e) => {
                        eprintln!("  Warning: Failed to encode '{}': {}", phrase, e);
                        None
                    }
                })
                .collect();

            if embeddings.is_empty() {
                return Err(anyhow!(
                    "Failed to create embeddings for category: {}",
                    category
                ));
            }

            let avg_embedding = Self::average_tensors(&embeddings)?;
            self.russian_prototypes
                .push((category.to_string(), avg_embedding));
            eprintln!("  Created negative prototype for '{}' category", category);
        }

        eprintln!("Negative prototype embeddings created successfully!");
        Ok(())
    }

    /// Encode a single text into embedding
    pub fn encode_text(&self, text: &str) -> Result<Tensor> {
        let mut encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        // Truncate to rubert-tiny2's maximum sequence length (1024 tokens)
        encoding.truncate(1024, 0, tokenizers::TruncationDirection::Right);

        let tokens = encoding.get_ids();
        let token_ids = Tensor::new(tokens.to_vec(), &self.device)?.unsqueeze(0)?; // Add batch dimension

        // Forward pass through BERT
        // BERT forward signature: forward(input_ids, token_type_ids, attention_mask)
        let token_type_ids = token_ids.zeros_like()?;
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;

        // Mean pooling over sequence length
        let pooled = Self::mean_pool(&embeddings)?;

        Ok(pooled)
    }

    /// Encode a batch of texts
    #[allow(dead_code)]
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Tensor>> {
        texts.iter().map(|text| self.encode_text(text)).collect()
    }

    /// Score a single record's Armenian relevance
    pub fn score_armenian_relevance(&self, text: &str) -> Result<f32> {
        let res = self.score_armenian_relevance_(text);
        if let Err(err) = &res {
            eprintln!("{:?}", err);
        }
        res
    }
    /// Score a single record's Armenian relevance
    pub fn score_armenian_relevance_(&self, text: &str) -> Result<f32> {
        // Use fine-tuned classifier if available
        if let Some(classifier) = &self.classifier {
            return self.score_with_classifier(text, classifier);
        }

        // Otherwise use prototype-based scoring
        if self.armenian_prototypes.is_empty() {
            return Err(anyhow!(
                "Armenian prototypes not initialized. Call create_armenian_prototypes() first."
            ));
        }

        // Encode the record text
        let record_embedding = self.encode_text(text)?;

        // Calculate max cosine similarity across all Armenian prototypes
        let mut armenian_similarity = 0.0_f32;
        for (_category, prototype) in &self.armenian_prototypes {
            let similarity = Self::cosine_similarity(&record_embedding, prototype)?;
            armenian_similarity = armenian_similarity.max(similarity);
        }

        // Calculate max cosine similarity with Russian negative prototypes (if available)
        let mut russian_similarity = 0.0_f32;
        if !self.russian_prototypes.is_empty() {
            for (_category, prototype) in &self.russian_prototypes {
                let similarity = Self::cosine_similarity(&record_embedding, prototype)?;
                russian_similarity = russian_similarity.max(similarity);
            }
        }

        // Apply negative penalty if Russian similarity is high
        // If russian_similarity > armenian_similarity, reduce the score
        let adjusted_similarity = if russian_similarity > 0.0 {
            // Penalize records that are more Russian than Armenian
            // Penalty increased from 0.2 to 0.5 for better filtering
            let penalty = (russian_similarity - armenian_similarity).max(0.0) * 0.5;
            (armenian_similarity - penalty).max(0.0)
        } else {
            armenian_similarity
        };

        // Apply Armenian character boost
        let has_armenian = Self::has_armenian_characters(text);
        let final_score = if has_armenian {
            (adjusted_similarity * 2.0).min(1.0) // Boost by 2x but cap at 1.0
        } else {
            adjusted_similarity
        };

        Ok(final_score)
    }

    /// Score using fine-tuned classifier
    fn score_with_classifier(&self, text: &str, classifier: &Linear) -> Result<f32> {
        // Tokenize and encode
        let mut encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        encoding.truncate(1024, 0, tokenizers::TruncationDirection::Right);

        let tokens = encoding.get_ids();
        let token_ids = Tensor::new(tokens.to_vec(), &self.device)?.unsqueeze(0)?;

        // Forward pass through BERT
        let token_type_ids = token_ids.zeros_like()?;
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;

        // Get [CLS] token embedding (first token)
        let cls_embedding = embeddings.i((0, 0))?; // Shape: [hidden_size]
        let cls_embedding = cls_embedding.unsqueeze(0)?; // Shape: [1, hidden_size]

        // Pass through classifier
        let logits = classifier.forward(&cls_embedding)?; // Shape: [1, 2]
        let probs = softmax(&logits, candle_core::D::Minus1)?;

        let prob_armenian = probs.i((0, 1))?.to_scalar::<f32>()?;
        // let logits = logits.squeeze(0)?; // Shape: [2]

        // // Get probability for "armenian" class (label 1)
        // // Apply softmax and return probability
        // let logits_vec = logits.to_vec1::<f32>()?;
        // let armenian_logit = logits_vec[1];
        // let not_armenian_logit = logits_vec[0];

        // // Softmax: exp(x_i) / sum(exp(x_j))
        // let exp_armenian = armenian_logit.exp();
        // let exp_not_armenian = not_armenian_logit.exp();
        // let prob_armenian = exp_armenian / (exp_armenian + exp_not_armenian);

        Ok(prob_armenian)
    }

    /// Score a batch of records
    pub fn score_batch(&self, texts: &[&str]) -> Result<Vec<f32>> {
        let Some(classifier) = &self.classifier else {
            anyhow::bail!("no classifier");
        };
        let Some(pooler) = &self.pooler else {
            anyhow::bail!("no pooler");
        };
        let mut tokenizer = self.tokenizer.clone();
        let tokenizer = tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: 1024 + 512,
                strategy: TruncationStrategy::LongestFirst,
                ..Default::default()
            }))
            .unwrap();

        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("Batch tokenization failed: {}", e))?;

        // 2. Prepare Tensors
        let mut all_ids = Vec::new();
        for encoding in encodings {
            all_ids.push(Tensor::new(encoding.get_ids(), &self.device)?);
        }

        // Stack individual tensors into a batch: [batch_size, seq_len]
        let token_ids = Tensor::stack(&all_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        // 3. GPU Forward Pass
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;

        // 4. Extract CLS tokens for the entire batch
        // We take index 0 from the sequence dimension (dim 1)
        let cls_embeddings = embeddings.narrow(1, 0, 1)?.squeeze(1)?; // [batch_size, hidden_size]

        // FIX: Force the tensor into a contiguous memory layout for CUDA
        let cls_embeddings = cls_embeddings.contiguous()?;

        // 1. Pass through Pooler Dense Layer
        let pooled_output = pooler.forward(&cls_embeddings)?;
        // 2. Apply Tanh Activation
        let pooled_output = pooled_output.tanh()?;

        // 5. Classifier + Softmax
        let logits = classifier.forward(&pooled_output)?;
        let probs = softmax(&logits, candle_core::D::Minus1)?; // [batch_size, 2]

        // 6. Convert to Vec for final output
        // This moves the results from GPU to CPU in one single block
        let probs_vec = probs.to_vec2::<f32>()?;

        // Extract the "Armenian" probability (index 1) for each item in the batch
        let scores = probs_vec.iter().map(|p| p[1]).collect();

        Ok(scores)
    }

    /// Check if text contains Armenian characters (U+0530-058F)
    fn has_armenian_characters(text: &str) -> bool {
        text.chars().any(|c| ('\u{0530}'..='\u{058F}').contains(&c))
    }

    /// Calculate cosine similarity between two tensors
    fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
        let dot_product = (a * b)?.sum_all()?.to_scalar::<f32>()?;
        let norm_a = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let norm_b = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;

        let similarity = dot_product / (norm_a * norm_b);
        Ok(similarity)
    }

    /// Average multiple tensors
    fn average_tensors(tensors: &[Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(anyhow!("Cannot average empty tensor list"));
        }

        let sum = tensors
            .iter()
            .skip(1)
            .try_fold(tensors[0].clone(), |acc, t| acc.add(t))?;

        let count = tensors.len() as f64;
        Ok(sum.affine(1.0 / count, 0.0)?)
    }

    /// Mean pooling over sequence dimension
    fn mean_pool(embeddings: &Tensor) -> Result<Tensor> {
        // embeddings shape: [batch_size, seq_len, hidden_size]
        // We want: [batch_size, hidden_size]
        let sum = embeddings.sum(1)?; // Sum over seq_len dimension
        let seq_len = embeddings.dim(1)? as f64;
        Ok(sum.affine(1.0 / seq_len, 0.0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_armenian_char_detection() {
        assert!(BertClassifier::has_armenian_characters("Հայաստան"));
        assert!(BertClassifier::has_armenian_characters(
            "армянский Հայերեն текст"
        ));
        assert!(!BertClassifier::has_armenian_characters("русский текст"));
        assert!(!BertClassifier::has_armenian_characters(""));
    }
}
