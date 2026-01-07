use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct BertClassifier {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    armenian_prototypes: Vec<(String, Tensor)>,
}

impl BertClassifier {
    /// Load BERT model from HuggingFace Hub
    pub fn load() -> Result<Self> {
        eprintln!("Loading BERT model...");

        // Auto-detect device: Metal (M1) or CPU
        let device = Self::select_device()?;
        eprintln!("Using device: {:?}", device);

        // Model: LaBSE (Language-agnostic BERT Sentence Encoder)
        let model_id = "sentence-transformers/LaBSE";
        let repo = Repo::new(model_id.to_string(), RepoType::Model);
        let api = Api::new()?;
        let repo_api = api.repo(repo);

        eprintln!("Downloading model files from HuggingFace...");

        // Download model files
        let config_path = repo_api.get("config.json")?;
        let tokenizer_path = repo_api.get("tokenizer.json")?;
        let weights_path = repo_api.get("model.safetensors")?;

        eprintln!("Loading model configuration...");
        let config = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config)?;

        eprintln!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        eprintln!("Loading model weights...");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)? };
        let model = BertModel::load(vb, &config)?;

        eprintln!("Model loaded successfully!");

        // Create Armenian prototypes
        let armenian_prototypes = Vec::new(); // Will be populated lazily

        Ok(Self {
            model,
            tokenizer,
            device,
            armenian_prototypes,
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
        Ok(Device::Cpu)
    }

    /// Create category-specific Armenian prototype embeddings
    pub fn create_armenian_prototypes(&mut self) -> Result<()> {
        eprintln!("Creating Armenian prototype embeddings...");

        let prototypes = vec![
            (
                "cultural",
                vec![
                    "армянская культура",
                    "армянское искусство",
                    "армянское наследие",
                ],
            ),
            (
                "geographic",
                vec![
                    "Армения",
                    "Ереван",
                    "Карабах",
                    "Закавказье",
                ],
            ),
            (
                "historical",
                vec![
                    "история Армении",
                    "армянская история",
                    "армянский народ",
                ],
            ),
            (
                "linguistic",
                vec![
                    "армянский язык",
                    "армянский текст",
                    "армянская письменность",
                ],
            ),
        ];

        self.armenian_prototypes.clear();

        for (category, phrases) in prototypes {
            // Encode all phrases in this category
            let embeddings: Vec<Tensor> = phrases
                .iter()
                .filter_map(|phrase| {
                    match self.encode_text(phrase) {
                        Ok(emb) => Some(emb),
                        Err(e) => {
                            eprintln!("  Warning: Failed to encode '{}': {}", phrase, e);
                            None
                        }
                    }
                })
                .collect();

            if embeddings.is_empty() {
                return Err(anyhow!("Failed to create embeddings for category: {}", category));
            }

            // Average the embeddings
            let avg_embedding = Self::average_tensors(&embeddings)?;

            self.armenian_prototypes.push((category.to_string(), avg_embedding));
            eprintln!("  Created prototype for '{}' category", category);
        }

        eprintln!("Prototype embeddings created successfully!");
        Ok(())
    }

    /// Encode a single text into embedding
    fn encode_text(&self, text: &str) -> Result<Tensor> {
        let mut encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        // Truncate to BERT's maximum sequence length (512 tokens)
        encoding.truncate(512, 0, tokenizers::TruncationDirection::Right);

        let tokens = encoding.get_ids();
        let token_ids = Tensor::new(tokens.to_vec(), &self.device)?
            .unsqueeze(0)?; // Add batch dimension

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
        texts.iter()
            .map(|text| self.encode_text(text))
            .collect()
    }

    /// Score a single record's Armenian relevance
    pub fn score_armenian_relevance(&self, text: &str) -> Result<f32> {
        if self.armenian_prototypes.is_empty() {
            return Err(anyhow!("Armenian prototypes not initialized. Call create_armenian_prototypes() first."));
        }

        // Encode the record text
        let record_embedding = self.encode_text(text)?;

        // Calculate max cosine similarity across all prototypes
        let mut max_similarity = 0.0_f32;
        for (_category, prototype) in &self.armenian_prototypes {
            let similarity = Self::cosine_similarity(&record_embedding, prototype)?;
            max_similarity = max_similarity.max(similarity);
        }

        // Apply Armenian character boost
        let has_armenian = Self::has_armenian_characters(text);
        let final_score = if has_armenian {
            (max_similarity * 2.0).min(1.0) // Boost by 2x but cap at 1.0
        } else {
            max_similarity
        };

        Ok(final_score)
    }

    /// Score a batch of records
    #[allow(dead_code)]
    pub fn score_batch(&self, texts: &[String]) -> Result<Vec<f32>> {
        texts.iter()
            .map(|text| self.score_armenian_relevance(text))
            .collect()
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

        let sum = tensors.iter()
            .skip(1)
            .try_fold(tensors[0].clone(), |acc, t| acc.add(t))?;

        let count = tensors.len() as f64;
        Ok(sum.affine(1.0 / count, 0.0)?)
    }

    /// Mean pooling over sequence dimension
    fn mean_pool(embeddings: &Tensor) -> Result<Tensor> {
        // embeddings shape: [batch_size, seq_len, hidden_size]
        // We want: [batch_size, hidden_size]
        let sum = embeddings.sum(1)?;  // Sum over seq_len dimension
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
        assert!(BertClassifier::has_armenian_characters("армянский Հայերեն текст"));
        assert!(!BertClassifier::has_armenian_characters("русский текст"));
        assert!(!BertClassifier::has_armenian_characters(""));
    }
}
