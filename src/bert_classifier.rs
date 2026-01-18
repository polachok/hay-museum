use anyhow::{anyhow, Result};
use candle_core::{Device, Module, Tensor};
use candle_nn::ops::softmax;
use candle_nn::{linear, Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

const MODEL_PATH: &str = "py/rubert-mini-armenian/final";

pub struct BertClassifier {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    classifier: Linear,
    pooler: Linear,
}

impl BertClassifier {
    pub fn load() -> Result<Self> {
        eprintln!("Loading BERT model...");

        let device = Self::select_device()?;
        eprintln!("Using device: {:?}", device);

        let base_path = Path::new(MODEL_PATH);
        eprintln!("Loading model from: {}", MODEL_PATH);

        let config_path = base_path.join("config.json");
        let tokenizer_path = base_path.join("tokenizer.json");
        let weights_path = base_path.join("model.safetensors");

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

        let model = BertModel::load(vb.pp("bert"), &config)?;

        eprintln!("Loading classification head...");
        let hidden_size = config.hidden_size;
        let pooler = linear(hidden_size, hidden_size, vb.pp("bert.pooler.dense"))?;
        let classifier = linear(hidden_size, 2, vb.pp("classifier"))?;

        eprintln!("Model loaded successfully!");

        Ok(Self {
            model,
            tokenizer,
            device,
            classifier,
            pooler,
        })
    }

    fn select_device() -> Result<Device> {
        #[cfg(feature = "cuda")]
        {
            return Ok(Device::new_cuda(0)?);
        }

        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }
        }

        Ok(Device::Cpu)
    }

    pub fn score_batch(&self, texts: &[&str]) -> Result<Vec<f32>> {
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

        let mut all_ids = Vec::new();
        let mut all_masks = Vec::new();
        for encoding in encodings {
            all_ids.push(Tensor::new(encoding.get_ids(), &self.device)?);
            all_masks.push(Tensor::new(encoding.get_attention_mask(), &self.device)?);
        }

        let token_ids = Tensor::stack(&all_ids, 0)?;
        let attention_mask = Tensor::stack(&all_masks, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Extract CLS tokens
        let cls_embeddings = embeddings.narrow(1, 0, 1)?.squeeze(1)?;
        let cls_embeddings = cls_embeddings.contiguous()?;

        // Pooler + tanh
        let pooled_output = self.pooler.forward(&cls_embeddings)?;
        let pooled_output = pooled_output.tanh()?;

        // Classifier + softmax
        let logits = self.classifier.forward(&pooled_output)?;
        let probs = softmax(&logits, candle_core::D::Minus1)?;

        let probs_vec = probs.to_vec2::<f32>()?;

        // Extract Armenian probability (index 1) with character boost
        let scores: Vec<_> = probs_vec
            .iter()
            .zip(texts.iter())
            .map(|(p, text)| {
                let score = p[1];
                if Self::has_armenian_characters(text) {
                    (score * 2.0).min(1.0)
                } else {
                    score
                }
            })
            .collect();

        Ok(scores)
    }

    fn has_armenian_characters(text: &str) -> bool {
        text.chars().any(|c| ('\u{0530}'..='\u{058F}').contains(&c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_armenian_char_detection() {
        assert!(BertClassifier::has_armenian_characters("Հdelays այdelays delays"));
        assert!(BertClassifier::has_armenian_characters("армянский Հdelay delays текст"));
        assert!(!BertClassifier::has_armenian_characters("русский текст"));
        assert!(!BertClassifier::has_armenian_characters(""));
    }
}
