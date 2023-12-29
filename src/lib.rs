mod llama;
pub mod decoding;

use std::process::exit;
use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use ffi::create_engine;
use futures::stream::BoxStream;
use llama::LlamaService;
use decoding::{helpers, TextGeneration, TextGenerationOptions};

#[cxx::bridge(namespace = "llama")]
mod ffi {
    struct StepOutput {
        request_id: u32,
        text: String,
    }

    unsafe extern "C++" {
        include!("Akira/src/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(
            use_gpu: bool,
            model_path: &str,
            parallelism: u8,
        ) -> UniquePtr<TextInferenceEngine>;

        fn add_request(
            self: Pin<&mut TextInferenceEngine>,
            request_id: u32,
            prompt: &str,
            max_input_length: usize,
        );
        fn stop_request(self: Pin<&mut TextInferenceEngine>, request_id: u32);
        fn step(self: Pin<&mut TextInferenceEngine>) -> Result<Vec<StepOutput>>;
    }
}

unsafe impl Send for ffi::TextInferenceEngine {}

unsafe impl Sync for ffi::TextInferenceEngine {}

#[derive(Builder, Debug)]
pub struct LlamaTextGenerationOptions {
    model_path: String,
    use_gpu: bool,
    parallelism: u8,
}

pub struct LlamaTextGeneration {
    service: LlamaService,
}

impl LlamaTextGeneration {
    pub fn new(options: LlamaTextGenerationOptions) -> Self {
        let engine = create_engine(options.use_gpu, &options.model_path, options.parallelism);
        if engine.is_null() {
            println!("Unable to load model: {}", options.model_path);
            exit(1);
        }

        Self {
            service: LlamaService::new(engine),
        }
    }
}

#[async_trait]
impl TextGeneration for LlamaTextGeneration {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let s = self.generate_stream(prompt, options).await;
        let text = helpers::stream_to_string(s).await;
        return text;
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        let mut rx = self
            .service
            .add_request(prompt, options.max_input_length)
            .await;

        let s = stream! {
            while let Some(new_text) = rx.recv().await {
                yield new_text;
            }

            rx.close();
        };

        Box::pin(s)
    }
}
