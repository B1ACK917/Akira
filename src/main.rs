mod decoding;

use Akira::{LlamaTextGeneration, LlamaTextGenerationOptionsBuilder};
use Akira::decoding::{TextGeneration, TextGenerationOptionsBuilder};

#[tokio::main]
async fn main() {
    let mut option = LlamaTextGenerationOptionsBuilder::default();
    option.use_gpu(true);
    option.model_path("/home/cyberdz/File/llm-model/codellama-7b-instruct.Q4_K_M.gguf".to_string());
    option.parallelism(1);
    let mut llama = LlamaTextGeneration::new(option.build().unwrap());
    let gen_option = TextGenerationOptionsBuilder::default()
        .max_input_length(2048)
        .max_decoding_length(1920)
        .sampling_temperature(0.1)
        .build()
        .unwrap();
    let res = llama.generate("#include<iostream>\nusing namespace std;\n\n// a gcd function", gen_option).await;
    println!("{}", res);
}