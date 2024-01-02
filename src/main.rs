use akira::{LlamaGeneration, LlamaGenerationOptionsBuilder, TextGeneration, TextGenerationOptionsBuilder};

#[tokio::main]
async fn main() {
    let option = LlamaGenerationOptionsBuilder::default()
        .use_gpu(true)
        .model_path("/home/cyberdz/File/llm-model/codellama-7b-instruct.Q4_K_M.gguf".to_string())
        .parallelism(1)
        .build().unwrap();
    let llama = LlamaGeneration::new(option);
    let gen_option = TextGenerationOptionsBuilder::default()
        .max_input_length(2048)
        .max_decoding_length(1920)
        .sampling_temperature(0.1)
        .build().unwrap();
    let res = llama.generate("#include<iostream>\nusing namespace std;\n\n// a gcd function", gen_option).await;
    println!("{}", res);
}