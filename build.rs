use std::env;
use std::path::PathBuf;
use std::process::exit;

fn check_llama_cpp_cloned() {
    let llama_cpp_makefile = PathBuf::from("llama.cpp/Makefile");
    if !llama_cpp_makefile.exists() {
        println!("Cannot locate {}, llama.cpp needs to be cloned first. Use git submodule update --init before building.", llama_cpp_makefile.display());
        exit(1);
    }
}

fn build_llama_cpp() {
    let mut config = cmake::Config::new("llama.cpp");
    if cfg!(feature = "metal") {
        println!("METAL Feature Enabled");
        config.define("LLAMA_METAL", "ON");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
    }
    if cfg!(feature = "cuda") {
        println!("CUDA Feature Enabled");
        config.define("LLAMA_CUBLAS", "ON");
        config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
        if cfg!(target_os = "windows") {
            let Ok(cuda_path) = env::var("CUDA_PATH") else {
                panic!("CUDA_PATH is not set");
            };
            println!(r"cargo:rustc-link-search=native={}\lib\x64", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-lib=culibos");
        }
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
    }
    println!("Building llama.cpp...");

    let dst = config.build();
    println!("cargo:rustc-link-search=native={}/build", dst.display());
}

fn build_cxx_binding() {
    cxx_build::bridge("src/lib.rs")
        .file("src/engine.cc")
        .flag_if_supported("-Iinclude")
        .flag_if_supported("-Illama.cpp")
        .flag_if_supported("-std=c++14")
        .compile("cxxbridge");
}

fn main() {
    check_llama_cpp_cloned();
    println!("cargo:rustc-link-lib=llama");
    println!("cargo:rustc-link-lib=ggml_static");
    build_llama_cpp();
    build_cxx_binding();
}