#[cfg(feature = "llama_cpp_backend")]
use llama_cpp::{standard_sampler::StandardSampler, LlamaModel, LlamaParams, SessionParams};

#[cfg(feature = "llama_cpp_backend")]
use std::time::Instant;

#[cfg(feature = "llama_cpp_backend")]
fn env_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(feature = "llama_cpp_backend")]
fn arg_value(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

#[cfg(feature = "llama_cpp_backend")]
fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = (values.len() as f64 - 1.0) * p;
    let f = k.floor() as usize;
    let c = (f + 1).min(values.len() - 1);
    if f == c {
        return values[f];
    }
    values[f] + (values[c] - values[f]) * (k - f as f64)
}

#[cfg(feature = "llama_cpp_backend")]
fn run_once(
    model: &LlamaModel,
    prompt: &str,
    max_tokens: usize,
    n_ctx: u32,
) -> Result<(f64, f64, usize, usize), String> {
    let mut sp = SessionParams::default();
    sp.n_ctx = n_ctx;
    sp.n_batch = n_ctx;
    sp.n_ubatch = n_ctx;

    let mut sess = model
        .create_session(sp)
        .map_err(|e| format!("create_session failed: {:?}", e))?;

    sess.advance_context(prompt)
        .map_err(|e| format!("advance_context failed: {:?}", e))?;

    let start = Instant::now();
    let mut first_token_ms = 0.0_f64;
    let mut first = true;
    let mut token_chunks = 0usize;
    let mut text_len = 0usize;

    let completion = sess
        .start_completing_with(StandardSampler::default(), max_tokens)
        .map_err(|e| format!("start_completing_with failed: {:?}", e))?;

    for token in completion.into_strings() {
        token_chunks += 1;
        text_len += token.len();
        if first {
            first_token_ms = start.elapsed().as_secs_f64() * 1000.0;
            first = false;
        }
    }

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((first_token_ms, total_ms, token_chunks, text_len))
}

#[cfg(feature = "llama_cpp_backend")]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path = arg_value(&args, "--model", "");
    if model_path.is_empty() {
        eprintln!("Usage: llama_cpp_microbench --model <path.gguf> [--prompt <text>] [--max-tokens 128] [--n-ctx 2048] [--warmup 1] [--repeats 5] [--conservative 1] [--use-mmap 0/1] [--use-mlock 0/1] [--output <path.json>]");
        std::process::exit(2);
    }

    let prompt = arg_value(
        &args,
        "--prompt",
        "Explain in one short paragraph what Rust ownership is.",
    );
    let max_tokens = arg_value(&args, "--max-tokens", "128")
        .parse::<usize>()
        .unwrap_or(128);
    let n_ctx = arg_value(&args, "--n-ctx", "2048")
        .parse::<u32>()
        .unwrap_or(2048);
    let warmup = arg_value(&args, "--warmup", "1")
        .parse::<usize>()
        .unwrap_or(1);
    let repeats = arg_value(&args, "--repeats", "5")
        .parse::<usize>()
        .unwrap_or(5);
    let conservative = env_bool(&arg_value(&args, "--conservative", "1")).unwrap_or(true);
    let use_mmap = env_bool(&arg_value(&args, "--use-mmap", "1")).unwrap_or(true);
    let use_mlock = env_bool(&arg_value(&args, "--use-mlock", "0")).unwrap_or(false);
    let output_path = arg_value(&args, "--output", "");

    if !std::path::Path::new(&model_path).is_file() {
        eprintln!("Model not found: {}", model_path);
        std::process::exit(1);
    }

    let mut params = LlamaParams::default();
    if conservative {
        params.n_gpu_layers = 0;
        params.use_mmap = false;
        params.use_mlock = false;
    } else {
        params.use_mmap = use_mmap;
        params.use_mlock = use_mlock;
        params.n_gpu_layers = 0;
    }

    let model = match LlamaModel::load_from_file(&model_path, params) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load_from_file failed: {:?}", e);
            std::process::exit(1);
        }
    };

    for _ in 0..warmup {
        if let Err(e) = run_once(&model, &prompt, max_tokens, n_ctx) {
            eprintln!("warmup failed: {}", e);
            std::process::exit(1);
        }
    }

    let mut ttfts: Vec<f64> = Vec::new();
    let mut totals: Vec<f64> = Vec::new();
    let mut chunks: Vec<f64> = Vec::new();
    let mut runs_json: Vec<String> = Vec::new();

    for i in 0..repeats.max(1) {
        match run_once(&model, &prompt, max_tokens, n_ctx) {
            Ok((ttft, total, token_chunks, text_len)) => {
                ttfts.push(ttft);
                totals.push(total);
                chunks.push(token_chunks as f64);
                runs_json.push(format!(
                    "{{\"run\":{},\"ttft_ms\":{:.4},\"total_ms\":{:.4},\"token_chunks\":{},\"text_len\":{}}}",
                    i, ttft, total, token_chunks, text_len
                ));
            }
            Err(e) => {
                eprintln!("run failed: {}", e);
                std::process::exit(1);
            }
        }
    }

    let mean = |v: &Vec<f64>| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };

    let report = format!(
        concat!(
            "{{\n",
            "  \"language\": \"rust\",\n",
            "  \"backend\": \"llama_cpp\",\n",
            "  \"model\": \"{}\",\n",
            "  \"n_ctx\": {},\n",
            "  \"max_tokens\": {},\n",
            "  \"warmup\": {},\n",
            "  \"repeats\": {},\n",
            "  \"conservative\": {},\n",
            "  \"use_mmap\": {},\n",
            "  \"use_mlock\": {},\n",
            "  \"prompt_len\": {},\n",
            "  \"mean_ttft_ms\": {:.4},\n",
            "  \"p50_ttft_ms\": {:.4},\n",
            "  \"p95_ttft_ms\": {:.4},\n",
            "  \"mean_total_ms\": {:.4},\n",
            "  \"p50_total_ms\": {:.4},\n",
            "  \"p95_total_ms\": {:.4},\n",
            "  \"mean_token_chunks\": {:.4},\n",
            "  \"runs\": [{}]\n",
            "}}"
        ),
        model_path.replace('\\', "\\\\"),
        n_ctx,
        max_tokens,
        warmup,
        repeats,
        if conservative { "true" } else { "false" },
        if use_mmap { "true" } else { "false" },
        if use_mlock { "true" } else { "false" },
        prompt.len(),
        mean(&ttfts),
        percentile(ttfts.clone(), 0.5),
        percentile(ttfts.clone(), 0.95),
        mean(&totals),
        percentile(totals.clone(), 0.5),
        percentile(totals, 0.95),
        mean(&chunks),
        runs_json.join(",")
    );

    println!("{}", report);

    if !output_path.is_empty() {
        if let Some(parent) = std::path::Path::new(&output_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(&output_path, &report) {
            eprintln!("failed writing output file: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "llama_cpp_backend"))]
fn main() {
    eprintln!("llama_cpp_microbench requires feature 'llama_cpp_backend'.");
    eprintln!("Run: cargo run --release --manifest-path rust_pipeline/Cargo.toml --features llama_cpp_backend --bin llama_cpp_microbench -- --model <path.gguf>");
    std::process::exit(2);
}
