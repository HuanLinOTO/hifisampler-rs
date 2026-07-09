//! Convert PyTorch (.pt) model checkpoints to Burn's native Burnpack (.bpk) format.
//!
//! Usage:
//!   convert_models --vocoder <input.pt> <output.bpk>
//!   convert_models --hnsep   <input.pt> <output.bpk>
//!   convert_models --all <vocoder_pt> <vocoder_bpk> <hnsep_pt> <hnsep_bpk>
//!
//! The resulting .bpk files load ~20x faster and use ~50% less peak memory
//! than .pt files (Burnpack: memory-mapped + lazy load + no PyTorch adapter).
//! Inference performance is unchanged — only model load is affected.
//!
//! The conversion is backend-agnostic: a .bpk saved with the wgpu backend
//! can be loaded by the cuda backend (and vice versa).

use std::path::PathBuf;

use clap::Parser;
use hifisampler_core::burn::cascadednet::CascadedNet;
use hifisampler_core::burn::vocoder::HifiGanNsf;
use hifisampler_core::ep::{select_burn_device, VocoderBackend};

#[derive(Parser, Debug)]
#[command(
    name = "convert_models",
    about = "Convert PyTorch .pt checkpoints to Burnpack .bpk format"
)]
struct Args {
    /// Convert vocoder model.
    #[arg(long)]
    vocoder: Option<Vec<PathBuf>>,

    /// Convert hnsep model.
    #[arg(long, num_args = 2)]
    hnsep: Option<Vec<PathBuf>>,

    /// Convert both models in one invocation.
    /// Args: <vocoder_pt> <vocoder_bpk> <hnsep_pt> <hnsep_bpk>
    #[arg(long, num_args = 4)]
    all: Option<Vec<PathBuf>>,
}

fn convert_vocoder(pt: &std::path::Path, bpk: &std::path::Path) -> anyhow::Result<()> {
    println!("[vocoder] loading PT from {} ...", pt.display());
    let t0 = std::time::Instant::now();
    let device = select_burn_device("auto");
    let model = HifiGanNsf::<VocoderBackend>::load_from_pt(pt, &device)?;
    println!(
        "[vocoder] loaded from PT in {:.2}s, saving to BPK {} ...",
        t0.elapsed().as_secs_f64(),
        bpk.display()
    );
    let t1 = std::time::Instant::now();
    model.save_to_bpk(bpk)?;
    println!(
        "[vocoder] saved BPK in {:.2}s ({})",
        t1.elapsed().as_secs_f64(),
        format_bytes(std::fs::metadata(bpk)?.len())
    );
    Ok(())
}

fn convert_hnsep(pt: &std::path::Path, bpk: &std::path::Path) -> anyhow::Result<()> {
    println!("[hnsep] loading PT from {} ...", pt.display());
    let t0 = std::time::Instant::now();
    let device = select_burn_device("auto");
    let model = CascadedNet::<VocoderBackend>::load_from_pt(pt, &device)?;
    println!(
        "[hnsep] loaded from PT in {:.2}s, saving to BPK {} ...",
        t0.elapsed().as_secs_f64(),
        bpk.display()
    );
    let t1 = std::time::Instant::now();
    model.save_to_bpk(bpk)?;
    println!(
        "[hnsep] saved BPK in {:.2}s ({})",
        t1.elapsed().as_secs_f64(),
        format_bytes(std::fs::metadata(bpk)?.len())
    );
    Ok(())
}

fn format_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    if n >= MB {
        format!("{:.1} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.1} KB", n as f64 / KB as f64)
    } else {
        format!("{} B", n)
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let mut did_anything = false;

    if let Some(paths) = args.all {
        let paths = paths;
        if paths.len() != 4 {
            anyhow::bail!("--all expects exactly 4 paths: <vocoder_pt> <vocoder_bpk> <hnsep_pt> <hnsep_bpk>");
        }
        convert_vocoder(&paths[0], &paths[1])?;
        convert_hnsep(&paths[2], &paths[3])?;
        did_anything = true;
    }

    if let Some(paths) = args.vocoder {
        if paths.len() != 2 {
            anyhow::bail!("--vocoder expects exactly 2 paths: <input.pt> <output.bpk>");
        }
        convert_vocoder(&paths[0], &paths[1])?;
        did_anything = true;
    }

    if let Some(paths) = args.hnsep {
        if paths.len() != 2 {
            anyhow::bail!("--hnsep expects exactly 2 paths: <input.pt> <output.bpk>");
        }
        convert_hnsep(&paths[0], &paths[1])?;
        did_anything = true;
    }

    if !did_anything {
        eprintln!("No conversion requested. Use --vocoder, --hnsep, or --all.");
        eprintln!("Example:");
        eprintln!(
            "  convert_models --all models/vocoder/vocoder_fused.pt models/vocoder/vocoder_fused.bpk models/hnsep/hnsep_fused.pt models/hnsep/hnsep_fused.bpk"
        );
        std::process::exit(1);
    }

    println!("\nDone. BPK files are ready for upload to HuggingFace.");
    Ok(())
}
