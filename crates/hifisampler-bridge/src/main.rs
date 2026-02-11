//! HiFiSampler Bridge - OpenUTAU/UTAU client executable.
//!
//! This binary replaces the original C# hifisampler.exe.
//! It acts as a bridge between OpenUTAU and the HiFiSampler server:
//! 1. Checks if the server is running
//! 2. If not, starts the server automatically
//! 3. Forwards UTAU resample parameters via HTTP POST
//!
//! Double-Checked Locking (DCL) pattern ensures only one instance starts the server.

use anyhow::Result;
use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const SERVER_HOST: &str = "127.0.0.1";
const SERVER_PORT: u16 = 8572;
const MAX_STARTUP_WAIT_SECS: u64 = 90;
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY_MS: u64 = 500;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        println!("HiFiSampler Bridge v{}", env!("CARGO_PKG_VERSION"));
        println!("Usage: hifisampler <UTAU resample arguments>");
        return Ok(());
    }

    let base_url = format!("http://{}:{}", SERVER_HOST, SERVER_PORT);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    // ── Step 1: Quick check if server is running ──
    if !is_server_ready(&client, &base_url) {
        // ── Step 2: Acquire mutex and double-check ──
        let _mutex = acquire_startup_mutex();

        if !is_server_ready(&client, &base_url) {
            // ── Step 3: Start server ──
            start_server()?;

            // ── Step 4: Wait for server to be ready ──
            wait_for_server(&client, &base_url)?;
        }
    }

    // ── Step 5: Send inference request ──
    let body = args.join(" ");
    send_request_with_retry(&client, &base_url, &body)?;

    Ok(())
}

/// Check if the server is ready.
fn is_server_ready(client: &reqwest::blocking::Client, base_url: &str) -> bool {
    match client
        .get(base_url)
        .timeout(Duration::from_secs(2))
        .send()
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Start the server process.
fn start_server() -> Result<()> {
    let exe_dir = env::current_exe()?
        .parent()
        .unwrap_or(&PathBuf::from("."))
        .to_path_buf();

    // Look for server binary
    let server_names = if cfg!(windows) {
        vec!["hifisampler-server.exe"]
    } else {
        vec!["hifisampler-server"]
    };

    let mut server_path = None;
    for name in &server_names {
        let path = exe_dir.join(name);
        if path.exists() {
            server_path = Some(path);
            break;
        }
    }

    let server_path = server_path.ok_or_else(|| {
        anyhow::anyhow!(
            "Cannot find hifisampler-server binary in {}",
            exe_dir.display()
        )
    })?;

    eprintln!("Starting server: {}", server_path.display());

    Command::new(&server_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    Ok(())
}

/// Wait for the server to become ready.
fn wait_for_server(client: &reqwest::blocking::Client, base_url: &str) -> Result<()> {
    let start = Instant::now();
    let timeout = Duration::from_secs(MAX_STARTUP_WAIT_SECS);

    while start.elapsed() < timeout {
        if is_server_ready(client, base_url) {
            eprintln!("Server ready after {:.1}s", start.elapsed().as_secs_f64());
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
    }

    anyhow::bail!(
        "Server did not become ready within {}s",
        MAX_STARTUP_WAIT_SECS
    );
}

/// Send inference request with retry logic.
fn send_request_with_retry(
    client: &reqwest::blocking::Client,
    base_url: &str,
    body: &str,
) -> Result<()> {
    for attempt in 0..MAX_RETRIES {
        match client
            .post(base_url)
            .body(body.to_string())
            .send()
        {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(());
                }

                let body_text = resp.text().unwrap_or_default();

                if status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                    || status == reqwest::StatusCode::INTERNAL_SERVER_ERROR
                {
                    if attempt < MAX_RETRIES - 1 {
                        eprintln!(
                            "Request failed ({}), retrying ({}/{})",
                            status,
                            attempt + 1,
                            MAX_RETRIES
                        );
                        thread::sleep(Duration::from_millis(RETRY_DELAY_MS));
                        continue;
                    }
                }

                anyhow::bail!("Server error {}: {}", status, body_text);
            }
            Err(e) => {
                if attempt < MAX_RETRIES - 1 {
                    eprintln!(
                        "Connection error, retrying ({}/{}): {}",
                        attempt + 1,
                        MAX_RETRIES,
                        e
                    );
                    thread::sleep(Duration::from_millis(RETRY_DELAY_MS));
                    continue;
                }
                anyhow::bail!("Connection failed after {} retries: {}", MAX_RETRIES, e);
            }
        }
    }

    anyhow::bail!("Failed after {} retries", MAX_RETRIES);
}

/// Platform-specific mutex for startup synchronization.
#[cfg(windows)]
fn acquire_startup_mutex() -> MutexGuard {
    use std::ffi::CString;

    unsafe {
        let name =
            CString::new("Global\\HifiSamplerServerStartupMutex_DCL_8572").unwrap();
        let handle = windows_sys::Win32::System::Threading::CreateMutexA(
            std::ptr::null_mut(),
            0, // not initially owned
            name.as_ptr() as *const u8,
        );

        if handle.is_null() {
            eprintln!("Warning: Failed to create mutex, proceeding without lock");
            return MutexGuard { handle: std::ptr::null_mut() };
        }

        windows_sys::Win32::System::Threading::WaitForSingleObject(
            handle,
            30000, // 30s timeout
        );

        MutexGuard { handle }
    }
}

#[cfg(windows)]
struct MutexGuard {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(windows)]
impl Drop for MutexGuard {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                windows_sys::Win32::System::Threading::ReleaseMutex(self.handle);
                windows_sys::Win32::Foundation::CloseHandle(self.handle);
            }
        }
    }
}

#[cfg(not(windows))]
fn acquire_startup_mutex() -> MutexGuard {
    use std::fs::OpenOptions;
    use std::io::Write;

    let lock_path = std::env::temp_dir().join("hifisampler_server.lock");
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .ok();

    // On Unix, we'd use flock() - simplified here
    MutexGuard { _file: file }
}

#[cfg(not(windows))]
struct MutexGuard {
    _file: Option<std::fs::File>,
}
