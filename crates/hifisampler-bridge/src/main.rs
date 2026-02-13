//! HiFiSampler Bridge - OpenUTAU/UTAU client executable.
//!
//! Minimal bridge between OpenUTAU and the HiFiSampler server.
//! Uses raw TCP sockets for HTTP — no external dependencies — to keep
//! the binary as small as possible (~200KB stripped).
//!
//! Flow:
//! 1. Check if the server is running (GET /)
//! 2. If not, acquire a startup mutex → start the server
//! 3. Forward UTAU resample parameters via HTTP POST /

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const HOST: &str = "127.0.0.1";
const PORT: u16 = 8572;
const ADDR: &str = "127.0.0.1:8572";
const MAX_STARTUP_WAIT: Duration = Duration::from_secs(90);
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY: Duration = Duration::from_millis(500);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const SERVER_PATH_CONFIG_FILE: &str = "hifisampler-server.path";
const MANAGED_IDLE_TIMEOUT_SECS: u64 = 600;

fn main() {
    if let Err(msg) = run() {
        eprintln!("Error: {msg}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        println!("HiFiSampler Bridge v{}", env!("CARGO_PKG_VERSION"));
        println!("Usage: hifisampler <UTAU resample arguments>");
        return Ok(());
    }

    // ── Step 1: Quick check if server is running ──
    if !is_server_ready() {
        // ── Step 2: Acquire mutex and double-check ──
        let _mutex = acquire_startup_mutex();

        if !is_server_ready() {
            // ── Step 3: Start server ──
            start_server()?;

            // ── Step 4: Wait for server to be ready ──
            wait_for_server()?;
        }
    }

    // ── Step 5: Send inference request ──
    let body = args.join(" ");
    send_request_with_retry(&body)
}

// ─────────────────── Minimal HTTP over raw TCP ───────────────────

/// Send a GET / and return true if status is 2xx/3xx.
fn is_server_ready() -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&ADDR.parse().unwrap(), CONNECT_TIMEOUT) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(CONNECT_TIMEOUT));
    let req = format!("GET / HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 32];
    let Ok(n) = stream.read(&mut buf) else {
        return false;
    };
    // Check "HTTP/1.x 2xx" or "HTTP/1.x 3xx"
    let head = std::str::from_utf8(&buf[..n]).unwrap_or("");
    matches!(head.get(9..10), Some("2") | Some("3"))
}

/// Send a POST / with the given body. Returns Ok if 2xx, otherwise Err with details.
fn http_post(body: &str) -> Result<(), String> {
    let mut stream = TcpStream::connect_timeout(&ADDR.parse().unwrap(), REQUEST_TIMEOUT)
        .map_err(|e| format!("Connection failed: {e}"))?;

    stream
        .set_read_timeout(Some(REQUEST_TIMEOUT))
        .map_err(|e| format!("Set timeout: {e}"))?;

    let req = format!(
        "POST / HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nContent-Length: {}\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("Write failed: {e}"))?;

    // Read response (we only need the status line)
    let mut resp = vec![0u8; 4096];
    let n = stream
        .read(&mut resp)
        .map_err(|e| format!("Read failed: {e}"))?;
    let resp_str = String::from_utf8_lossy(&resp[..n]);

    // Parse status code from "HTTP/1.x NNN ..."
    let status: u16 = resp_str
        .get(9..12)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if (200..300).contains(&status) {
        Ok(())
    } else {
        // Extract body after \r\n\r\n for error message
        let body_part = resp_str
            .split_once("\r\n\r\n")
            .map(|(_, b)| b)
            .unwrap_or(&resp_str);
        Err(format!("Server error {status}: {body_part}"))
    }
}

// ─────────────────── Server lifecycle ───────────────────

fn start_server() -> Result<(), String> {
    let server_path = resolve_server_path()?;
    let server_dir = server_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    eprintln!("Starting server: {}", server_path.display());
    Command::new(&server_path)
        .arg("--managed")
        .arg("--idle-timeout-secs")
        .arg(MANAGED_IDLE_TIMEOUT_SECS.to_string())
        .current_dir(&server_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Spawn failed: {e}"))?;

    Ok(())
}

fn resolve_server_path() -> Result<PathBuf, String> {
    let bridge_exe = env::current_exe().map_err(|e| format!("Cannot determine exe path: {e}"))?;
    let exe_dir = bridge_exe
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let cfg_path = exe_dir.join(SERVER_PATH_CONFIG_FILE);
    let expected_name = expected_server_binary_name();
    if cfg_path.exists() {
        let raw = fs::read_to_string(&cfg_path)
            .map_err(|e| format!("Cannot read {}: {e}", cfg_path.display()))?;
        let line = raw
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty() && !line.starts_with('#'))
            .ok_or_else(|| {
                format!(
                    "{} is empty. Expected first non-empty line to be server path.",
                    cfg_path.display()
                )
            })?;

        let configured_raw = line.trim_matches('"');
        let configured = normalize_config_path(configured_raw);
        let resolved = if configured.is_absolute() {
            configured.clone()
        } else {
            exe_dir.join(configured)
        };

        if is_trusted_server_binary(&resolved, expected_name) {
            return Ok(resolved);
        }

        #[cfg(windows)]
        {
            let alt = strip_windows_verbatim_prefix(configured_raw);
            if alt != configured_raw {
                let alt_path = PathBuf::from(&alt);
                let alt_resolved = if alt_path.is_absolute() {
                    alt_path
                } else {
                    exe_dir.join(&alt)
                };
                if is_trusted_server_binary(&alt_resolved, expected_name) {
                    return Ok(alt_resolved);
                }
            }
        }

        return Err(format!(
            "Server path from {} is invalid (must be file named {}): {}",
            cfg_path.display(),
            expected_name,
            resolved.display()
        ));
    }

    let name = if cfg!(windows) {
        "hifisampler-server.exe"
    } else {
        "hifisampler-server"
    };
    let fallback = exe_dir.join(name);
    if is_trusted_server_binary(&fallback, expected_name) {
        return Ok(fallback);
    }

    Err(format!(
        "Cannot find server binary. Tried {} and {}. Install bridge from WebUI Setup first.",
        cfg_path.display(),
        fallback.display()
    ))
}

fn expected_server_binary_name() -> &'static str {
    if cfg!(windows) {
        "hifisampler-server.exe"
    } else {
        "hifisampler-server"
    }
}

fn is_trusted_server_binary(path: &Path, expected_name: &str) -> bool {
    if !path.is_file() {
        return false;
    }
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    if cfg!(windows) {
        name.eq_ignore_ascii_case(expected_name)
    } else {
        name == expected_name
    }
}

fn normalize_config_path(raw: &str) -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(strip_windows_verbatim_prefix(raw))
    }
    #[cfg(not(windows))]
    {
        PathBuf::from(raw)
    }
}

#[cfg(windows)]
fn strip_windows_verbatim_prefix(path: &str) -> String {
    if let Some(rest) = path.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{}", rest);
    }
    if let Some(rest) = path.strip_prefix(r"\\?\") {
        return rest.to_string();
    }
    if let Some(rest) = path.strip_prefix(r"\??\") {
        return rest.to_string();
    }
    path.to_string()
}

fn wait_for_server() -> Result<(), String> {
    let start = Instant::now();
    while start.elapsed() < MAX_STARTUP_WAIT {
        if is_server_ready() {
            eprintln!("Server ready after {:.1}s", start.elapsed().as_secs_f64());
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
    }
    Err(format!(
        "Server did not become ready within {}s",
        MAX_STARTUP_WAIT.as_secs()
    ))
}

fn send_request_with_retry(body: &str) -> Result<(), String> {
    for attempt in 0..MAX_RETRIES {
        match http_post(body) {
            Ok(()) => return Ok(()),
            Err(e) => {
                if attempt < MAX_RETRIES - 1 {
                    eprintln!(
                        "Request failed, retrying ({}/{}): {e}",
                        attempt + 1,
                        MAX_RETRIES
                    );
                    thread::sleep(RETRY_DELAY);
                } else {
                    return Err(format!("Failed after {MAX_RETRIES} retries: {e}"));
                }
            }
        }
    }
    Err(format!("Failed after {MAX_RETRIES} retries"))
}

// ─────────────────── Platform mutex ───────────────────

#[cfg(windows)]
fn acquire_startup_mutex() -> MutexGuard {
    use std::ffi::CString;

    unsafe {
        let name = CString::new("Global\\HifiSamplerServerStartupMutex_DCL_8572").unwrap();
        let handle = windows_sys::Win32::System::Threading::CreateMutexA(
            std::ptr::null_mut(),
            0,
            name.as_ptr() as *const u8,
        );

        if handle.is_null() {
            eprintln!("Warning: Failed to create mutex, proceeding without lock");
            return MutexGuard {
                handle: std::ptr::null_mut(),
            };
        }

        windows_sys::Win32::System::Threading::WaitForSingleObject(handle, 30000);
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
    let lock_path = std::env::temp_dir().join("hifisampler_server.lock");
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .ok();
    MutexGuard { _file: file }
}

#[cfg(not(windows))]
struct MutexGuard {
    _file: Option<std::fs::File>,
}
