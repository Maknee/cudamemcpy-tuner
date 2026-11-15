//! Pinned package downloads. Installers and archive members are never executed.
//! This module owns the manifest, cache identity and extraction. Hash pins are
//! input identity, not evidence that a runtime patch is safe. Table membership
//! is explicit: survey inputs can be fetched without enabling generated rows.
//! Only the requested regular libcuda file is copied out of an archive.

use crate::image::Arch;
use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    time::Duration,
};
use tempfile::NamedTempFile;

const MAX_PACKAGE: u64 = 2 << 30;
const MAX_LIBRARY: u64 = 1 << 30;

/// Table entries are emitted by generate/verify. Scanned entries are emitted too, but only
/// the static scan vouches for them (no ground truth), so the hook uses them only on opt-in.
/// Survey entries are fetchable experiments and never become patch rows just because a scan
/// succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    Table,
    Scanned,
    Survey,
}
impl Purpose {
    pub fn name(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Scanned => "scanned",
            Self::Survey => "survey",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Driver {
    pub arch: Arch,
    pub purpose: Purpose,
    pub version: String,
    pub url: String,
    pub package_sha256: String,
    pub member: String,
    pub library_sha256: String,
}

impl Driver {
    pub fn library(&self, cache: &Path) -> PathBuf {
        cache
            .join(self.arch.to_string())
            .join(&self.version)
            .join(format!("libcuda.so.{}", self.version))
    }
}

pub fn manifest(path: &Path) -> Result<Vec<Driver>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read manifest {}", path.display()))?;
    let mut rows = Vec::new();
    let mut keys = BTreeSet::new();
    for (line_number, line) in text.lines().enumerate() {
        let fields: Vec<_> = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        if fields.is_empty() {
            continue;
        }
        ensure!(
            fields.len() == 7,
            "manifest line {}: expected seven fields (last is table|survey)",
            line_number + 1
        );
        let arch = match fields[0] {
            "x86_64" => Arch::X86_64,
            "aarch64" => Arch::Aarch64,
            other => bail!(
                "manifest line {}: unsupported architecture {other}",
                line_number + 1
            ),
        };
        ensure!(
            valid_version(fields[1]),
            "invalid driver version {}",
            fields[1]
        );
        ensure!(
            keys.insert((arch, fields[1].to_owned())),
            "duplicate manifest row: {arch} {}",
            fields[1]
        );
        for digest in [fields[3], fields[5]] {
            ensure!(
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
                "invalid SHA-256 on manifest line {}",
                line_number + 1
            );
        }
        let url = reqwest::Url::parse(fields[2]).context("parse package URL")?;
        ensure!(
            url.scheme() == "https" && url.username().is_empty() && url.password().is_none(),
            "package URL must use HTTPS without credentials"
        );
        ensure!(
            url.path().ends_with(".deb") || url.path().ends_with(".run"),
            "unsupported package type: {url}"
        );
        normalized_member(Path::new(fields[4]))?;
        let purpose = match fields[6] {
            "table" => Purpose::Table,
            "scanned" => Purpose::Scanned,
            "survey" => Purpose::Survey,
            other => bail!(
                "manifest line {}: unknown purpose {other}; expected table, scanned or survey",
                line_number + 1
            ),
        };
        rows.push(Driver {
            arch,
            purpose,
            version: fields[1].to_owned(),
            url: fields[2].to_owned(),
            package_sha256: fields[3].to_owned(),
            member: fields[4].to_owned(),
            library_sha256: fields[5].to_owned(),
        });
    }
    ensure!(!rows.is_empty(), "empty driver manifest");
    Ok(rows)
}

pub fn valid_version(version: &str) -> bool {
    let parts: Vec<_> = version.split('.').collect();
    (2..=3).contains(&parts.len())
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|c| c.is_ascii_digit()))
}

pub fn validate_versions(versions: &[String]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for version in versions {
        ensure!(valid_version(version), "invalid version {version}");
        ensure!(seen.insert(version), "duplicate version {version}");
    }
    Ok(())
}

pub fn select<'a>(
    manifest: &'a [Driver],
    arch: Option<Arch>,
    versions: &[String],
) -> Result<Vec<&'a Driver>> {
    validate_versions(versions)?;
    let selected: Vec<_> = manifest
        .iter()
        .filter(|d| {
            arch.is_none_or(|arch| d.arch == arch)
                && (versions.is_empty() || versions.contains(&d.version))
        })
        .collect();
    for version in versions {
        ensure!(
            selected.iter().any(|d| &d.version == version),
            "version {version} is absent from the manifest for this architecture"
        );
    }
    ensure!(
        !selected.is_empty(),
        "selection matches no manifest entries"
    );
    Ok(selected)
}

pub fn hash_file(path: &Path) -> Result<String> {
    let mut input = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 << 10];
    loop {
        let count = input
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn verify_file(path: &Path, expected: &str) -> Result<()> {
    let actual = hash_file(path)?;
    ensure!(
        actual == expected,
        "SHA-256 mismatch for {}: expected {expected}, got {actual}",
        path.display()
    );
    Ok(())
}

fn normalized_member(path: &Path) -> Result<PathBuf> {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => (),
            Component::Normal(name) => result.push(name),
            _ => bail!("unsafe archive member {}", path.display()),
        }
    }
    ensure!(!result.as_os_str().is_empty(), "empty archive member");
    Ok(result)
}

fn http_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(1800))
        .user_agent("cmt-sigscan/0.1")
        .build()
        .context("create HTTPS client")
}

pub fn fetch(driver: &Driver, cache: &Path) -> Result<PathBuf> {
    let library = driver.library(cache);
    if library.exists() && verify_file(&library, &driver.library_sha256).is_ok() {
        eprintln!(
            "{} {}: cached library verified",
            driver.arch, driver.version
        );
        return Ok(library);
    }
    let directory = library
        .parent()
        .context("library has no parent directory")?;
    let packages = directory.join("pkg");
    fs::create_dir_all(&packages)?;
    let url = reqwest::Url::parse(&driver.url)?;
    let filename = url
        .path_segments()
        .and_then(|mut p| p.next_back())
        .context("package URL has no filename")?;
    let package = packages.join(filename);
    if !package.exists() || verify_file(&package, &driver.package_sha256).is_err() {
        let client = http_client()?;
        if filename.ends_with(".run") {
            let mut response = client
                .get(format!("{}.sha256sum", driver.url))
                .send()?
                .error_for_status()?;
            let mut text = String::new();
            response.by_ref().take(4097).read_to_string(&mut text)?;
            ensure!(
                text.len() <= 4096
                    && text.split_whitespace().next() == Some(driver.package_sha256.as_str()),
                "published installer hash differs from the pinned manifest"
            );
        }
        eprintln!(
            "{} {}: downloading {}",
            driver.arch, driver.version, driver.url
        );
        let mut response = client
            .get(&driver.url)
            .send()
            .context("download package")?
            .error_for_status()?;
        ensure!(
            response.content_length().is_none_or(|n| n <= MAX_PACKAGE),
            "package exceeds 2 GiB limit"
        );
        let mut temporary = NamedTempFile::new_in(&packages)?;
        let bytes = io::copy(&mut response.by_ref().take(MAX_PACKAGE + 1), &mut temporary)?;
        ensure!(bytes <= MAX_PACKAGE, "package exceeds 2 GiB limit");
        temporary.flush()?;
        verify_file(temporary.path(), &driver.package_sha256)?;
        temporary
            .persist(&package)
            .map_err(|e| e.error)
            .context("publish verified package")?;
    }
    // A download or extraction failure leaves an existing library unchanged.
    let mut temporary = NamedTempFile::new_in(directory)?;
    extract(&package, driver, temporary.as_file_mut())?;
    temporary.flush()?;
    verify_file(temporary.path(), &driver.library_sha256)?;
    temporary
        .persist(&library)
        .map_err(|e| e.error)
        .context("publish verified library")?;
    eprintln!(
        "{} {}: extracted library verified",
        driver.arch, driver.version
    );
    Ok(library)
}

pub fn extract(package: &Path, driver: &Driver, output: &mut File) -> Result<()> {
    if package.extension().is_some_and(|e| e == "deb") {
        let mut archive = ar::Archive::new(File::open(package)?);
        let mut found = false;
        while let Some(entry) = archive.next_entry() {
            let entry = entry.context("read Debian archive")?;
            let name = String::from_utf8_lossy(entry.header().identifier())
                .trim_end_matches('/')
                .to_owned();
            if !name.starts_with("data.tar") {
                continue;
            }
            ensure!(!found, "multiple data archives in Debian package");
            let reader: Box<dyn Read + '_> = match name.as_str() {
                "data.tar.xz" => Box::new(xz2::read::XzDecoder::new(entry)),
                "data.tar.zst" => Box::new(zstd::stream::read::Decoder::new(entry)?),
                "data.tar.gz" => Box::new(flate2::read::GzDecoder::new(entry)),
                "data.tar" => Box::new(entry),
                _ => bail!("unsupported Debian data archive {name}"),
            };
            extract_tar(reader, &driver.member, output)?;
            found = true;
        }
        ensure!(found, "Debian package has no data archive");
    } else {
        let (input, compression) = run_payload(package, driver)?;
        let reader: Box<dyn Read> = match compression.as_str() {
            "xz" => Box::new(xz2::read::XzDecoder::new(input)),
            "zstd" => Box::new(zstd::stream::read::Decoder::new(input)?),
            _ => unreachable!("run_payload validates compression"),
        };
        extract_tar(reader, &driver.member, output)?;
    }
    Ok(())
}

fn extract_tar(reader: impl Read, member: &str, output: &mut File) -> Result<()> {
    let wanted = normalized_member(Path::new(member))?;
    let mut archive = tar::Archive::new(reader);
    let mut found = false;
    for entry in archive.entries().context("read tar archive")? {
        let mut entry = entry.context("read tar member")?;
        let path = entry.path().context("read tar member name")?;
        // Unselected entries are never unpacked. A malicious path cannot escape
        // because the selected bytes go to an already-open temporary file.
        if normalized_member(&path).ok().as_ref() != Some(&wanted) {
            continue;
        }
        ensure!(!found, "duplicate archive member {member}");
        ensure!(
            entry.header().entry_type().is_file(),
            "selected member is not a regular file: {member}"
        );
        ensure!(entry.size() <= MAX_LIBRARY, "library exceeds 1 GiB limit");
        let copied = io::copy(&mut entry, output)?;
        ensure!(copied == entry.size(), "truncated library member");
        found = true;
    }
    ensure!(found, "archive member not found: {member}");
    Ok(())
}

fn run_payload(path: &Path, driver: &Driver) -> Result<(BufReader<File>, String)> {
    let mut input = BufReader::new(File::open(path)?);
    let mut header = Vec::new();
    input.by_ref().take(65536).read_to_end(&mut header)?;
    let text = String::from_utf8_lossy(&header);
    let field = |pattern: &str| -> Result<String> {
        let regex = regex::Regex::new(pattern)?;
        let matches: Vec<_> = regex.captures_iter(&text).collect();
        ensure!(
            matches.len() == 1,
            "installer header requires one field matching {pattern}"
        );
        Ok(matches[0][1].to_owned())
    };
    let skip: u64 = field(r"(?m)^skip=([0-9]+)$")?.parse()?;
    let decompress: u64 = field(r"(?m)^skip_decompress=([0-9]+)$")?.parse()?;
    let decompressor_lines: u64 = field(r"(?m)^size_decompress=([0-9]+)$")?.parse()?;
    let version = field(r"(?m)^version_string=([0-9.]+)$")?;
    let arch = field(r#"(?m)^TARGET_ARCH="([a-z0-9_]+)"$"#)?;
    let compression = field(r#"(?m)^ *echo "  Compression +: ([a-z0-9]+)"$"#)?;
    ensure!(
        version == driver.version && arch == driver.arch.to_string(),
        "installer version/architecture differs from manifest"
    );
    ensure!(
        skip > 0 && skip <= 1_000_000 && decompress.checked_add(decompressor_lines) == Some(skip),
        "unsupported installer payload boundary"
    );
    let magic: &[u8] = match compression.as_str() {
        "xz" => b"\xfd7zXZ\0",
        "zstd" => b"\x28\xb5\x2f\xfd",
        _ => bail!("unsupported installer compression {compression}"),
    };
    input.seek(SeekFrom::Start(0))?;
    let mut line = Vec::new();
    for _ in 1..skip {
        line.clear();
        ensure!(
            input.read_until(b'\n', &mut line)? > 0,
            "installer ends before payload"
        );
    }
    ensure!(
        input.fill_buf()?.starts_with(magic),
        "installer payload has wrong compression magic"
    );
    Ok((input, compression))
}
