// These assembly fixtures predate the Rust port. Their descriptor offsets are
// deliberately different from every supported driver; a version lookup cannot pass.
use super::*;
use drivers::{Driver, Purpose};
use image::{Function, Image, Segment};
use scan::Status;
use std::collections::BTreeMap;
use std::{fs, path::Path};
use tables::TableUpdate;

// Test-only reader of the independent reference table. Production membership
// comes exclusively from drivers.manifest.
fn table_versions(text: &str) -> Result<Vec<String>> {
    let regex = regex::Regex::new(r#"\.version = "([0-9.]+)""#)?;
    let versions: Vec<_> = regex.captures_iter(text).map(|m| m[1].to_owned()).collect();
    drivers::validate_versions(&versions)?;
    ensure!(!versions.is_empty(), "reference table has no versions");
    Ok(versions)
}

fn synthetic(arch: Arch, variant: usize) -> Image {
    let source = include_str!("synthetic/synthetic_code.inc");
    let regex = regex::Regex::new(r"\b0x([0-9a-fA-F]{2})\b").unwrap();
    let all: Vec<u8> = regex
        .captures_iter(source)
        .map(|m| u8::from_str_radix(&m[1], 16).unwrap())
        .collect();
    assert_eq!(all.len(), 2 * 8 * 0xf00);
    let index = (usize::from(arch == Arch::Aarch64) * 8 + variant) * 0xf00;
    let base = 0x10000;
    let mut functions: Vec<_> = [
        (0, 0x400),
        (0x400, 0x80),
        (0x480, 0x80),
        (0x500, 0x100),
        (0x600, 0x100),
        (0x700, 0x100),
    ]
    .into_iter()
    .map(|(offset, size)| Function {
        start: base + offset,
        size,
    })
    .collect();
    if variant == 1 {
        functions.push(Function {
            start: base + 0x900,
            size: 0x400,
        });
    }
    if variant == 7 {
        for offset in [0xd00, 0xe00] {
            functions.push(Function {
                start: base + offset,
                size: 0x100,
            });
        }
    }
    let mut image = Image {
        arch,
        version: "0.0.0".into(),
        image_base: base,
        bytes: all[index..index + 0xf00].to_vec(),
        segments: vec![Segment {
            vaddr: base,
            file_offset: 0,
            file_size: 0xf00,
            executable: true,
        }],
        functions,
        plt: vec![],
        got_symbols: BTreeMap::new(),
    };
    if arch == Arch::X86_64 {
        let (gate, _) = inline::tests::fixture();
        image.bytes.resize(0x1000, 0x90);
        image.bytes.extend(&gate.bytes);
        image.segments[0].file_size = image.bytes.len() as u64;
        image
            .functions
            .extend(gate.functions.iter().map(|f| Function {
                start: base + 0x1000 + f.start - gate.image_base,
                size: f.size,
            }));
    }
    if arch == Arch::Aarch64 {
        image.plt.push(base + 0x800..base + 0x810);
        image.got_symbols.insert(base + 0x20010, "memcpy".into());
    }
    image
}

#[test]
fn independent_assembly_signatures_both_architectures() {
    use Status::*;
    for arch in [Arch::X86_64, Arch::Aarch64] {
        let normal = if arch == Arch::X86_64 { Found } else { Absent };
        let expected = [
            (Found, normal, Found),
            (Ambiguous, NotFound, Found),
            (NotFound, NotFound, Found),
            (
                Found,
                if arch == Arch::X86_64 {
                    NotFound
                } else {
                    Found
                },
                Found,
            ),
            (Found, normal, NotFound),
            (Found, normal, NotFound),
            (Found, normal, Found),
            (Found, normal, Ambiguous),
        ];
        for (variant, (engine, copy, staging)) in expected.into_iter().enumerate() {
            let image = synthetic(arch, variant);
            let result = scan::scan(&image).unwrap();
            for (site, status, offset) in [
                (&result.engine, engine, 0),
                (&result.row_copy, copy, 0x400),
                (&result.staging, staging, 0x600),
            ] {
                assert_eq!(
                    site.status,
                    status,
                    "{arch} variant {variant}: {}",
                    result.explain()
                );
                if status == Found {
                    assert_eq!(site.offset, offset);
                    assert_eq!(site.address, image.image_base + offset);
                    assert_eq!(
                        site.prologue,
                        &image.bytes[offset as usize..offset as usize + 16]
                    );
                } else {
                    assert_eq!(site.offset, 0);
                    assert!(site.prologue.is_empty());
                }
            }
            if engine == Found {
                let values: Vec<_> = result
                    .descriptor
                    .fields()
                    .into_iter()
                    .map(|(_, value)| value)
                    .collect();
                assert_eq!(
                    values,
                    [
                        0xe0, 0, 0x18, 0x20, 0x28, 0x30, 0x48, 0x50, 0x58, 0x70, 0x78, 0x80, 0xc0,
                        0xc8, 0xd0, 0x1d0, 0x1d8, 0x1e0, 0x90
                    ]
                );
                assert_eq!(
                    (
                        result.descriptor.op_kind_1,
                        result.descriptor.op_kind_2,
                        result.descriptor.op_kind_3
                    ),
                    (1, 2, 3)
                );
            } else {
                assert_eq!(result.descriptor.op_stride, 0);
            }
            if engine == Ambiguous {
                assert_eq!(result.engine.candidates, [0x10000, 0x10900]);
            }
            if staging == Ambiguous {
                assert_eq!(result.staging.candidates, [0x10600, 0x10e00]);
            }
            if copy == Absent {
                assert!(result.row_copy.reason.contains("memcpy@plt"));
            }
            assert_eq!(
                result.complete(),
                engine == Found && matches!(copy, Found | Absent) && staging == Found
            );
        }
    }
}

#[test]
fn incomplete_scan_cannot_be_serialized_directly() {
    let image = synthetic(Arch::X86_64, 2); // copy engine intentionally absent
    let scan = scan::scan(&image).unwrap();
    assert!(!scan.complete());
    assert!(tables::row(&scan, false).is_err());
    assert!(tables::render(&[(scan, false)]).is_err());
}

#[test]
fn banner_and_input_validation() {
    assert_eq!(
        image::version_banner(b"foo Version 615.71.09. Copyright (C) bar").unwrap(),
        "615.71.09"
    );
    assert!(image::version_banner(b"Version nonsense").is_err());
    assert!(
        image::version_banner(b"Version 1.2. Copyright (C) Version 2.3. Copyright (C) ").is_err()
    );
    assert!(Image::parse(vec![0; 64]).is_err());
    assert!(Image::parse(b"\x7fELF".to_vec()).is_err());
    assert!(drivers::validate_versions(&["1.2".into(), "1.2".into()]).is_err());
    assert!(drivers::validate_versions(&["../1.2".into()]).is_err());
}

#[test]
fn failed_generation_preserves_both_tables() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("src/gen");
    fs::create_dir_all(&dir).unwrap();
    let original = "/* generated by fixture */\n.version = \"1.2\"\n";
    for arch in [Arch::X86_64, Arch::Aarch64] {
        fs::write(dir.join(arch.table_name()), original).unwrap();
    }
    let manifest = vec![Driver {
        purpose: Purpose::Table,
        arch: Arch::X86_64,
        version: "1.2".into(),
        url: "https://invalid.test/driver.deb".into(),
        package_sha256: "0".repeat(64),
        member: "libcuda.so.1.2".into(),
        library_sha256: "0".repeat(64),
    }];
    assert!(tables::prepare(temp.path(), temp.path(), &manifest, None, false).is_err());
    let library = manifest[0].library(temp.path());
    fs::create_dir_all(library.parent().unwrap()).unwrap();
    fs::write(&library, "corrupt").unwrap();
    assert!(tables::prepare(temp.path(), temp.path(), &manifest, None, false).is_err());
    for arch in [Arch::X86_64, Arch::Aarch64] {
        assert_eq!(
            fs::read_to_string(dir.join(arch.table_name())).unwrap(),
            original
        );
    }
}

#[test]
fn publish_failure_rolls_back_prior_table() {
    let temp = tempfile::tempdir().unwrap();
    let first = temp.path().join("first");
    fs::write(&first, "original").unwrap();
    let second = temp.path().join("second");
    fs::create_dir(&second).unwrap();
    let updates = [
        TableUpdate {
            path: first.clone(),
            original: "original".into(),
            generated: "new".into(),
        },
        TableUpdate {
            path: second,
            original: "old".into(),
            generated: "new".into(),
        },
    ];
    assert!(tables::publish(&updates).is_err());
    assert_eq!(fs::read_to_string(first).unwrap(), "original");
}

#[test]
#[ignore = "reads the cached real driver corpus; run under the measurement lock"]
fn every_checked_in_driver_row_matches_independent_reference() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference = std::env::var_os("CMT_SIGSCAN_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("src/gen"));
    let cache = std::env::var_os("CMT_SIGSCAN_DRIVERS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("tmp/drivers"));
    let manifest = drivers::manifest(&repo.join("tools/sigscan/drivers.manifest")).unwrap();
    for arch in [Arch::X86_64, Arch::Aarch64] {
        let original = fs::read_to_string(reference.join(arch.table_name())).unwrap();
        let explicit_inline = original.contains(".inline_paged_value");
        let original = if explicit_inline {
            original
        } else {
            original
                .lines()
                .filter(|line| !line.trim_start().starts_with(".inline_"))
                .map(|line| format!("{line}\n"))
                .collect()
        };
        for version in table_versions(&original).unwrap() {
            let driver = manifest
                .iter()
                .find(|d| d.arch == arch && d.version == version)
                .unwrap();
            drivers::verify_file(&driver.library(&cache), &driver.library_sha256).unwrap();
            let scan = scan::file(&driver.library(&cache), Some(arch), false).unwrap();
            let mut row = tables::row(&scan, driver.purpose == drivers::Purpose::Scanned).unwrap();
            // Older independent references predate the inline gate fields.
            // Compare their complete original schema; inline_threshold_corpus
            // separately pins the new captures against disassembled binaries.
            if !explicit_inline {
                row = row
                    .lines()
                    .filter(|line| !line.trim_start().starts_with(".inline_"))
                    .map(|line| format!("{line}\n"))
                    .collect();
            }
            // Compare every initializer, including bytes. The old provenance comments differ.
            let initializer = &row[row.find(" {\n  .version").unwrap()..];
            assert!(
                original.contains(initializer),
                "{arch} {version} differs from independent table:\n{row}"
            );
        }
    }
}

fn tar_fixture(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut archive = tar::Builder::new(Vec::new());
    for (name, bytes) in entries {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive.append_data(&mut header, *name, *bytes).unwrap();
    }
    archive.into_inner().unwrap()
}

fn fixture_driver(temp: &Path, extension: &str, package: &[u8], library: &[u8]) -> Driver {
    use sha2::{Digest, Sha256};
    let driver = Driver {
        purpose: Purpose::Table,
        arch: Arch::X86_64,
        version: "1.2".into(),
        url: format!("https://invalid.test/driver.{extension}"),
        package_sha256: format!("{:x}", Sha256::digest(package)),
        member: "libcuda.so.1.2".into(),
        library_sha256: format!("{:x}", Sha256::digest(library)),
    };
    let packages = driver.library(temp).parent().unwrap().join("pkg");
    fs::create_dir_all(&packages).unwrap();
    fs::write(packages.join(format!("driver.{extension}")), package).unwrap();
    driver
}

#[test]
fn cached_debian_extraction_and_failure_do_not_clobber_library() {
    let temp = tempfile::tempdir().unwrap();
    let tar = tar_fixture(&[("./libcuda.so.1.2", b"verified fixture")]);
    let mut package = ar::Builder::new(Vec::new());
    package
        .append(
            &ar::Header::new(b"data.tar".to_vec(), tar.len() as u64),
            tar.as_slice(),
        )
        .unwrap();
    let bytes = package.into_inner().unwrap();
    let mut driver = fixture_driver(temp.path(), "deb", &bytes, b"verified fixture");
    let path = drivers::fetch(&driver, temp.path()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"verified fixture");
    driver.library_sha256 = "0".repeat(64);
    assert!(drivers::fetch(&driver, temp.path()).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"verified fixture");
    driver.member = "../libcuda.so.1.2".into();
    assert!(drivers::fetch(&driver, temp.path()).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"verified fixture");
}

#[test]
fn installer_extraction_uses_header_not_execution() {
    use std::io::Write;
    let temp = tempfile::tempdir().unwrap();
    let tar = tar_fixture(&[("libcuda.so.1.2", b"installer fixture")]);
    for compression in ["xz", "zstd"] {
        let payload = if compression == "xz" {
            let mut encoder = xz2::write::XzEncoder::new(Vec::new(), 1);
            encoder.write_all(&tar).unwrap();
            encoder.finish().unwrap()
        } else {
            zstd::stream::encode_all(tar.as_slice(), 1).unwrap()
        };
        let mut package = format!("skip=8\nskip_decompress=8\nsize_decompress=0\nversion_string=1.2\nTARGET_ARCH=\"x86_64\"\necho \"  Compression : {compression}\"\n# Do not execute\n").into_bytes();
        package.extend(payload);
        let mut driver = fixture_driver(temp.path(), "run", &package, b"installer fixture");
        let package_path = driver
            .library(temp.path())
            .parent()
            .unwrap()
            .join("pkg/driver.run");
        let mut output = tempfile::tempfile().unwrap();
        drivers::extract(&package_path, &driver, &mut output).unwrap();
        driver.arch = Arch::Aarch64;
        assert!(drivers::extract(&package_path, &driver, &mut output).is_err());
        driver.arch = Arch::X86_64;
        driver.version = "9.9".into();
        assert!(drivers::extract(&package_path, &driver, &mut output).is_err());
    }
}

#[test]
fn duplicate_and_missing_members_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    for entries in [
        vec![
            ("libcuda.so.1.2", b"first".as_slice()),
            ("libcuda.so.1.2", b"second".as_slice()),
        ],
        vec![("other", b"other".as_slice())],
    ] {
        let tar = tar_fixture(&entries);
        let mut package = ar::Builder::new(Vec::new());
        package
            .append(
                &ar::Header::new(b"data.tar".to_vec(), tar.len() as u64),
                tar.as_slice(),
            )
            .unwrap();
        let driver = fixture_driver(
            temp.path(),
            "deb",
            &package.into_inner().unwrap(),
            b"not selected",
        );
        let path = driver.library(temp.path());
        fs::write(&path, "original").unwrap();
        assert!(drivers::fetch(&driver, temp.path()).is_err());
        assert_eq!(fs::read_to_string(path).unwrap(), "original");
    }
}

fn elf_fixture(machine: u16) -> Vec<u8> {
    fn put(bytes: &mut [u8], at: usize, value: u64, width: usize) {
        bytes[at..at + width].copy_from_slice(&value.to_le_bytes()[..width]);
    }
    let mut bytes = vec![0; 1024];
    bytes[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
    for (at, value, width) in [
        (16, 3, 2),
        (18, u64::from(machine), 2),
        (20, 1, 4),
        (32, 64, 8),
        (40, 768, 8),
        (52, 64, 2),
        (54, 56, 2),
        (56, 1, 2),
        (58, 64, 2),
        (60, 3, 2),
        (62, 1, 2),
    ] {
        put(&mut bytes, at, value, width);
    }
    for (at, value, width) in [
        (64, 1, 4),
        (68, 5, 4),
        (96, 1024, 8),
        (104, 1024, 8),
        (112, 4096, 8),
    ] {
        put(&mut bytes, at, value, width);
    }
    let strings = b"\0.shstrtab\0.eh_frame\0";
    bytes[128..128 + strings.len()].copy_from_slice(strings);
    let banner = b"Version 1.2. Copyright (C) fixture";
    bytes[180..180 + banner.len()].copy_from_slice(banner);
    let mut unwind = 16u32.to_le_bytes().to_vec();
    unwind.extend([0, 0, 0, 0, 1, b'z', b'R', 0, 1, 0x78, 16, 1, 0x1b, 0, 0, 0]);
    for (start, size) in [(0x1100u64, 0x20u32), (0x1000, 0x40), (0x1000, 0x10)] {
        unwind.extend(16u32.to_le_bytes());
        unwind.extend((unwind.len() as u32).to_le_bytes());
        let displacement = (start as i64 - (0x5000 + unwind.len()) as i64) as i32;
        unwind.extend(displacement.to_le_bytes());
        unwind.extend(size.to_le_bytes());
        unwind.extend([0; 4]);
    }
    unwind.extend([0; 4]);
    bytes[256..256 + unwind.len()].copy_from_slice(&unwind);
    for (index, name, kind, address, offset, size) in [
        (1, 1, 3, 0, 128, strings.len()),
        (2, 11, 1, 0x5000, 256, unwind.len()),
    ] {
        let s = 768 + 64 * index;
        for (at, value, width) in [
            (s, name, 4),
            (s + 4, kind, 4),
            (s + 16, address, 8),
            (s + 24, offset, 8),
            (s + 32, size as u64, 8),
            (s + 48, 1, 8),
        ] {
            put(&mut bytes, at, value, width);
        }
    }
    bytes
}

#[test]
fn elf_unwind_boundaries_match_original_independent_fixture() {
    for (machine, arch) in [(62, Arch::X86_64), (183, Arch::Aarch64)] {
        let image = Image::parse(elf_fixture(machine)).unwrap();
        assert_eq!(image.arch, arch);
        assert_eq!(
            image
                .functions
                .iter()
                .map(|f| (f.start, f.size))
                .collect::<Vec<_>>(),
            [(0x1000, 0x40), (0x1100, 0x20)]
        );
    }
    assert!(Image::parse(elf_fixture(40)).is_err()); // ELF64 marked as 32-bit ARM
    let mut truncated = elf_fixture(62);
    truncated.truncate(850);
    assert!(Image::parse(truncated).is_err());
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("libcuda.so.1.2");
    fs::write(&file, elf_fixture(62)).unwrap();
    assert!(scan::file(&file, Some(Arch::Aarch64), false)
        .err()
        .unwrap()
        .to_string()
        .contains("architecture"));
}

#[test]
fn unknown_import_is_not_assumed_to_be_memcpy() {
    for symbol in [Some("other_function"), None] {
        let mut image = synthetic(Arch::Aarch64, 0);
        image.got_symbols.clear();
        if let Some(symbol) = symbol {
            image.got_symbols.insert(0x30010, symbol.into());
        }
        let result = scan::scan(&image).unwrap();
        assert_eq!(result.engine.status, Status::Found);
        assert_eq!(result.row_copy.status, Status::NotFound);
        assert!(!result.complete());
    }
}

#[test]
fn manifest_rejects_duplicate_and_untrusted_fields() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("drivers.manifest");
    let row = format!(
        "x86_64 1.2 https://invalid.test/driver.deb {} libcuda.so.1.2 {} table\n",
        "0".repeat(64),
        "1".repeat(64)
    );
    fs::write(&path, &row).unwrap();
    assert_eq!(drivers::manifest(&path).unwrap().len(), 1);
    for invalid in [
        row.repeat(2),
        row.replace("https://", "http://"),
        row.replace(" libcuda.so.1.2 ", " ../libcuda.so.1.2 "),
        row.replace("x86_64", "unknown"),
        row.replace(" table", " unknown"),
        row.replace(" table", ""),
        row.replace("1.2 https", "../1.2 https"),
    ] {
        fs::write(&path, invalid).unwrap();
        assert!(drivers::manifest(&path).is_err());
    }
}

#[test]
fn version_filters_cannot_rewrite_tables() {
    for command in ["generate", "verify"] {
        assert!(Cli::try_parse_from([
            "cmt-sigscan",
            command,
            "--arch",
            "x86_64",
            "--versions",
            "1.2"
        ])
        .is_err());
        assert!(Cli::try_parse_from(["cmt-sigscan", command, "--arch", "x86_64"]).is_ok());
    }
    for command in ["list", "fetch"] {
        assert!(Cli::try_parse_from(["cmt-sigscan", command, "--versions", "1.2"]).is_err());
        assert!(Cli::try_parse_from([
            "cmt-sigscan",
            command,
            "--arch",
            "x86_64",
            "--versions",
            "1.2,2.3"
        ])
        .is_ok());
    }
    assert!(Cli::try_parse_from(["cmt-sigscan", "scan"]).is_err());
}

#[test]
fn filename_must_agree_with_banner() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("libcuda.so.9.9");
    fs::write(&file, elf_fixture(62)).unwrap();
    assert!(scan::file(&file, None, false)
        .err()
        .unwrap()
        .to_string()
        .contains("filename version"));
}

fn scanner_elf_fixture() -> Vec<u8> {
    fn put(bytes: &mut [u8], at: usize, value: u64) {
        bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }
    let image = synthetic(Arch::X86_64, 0);
    let mut bytes = elf_fixture(62);
    bytes.resize(0x3000, 0);
    bytes[0x1000..0x1000 + image.bytes.len()].copy_from_slice(&image.bytes);
    put(&mut bytes, 80, 0xf000);
    put(&mut bytes, 96, 0x3000);
    put(&mut bytes, 104, 0x3000);
    let section = 0xf100;
    let mut unwind = 16u32.to_le_bytes().to_vec();
    unwind.extend([0, 0, 0, 0, 1, b'z', b'R', 0, 1, 0x78, 16, 1, 0x1b, 0, 0, 0]);
    for function in image.functions {
        unwind.extend(16u32.to_le_bytes());
        unwind.extend((unwind.len() as u32).to_le_bytes());
        let displacement = (function.start as i64 - (section + unwind.len() as u64) as i64) as i32;
        unwind.extend(displacement.to_le_bytes());
        unwind.extend((function.size as u32).to_le_bytes());
        unwind.extend([0; 4]);
    }
    unwind.extend([0; 4]);
    bytes[256..256 + unwind.len()].copy_from_slice(&unwind);
    put(&mut bytes, 896 + 16, section);
    put(&mut bytes, 896 + 32, unwind.len() as u64);
    bytes
}

#[test]
fn first_valid_table_is_not_written_when_second_architecture_fails() {
    use sha2::{Digest, Sha256};
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("src/gen");
    fs::create_dir_all(&dir).unwrap();
    let original = "/* generated by fixture */\n.version = \"1.2\"\n";
    for arch in [Arch::X86_64, Arch::Aarch64] {
        fs::write(dir.join(arch.table_name()), original).unwrap();
    }
    let bytes = scanner_elf_fixture();
    let driver = Driver {
        purpose: Purpose::Table,
        arch: Arch::X86_64,
        version: "1.2".into(),
        url: "https://invalid.test/driver.deb".into(),
        package_sha256: "0".repeat(64),
        member: "libcuda.so.1.2".into(),
        library_sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    let library = driver.library(temp.path());
    fs::create_dir_all(library.parent().unwrap()).unwrap();
    fs::write(&library, bytes).unwrap();
    let row = scan::file(&library, Some(Arch::X86_64), false).unwrap();
    assert_eq!(
        (row.engine.offset, row.row_copy.offset, row.staging.offset),
        (0x1000, 0x1400, 0x1600)
    );
    let manifest = vec![
        driver.clone(),
        Driver {
            arch: Arch::Aarch64,
            ..driver
        },
    ];
    assert!(tables::prepare(temp.path(), temp.path(), &manifest, None, false).is_err());
    for arch in [Arch::X86_64, Arch::Aarch64] {
        assert_eq!(
            fs::read_to_string(dir.join(arch.table_name())).unwrap(),
            original
        );
    }
    let updates = tables::prepare(
        temp.path(),
        temp.path(),
        &manifest,
        Some(Arch::X86_64),
        false,
    )
    .unwrap();
    tables::publish(&updates).unwrap();
    assert!(fs::read_to_string(dir.join(Arch::X86_64.table_name()))
        .unwrap()
        .contains(&tables::row(&row, false).unwrap()));
    assert_eq!(
        fs::read_to_string(dir.join(Arch::Aarch64.table_name())).unwrap(),
        original
    );
}

// Addresses/captures recorded by a separate Python byte search and Capstone
// disassembly of the SHA-256-pinned corpus, not read from the generated tables.
#[test]
#[ignore = "reads all ten cached x86-64 driver libraries"]
fn inline_threshold_corpus() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cache = std::env::var_os("CMT_SIGSCAN_DRIVERS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("tmp/drivers"));
    let manifest = drivers::manifest(&repo.join("tools/sigscan/drivers.manifest")).unwrap();
    for (version, address, width, arg, slot) in [
        ("570.124.06", 0xd4c4ca, 0x190, 0x16, 0x45a0),
        ("570.211.01", 0xd4cb8a, 0x190, 0x16, 0x45a0),
        ("575.64.05", 0xf4236a, 0x190, 0x16, 0x4aa0),
        ("580.196.07", 0xf999ea, 0x190, 0x17, 0x64c0),
        ("590.48.01", 0x10b7e35, 0x1a0, 0x17, 0x6510),
        ("595.112.07", 0x11742b5, 0x1a0, 0x17, 0x85a0),
        ("595.99.02", 0x11740b5, 0x1a0, 0x17, 0x85a0),
        ("610.57.04", 0x131306d, 0x1b0, 0x18, 0x88c8),
        ("615.71.09", 0x132d63d, 0x1b0, 0x19, 0x96c0),
        ("615.78.08", 0x132f5fd, 0x1b0, 0x19, 0x96c0),
    ] {
        let driver = manifest
            .iter()
            .find(|d| d.arch == Arch::X86_64 && d.version == version)
            .unwrap();
        let path = driver.library(&cache);
        drivers::verify_file(&path, &driver.library_sha256).unwrap();
        let mut image = Image::load(&path).unwrap();
        let result = scan::scan(&image).unwrap();
        assert!(result.complete(), "{}", result.explain());
        let gate = result.inline.gate().unwrap();
        assert_eq!(gate.address(), address, "{version}");
        assert_eq!(gate.pinned_address(), address + 13, "{version}");
        assert_eq!(
            (
                gate.paged_limit(),
                gate.size_disp(),
                gate.call_arg(),
                gate.call_disp()
            ),
            (65536, width, arg, slot),
            "{version}"
        );
        assert_eq!(*gate.original(), image.at(address).unwrap()[..33]);
        // A real driver with one mismatched size field must not emit this site.
        let segment = image
            .segments
            .iter()
            .find(|s| address >= s.vaddr && address - s.vaddr < s.file_size)
            .unwrap();
        let offset = (segment.file_offset + address - segment.vaddr) as usize;
        image.bytes[offset + 27] ^= 8;
        let bad = inline::scan(&image, &result.descriptor).unwrap();
        assert_eq!(
            bad.status(),
            Status::NotFound,
            "{version}: mutated size field accepted"
        );
        assert!(bad.gate().is_none());
        eprintln!("{version}: gate {address:#x}, captures and corrupted-field rejection verified");
    }
}

#[test]
fn manifest_owns_table_membership_and_survey_inputs_stay_disabled() {
    use sha2::{Digest, Sha256};
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("src/gen");
    fs::create_dir_all(&dir).unwrap();
    let table = dir.join(Arch::X86_64.table_name());
    // A stale output must not control which inputs generation scans. The
    // handwritten preamble survives, but this old version is not configuration.
    let original = "// hand-written preamble\n/* generated by fixture */\n.version = \"9.9\"\n";
    fs::write(&table, original).unwrap();
    let mut manifest = Vec::new();
    for version in ["1.2", "1.3"] {
        let mut bytes = scanner_elf_fixture();
        let at = bytes
            .windows(12)
            .position(|b| b == b"Version 1.2.")
            .unwrap();
        bytes[at + 8..at + 11].copy_from_slice(version.as_bytes());
        let driver = Driver {
            arch: Arch::X86_64,
            purpose: Purpose::Table,
            version: version.into(),
            url: "https://invalid.test/driver.deb".into(),
            package_sha256: "0".repeat(64),
            member: format!("libcuda.so.{version}"),
            library_sha256: format!("{:x}", Sha256::digest(&bytes)),
        };
        let library = driver.library(temp.path());
        fs::create_dir_all(library.parent().unwrap()).unwrap();
        fs::write(library, bytes).unwrap();
        manifest.push(driver);
    }
    // No cached file exists for the survey row. Accidentally including it must
    // fail generation, rather than silently producing an unreviewed patch row.
    manifest.push(Driver {
        purpose: Purpose::Survey,
        version: "1.4".into(),
        ..manifest[0].clone()
    });
    let updates = tables::prepare(
        temp.path(),
        temp.path(),
        &manifest,
        Some(Arch::X86_64),
        false,
    )
    .unwrap();
    assert_eq!(
        table_versions(&updates[0].generated).unwrap(),
        ["1.2", "1.3"]
    );
    assert!(updates[0]
        .generated
        .starts_with("// hand-written preamble\n"));
    assert_eq!(fs::read_to_string(&table).unwrap(), original); // preparation does not publish
    tables::publish(&updates).unwrap();
    assert_eq!(fs::read_to_string(&table).unwrap(), updates[0].generated);
    // Fetch/list still allow inspection of a survey entry. Disabling generation
    // must not make a research input impossible to retrieve by version.
    let selected = drivers::select(&manifest, Some(Arch::X86_64), &["1.4".into()]).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].purpose, Purpose::Survey);
}

#[test]
fn empty_manifest_selection_cannot_erase_a_table() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("src/gen");
    fs::create_dir_all(&dir).unwrap();
    let table = dir.join(Arch::X86_64.table_name());
    let original = "/* generated by fixture */\n.version = \"1.2\"\n";
    fs::write(&table, original).unwrap();
    assert!(tables::prepare(temp.path(), temp.path(), &[], Some(Arch::X86_64), false).is_err());
    assert_eq!(fs::read_to_string(table).unwrap(), original);
}

#[test]
fn manifest_migration_preserves_checked_in_table_membership() {
    // This is a cheap inventory check, not a claim about real-driver matching.
    // The separate ignored corpus test checks every field against actual ELF.
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest = drivers::manifest(&repo.join("tools/sigscan/drivers.manifest")).unwrap();
    for arch in [Arch::X86_64, Arch::Aarch64] {
        let expected = table_versions(
            &fs::read_to_string(repo.join("src/gen").join(arch.table_name())).unwrap(),
        )
        .unwrap();
        let selected: Vec<_> = manifest
            .iter()
            .filter(|d| d.arch == arch && matches!(d.purpose, Purpose::Table | Purpose::Scanned))
            .map(|d| d.version.clone())
            .collect();
        assert_eq!(
            selected, expected,
            "{arch}: review manifest membership and regenerate the table"
        );
    }
}
