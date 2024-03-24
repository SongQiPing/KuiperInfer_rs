use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
// 部分代码参考：
// https://github.com/makepad/makepad/blob/c74cb133fb1e8e5a03d287374c5cdf20230ec53a/libs/miniz/src/zip_file.rs#L173
//
pub const COMPRESS_METHOD_UNCOMPRESSED: u16 = 0;
pub const COMPRESS_METHOD_DEFLATED: u16 = 8;

pub const LOCAL_FILE_HEADER_SIGNATURE: u32 = 0x04034b50;
pub const LOCAL_FILE_HEADER_SIZE: usize = 30;
#[derive(Clone, Debug)]
pub struct LocalFileHeader {
    pub signature: u32,
    pub version_needed_to_extract: u16,
    pub general_purpose_bit_flag: u16,
    pub compression_method: u16,
    pub last_mod_file_time: u16,
    pub last_mod_file_date: u16,
    pub crc32: u32,
    pub compressed_size: u32,
    pub uncompressed_size: u32,
    pub file_name_length: u16,
    pub extra_field_length: u16,
    pub file_name: String,
}
impl LocalFileHeader {
    pub fn from_stream(zip_data: &mut (impl Seek + Read)) -> Result<Self, ZipError> {
        let signature = read_u32(zip_data)?;
        if signature != LOCAL_FILE_HEADER_SIGNATURE {
            return Err(ZipError::LocalFileHeaderInvalid);
        }
        let version_needed_to_extract = read_u16(zip_data)?;
        let general_purpose_bit_flag = read_u16(zip_data)?;
        let compression_method = read_u16(zip_data)?;
        let last_mod_file_time = read_u16(zip_data)?;
        let last_mod_file_date = read_u16(zip_data)?;
        let crc32 = read_u32(zip_data)?;
        let compressed_size = read_u32(zip_data)?;
        let uncompressed_size = read_u32(zip_data)?;
        let file_name_length = read_u16(zip_data)?;
        let extra_field_length = read_u16(zip_data)?;

        let file_name = read_string(zip_data, file_name_length as usize)?.clone();
        zip_data
            .seek(SeekFrom::Current(extra_field_length as i64))
            .map_err(|_| ZipError::CantSeekSkip)?;

        Ok(Self {
            signature,
            version_needed_to_extract,
            general_purpose_bit_flag,
            compression_method,
            last_mod_file_time,
            last_mod_file_date,
            crc32,
            compressed_size,
            uncompressed_size,
            file_name_length,
            extra_field_length,
            file_name,
        })
    }
}

pub const CENTRAL_DIR_FILE_HEADER_SIGNATURE: u32 = 0x02014b50;
pub const CENTRAL_DIR_FILE_HEADER_SIZE: usize = 46;
#[derive(Clone, Debug)]
pub struct CentralDirectoryFileHeader {
    pub signature: u32,
    pub version_made_by: u16,
    pub version_needed_to_extract: u16,
    pub general_purpose_bit_flag: u16,
    pub compression_method: u16,
    pub last_mod_file_time: u16,
    pub last_mod_file_date: u16,
    pub crc32: u32,
    pub compressed_size: u32,
    pub uncompressed_size: u32,
    pub file_name_length: u16,
    pub extra_field_length: u16,
    pub file_comment_length: u16,
    pub disk_number_start: u16,
    pub internal_file_attributes: u16,
    pub external_file_attributes: u32,
    pub relative_offset_of_local_header: u32,

    pub file_name: String,
    pub file_comment: String,
}

impl CentralDirectoryFileHeader {
    pub fn from_stream(zip_data: &mut (impl Seek + Read)) -> Result<Self, ZipError> {
        let signature = read_u32(zip_data)?;

        if signature != CENTRAL_DIR_FILE_HEADER_SIGNATURE {
            return Err(ZipError::CentralDirectoryFileHeaderInvalid);
        }
        let version_made_by = read_u16(zip_data)?;
        let version_needed_to_extract = read_u16(zip_data)?;
        let general_purpose_bit_flag = read_u16(zip_data)?;
        let compression_method = read_u16(zip_data)?;
        let last_mod_file_time = read_u16(zip_data)?;
        let last_mod_file_date = read_u16(zip_data)?;
        let crc32 = read_u32(zip_data)?;
        let compressed_size = read_u32(zip_data)?;
        let uncompressed_size = read_u32(zip_data)?;
        let file_name_length = read_u16(zip_data)?;
        let extra_field_length = read_u16(zip_data)?;
        let file_comment_length = read_u16(zip_data)?;
        let disk_number_start = read_u16(zip_data)?;
        let internal_file_attributes = read_u16(zip_data)?;
        let external_file_attributes = read_u32(zip_data)?;
        let relative_offset_of_local_header = read_u32(zip_data)?;
        let file_name = read_string(zip_data, file_name_length as usize)?;
        zip_data
            .seek(SeekFrom::Current(extra_field_length as i64))
            .map_err(|_| ZipError::CantSeekSkip)?;
        let file_comment = read_string(zip_data, file_comment_length as usize)?;

        Ok(Self {
            signature,
            version_made_by,
            version_needed_to_extract,
            general_purpose_bit_flag,
            compression_method,
            last_mod_file_time,
            last_mod_file_date,
            crc32,
            compressed_size,
            uncompressed_size,
            file_name_length,
            extra_field_length,
            file_comment_length,
            disk_number_start,
            internal_file_attributes,
            external_file_attributes,
            relative_offset_of_local_header,
            file_name,
            file_comment,
        })
    }
}

pub const END_OF_CENTRAL_DIRECTORY_SIGNATURE: u32 = 0x06054b50;
pub const END_OF_CENTRAL_DIRECTORY_SIZE: usize = 22;
#[derive(Clone, Debug)]
pub struct EndOfCentralDirectory {
    pub signature: u32,
    pub number_of_disk: u16,
    pub number_of_start_central_directory_disk: u16,
    pub total_entries_this_disk: u16,
    pub total_entries_all_disk: u16,
    pub size_of_the_central_directory: u32,
    pub central_directory_offset: u32,
    pub zip_file_comment_length: u16,
}

impl EndOfCentralDirectory {
    pub fn from_stream(zip_data: &mut impl Read) -> Result<Self, ZipError> {
        let signature = read_u32(zip_data)?;
        if signature != END_OF_CENTRAL_DIRECTORY_SIGNATURE {
            return Err(ZipError::EndOfCentralDirectoryInvalid);
        }
        Ok(Self {
            signature,
            number_of_disk: read_u16(zip_data)?,
            number_of_start_central_directory_disk: read_u16(zip_data)?,
            total_entries_this_disk: read_u16(zip_data)?,
            total_entries_all_disk: read_u16(zip_data)?,
            size_of_the_central_directory: read_u32(zip_data)?,
            central_directory_offset: read_u32(zip_data)?,
            zip_file_comment_length: read_u16(zip_data)?,
        })
    }
}
fn read_u16(zip_data: &mut impl Read) -> Result<u16, ZipError> {
    let mut bytes = [0u8; 2];
    if let Ok(size) = zip_data.read(&mut bytes) {
        if size != 2 {
            return Err(ZipError::DataReadError);
        }
        return Ok(u16::from_le_bytes(bytes));
    }
    Err(ZipError::DataReadError)
}

fn read_u32(zip_data: &mut impl Read) -> Result<u32, ZipError> {
    let mut bytes = [0u8; 4];
    if let Ok(size) = zip_data.read(&mut bytes) {
        if size != 4 {
            return Err(ZipError::DataReadError);
        }
        return Ok(u32::from_le_bytes(bytes));
    }
    Err(ZipError::DataReadError)
}

fn read_string(zip_data: &mut impl Read, len: usize) -> Result<String, ZipError> {
    let mut data = Vec::new();
    data.resize(len, 0u8);
    if let Ok(size) = zip_data.read(&mut data) {
        if size != data.len() {
            return Err(ZipError::DataReadError);
        }
        if let Ok(s) = String::from_utf8(data) {
            return Ok(s);
        }
        return Err(ZipError::ReadStringError);
    }
    Err(ZipError::DataReadError)
}

fn read_binary(zip_data: &mut impl Read, len: usize) -> Result<Vec<u8>, ZipError> {
    let mut data = Vec::new();
    data.resize(len, 0u8);
    if let Ok(size) = zip_data.read(&mut data) {
        if size != data.len() {
            return Err(ZipError::DataReadError);
        }
        return Ok(data);
    }
    Err(ZipError::DataReadError)
}
pub struct ZipCentralDirectory {
    pub eocd: EndOfCentralDirectory,
    pub file_headers: Vec<CentralDirectoryFileHeader>,
}

// impl CentralDirectoryFileHeader{
//     // lets read and unzip specific files.
//     pub fn extract(&self, zip_data: &mut (impl Seek+Read))->Result<Vec<u8>, ZipError>{
//         zip_data.seek(SeekFrom::Start(self.relative_offset_of_local_header as u64)).map_err(|_| ZipError::CantSeekToFileHeader)?;
//         let header = LocalFileHeader::from_stream(zip_data)?;
//         if header.compression_method == COMPRESS_METHOD_UNCOMPRESSED{
//             let decompressed = read_binary(zip_data, self.uncompressed_size as usize)?;
//             return Ok(decompressed)
//         }
//         else if header.compression_method == COMPRESS_METHOD_DEFLATED{
//             let compressed = read_binary(zip_data, self.compressed_size as usize)?;
//             if let Ok(decompressed) = decompress_to_vec(&compressed){
//                 return Ok(decompressed);
//             }
//             else{
//                 return Err(ZipError::DecompressionError)
//             }
//         }
//         Err(ZipError::UnsupportedCompressionMethod)
//     }
// }

#[derive(Debug)]
pub enum ZipError {
    LocalFileHeaderInvalid,
    CentralDirectoryFileHeaderInvalid,
    EndOfCentralDirectoryInvalid,
    CantSeekToDirEnd,
    CantSeekToFileHeader,
    CantSeekSkip,
    ParseError,
    ReadStringError,
    CantSeekToDirStart,
    UnsupportedCompressionMethod,
    DecompressionError,
    DataReadError,
}
pub struct StoreZipReader {
    zip_reader: Option<BufReader<File>>,
    filemetas: HashMap<String, CentralDirectoryFileHeader>,
}

fn read_file(path: &str) -> Option<File> {
    let file = File::open(path).unwrap();
    Some(file)
}

impl StoreZipReader {
    pub fn new() -> Self {
        StoreZipReader {
            zip_reader: None,
            filemetas: HashMap::new(),
        }
    }

    pub fn from_file(path: &str) -> Self {
        let mut zip_reader: BufReader<File> = BufReader::new(read_file(path).unwrap());

        zip_reader
            .seek(SeekFrom::End(-(END_OF_CENTRAL_DIRECTORY_SIZE as i64)))
            .map_err(|_| ZipError::CantSeekToDirEnd)
            .unwrap();
        let eocd = EndOfCentralDirectory::from_stream(&mut zip_reader).unwrap();
        
        zip_reader
            .seek(SeekFrom::Start(eocd.central_directory_offset as u64))
            .map_err(|_| ZipError::CantSeekToDirStart)
            .unwrap();

        let mut filemetas: HashMap<String, CentralDirectoryFileHeader> = HashMap::new();

        for _ in 0..eocd.total_entries_all_disk as usize {
            let file_header = CentralDirectoryFileHeader::from_stream(&mut zip_reader).unwrap();
            filemetas.insert(file_header.file_name.clone(), file_header);
        }

        StoreZipReader {
            zip_reader: Some(zip_reader),
            filemetas: filemetas,
        }
    }

    pub fn get_file_size(&self, name: &str) -> Option<usize> {
        self.filemetas
            .get(name)
            .map(|meta| meta.compressed_size as usize)
    }

    pub fn read_file(& mut self, name: &String) -> Vec<u8> {
        let file_head: &CentralDirectoryFileHeader = self.filemetas.get(name).unwrap();
        // let mut zip_reader = & self.zip_reader.unwrap();
        let mut zip_reader = self.zip_reader.as_mut().unwrap();
        
        file_head.extract(  & mut zip_reader).unwrap()

    }

    // pub fn close(&mut self){
    //     self.fp = None;
    // }
}
use flate2::read::ZlibDecoder;
pub fn decompress_to_vec(compressed_data: &[u8]) -> Vec<u8> {
    let mut decoder = ZlibDecoder::new(compressed_data);
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data).unwrap();

    decompressed_data
}
impl CentralDirectoryFileHeader {
    // lets read and unzip specific files.
    pub fn extract(&self, zip_data: &mut (impl Seek + Read)) -> Result<Vec<u8>, ZipError> {
        zip_data
            .seek(SeekFrom::Start(self.relative_offset_of_local_header as u64))
            .map_err(|_| ZipError::CantSeekToFileHeader)?;
        let header = LocalFileHeader::from_stream(zip_data)?;
        if header.compression_method == COMPRESS_METHOD_UNCOMPRESSED {
            let decompressed = read_binary(zip_data, self.uncompressed_size as usize)?;
            return Ok(decompressed);
        } else if header.compression_method == COMPRESS_METHOD_DEFLATED {
            let compressed = read_binary(zip_data, self.compressed_size as usize)?;
            let decompressed = decompress_to_vec(&compressed);
            return Ok(decompressed);
        }
        Err(ZipError::UnsupportedCompressionMethod)
    }
}
#[cfg(test)]
mod test_store_zip_read {
    use super::*;

    #[test]
    fn new_store_zip() {
        let _store_zip_reader = StoreZipReader::new();
    }
    #[test]
    fn open_file() {
        let file_path = "model_file/test_linear.pnnx.bin".to_string();
        let _store_zip_reader = StoreZipReader::from_file(&file_path);
        // store_zip_reader.open(&file_path)`e
    }
}
#[cfg(test)]
mod test_zip_rs {
    use std::io::{Read};

    fn read_u16(zip_data: &mut impl Read) -> u16 {
        let mut bytes = [0u8; 2];
        if let Ok(size) = zip_data.read(&mut bytes) {
            if size != 2 {
                panic!();
            }
            return u16::from_le_bytes(bytes);
        }
        // peanic!();
        0
    }

    #[test]
    fn test() {
        use std::fs;
        use std::io;

        let file_path = "model_file/test_linear.pnnx.bin".to_string();
        let zipfile = std::fs::File::open(file_path).unwrap();
        let mut archive = zip::ZipArchive::new(zipfile).unwrap();

        for i in 0..archive.len() {
            let mut file = archive.by_index(i).unwrap();
            let outpath = match file.enclosed_name() {
                Some(path) => path.to_owned(),
                None => continue,
            };
            {
                let comment = file.comment();
                if !comment.is_empty() {
                    println!("File {i} comment: {comment}");
                }
            }
            if (*file.name()).ends_with('/') {
                println!("File {} extracted to \"{}\"", i, outpath.display());
                fs::create_dir_all(&outpath).unwrap();
            } else {
                println!(
                    "File {} extracted to \"{}\" ({} bytes)",
                    i,
                    outpath.display(),
                    file.size()
                );
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        fs::create_dir_all(p).unwrap();
                    }
                }
                let mut outfile = fs::File::create(&outpath).unwrap();
                io::copy(&mut file, &mut outfile).unwrap();
            }
        }
    }
    #[test]
    fn test_read_local_file_header() {
        use std::io::{BufReader};
        let file_path = "model_file/test_linear.pnnx.bin".to_string();
        let zipfile = std::fs::File::open(file_path).unwrap();
        // let x = zipfile.bytes();
        let mut reader = BufReader::new(zipfile);
        let x = read_u16(&mut reader);
        println!("{:?}", x);
    }
}
