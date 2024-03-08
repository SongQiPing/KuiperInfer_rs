use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
const LOCAL_FILE_HEADER_MAGIC: u32 = 0x04034b50;

#[repr(packed)]
#[derive(Debug)]
struct LocalFileHeader {
    pub(crate) version: u16,
    pub(crate) flag: u16,
    pub(crate) compression: u16,
    pub(crate) last_modify_time: u16,
    pub(crate) last_modify_date: u16,
    pub(crate) crc32: u32,
    pub(crate) compressed_size: u32,
    pub(crate) uncompressed_size: u32,
    pub(crate) file_name_length: u16,
    pub(crate) extra_field_length: u16,
}

impl LocalFileHeader {
    fn from_bytes(bytes: Vec<u8>) -> Result<Self, &'static str> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return Err("Insufficient bytes to create LocalFileHeader");
        }

        let header: LocalFileHeader = unsafe {
            const HEADER_BYTES_SIZE: usize = std::mem::size_of::<LocalFileHeader>();
            let bytes: &[u8; HEADER_BYTES_SIZE] = bytes[..HEADER_BYTES_SIZE]
                .as_ref()
                .try_into()
                .expect("Invalid byte slice size");
            std::mem::transmute_copy(bytes)
        };

        Ok(header)
    }
}

pub struct StoreZipReader {
    fp: Option<File>,
    filemetas: HashMap<String, StoreZipMeta>,
}

pub struct StoreZipMeta {
    offset: usize,
    size: usize,
}
fn read_file(path: &str) -> Option<File> {
    let file = File::open(path).unwrap();
    Some(file)
}

impl StoreZipReader {
    pub fn new() -> Self {
        StoreZipReader {
            fp: None,
            filemetas: HashMap::new(),
        }
    }
    fn from_file(path: &str) -> Self {
        let mut store_zip_reader = StoreZipReader::new();
        store_zip_reader.fp = read_file(path);
        println!("{:?}", &store_zip_reader.fp);

        let mut file = &store_zip_reader.fp.unwrap();
        let mut signature_buf: Vec<u8> = Vec::new();
        file.read_to_end(&mut signature_buf);
        // println!("{:?}", &signature_buf);
        for chunk in signature_buf.chunks_exact(4) {
            // 将每个块的字节转换为u32
            let signature = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            println!("Read 32-bit value: {}, {}", signature, 0x04034b50);
            match signature {
                LOCAL_FILE_HEADER_MAGIC => {}

                _ => {}
            }
            break;
        }
        StoreZipReader::new()
    }

    pub fn get_file_size(&self, name: &str) -> Option<usize> {
        self.filemetas.get(name).map(|meta| meta.size)
    }

    // pub fn read_file(&mut self, name: &str, data: &mut Vec<u8>) -> io::Result<usize> {
    //     let meta = self
    //         .filemetas
    //         .get(name)
    //         .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found"))?;

    //     let mut file = self.fp.as_ref().ok_or_else(|| {
    //         io::Error::new(io::ErrorKind::Other, "File handle not initialized")
    //     })?;

    //     file.seek(SeekFrom::Start(meta.offset as u64))?;
    //     let bytes_read = file.take(meta.size as u64).read_to_end(data)?;

    //     Ok(bytes_read)
    // }

    pub fn close(&mut self) -> io::Result<()> {
        self.fp = None;
        Ok(())
    }
}
#[cfg(test)]
mod test_local_file_header {
    use super::*;
    #[test]
    fn test_from_bytes() {
        let bytes: Vec<u8> = vec![
            0x01, 0x00, // version
            0x00, 0x00, // flag
            0x00, 0x00, // compression
            0x00, 0x00, // last_modify_time
            0x00, 0x00, // last_modify_date
            0x00, 0x00, 0x00, 0x00, // crc32
            0x00, 0x00, 0x00, 0x00, // compressed_size
            0x00, 0x00, 0x00, 0x00, // uncompressed_size
            0x0A, 0x00, // file_name_length
            0x00, 0x00, // extra_field_length
        ];
        match LocalFileHeader::from_bytes(bytes) {
            Ok(header) => println!("{:?}", header),
            Err(err) => println!("Error: {}", err),
        }
    }
    #[test]
    fn test_from_bytes_by_less() {
        let bytes: Vec<u8> = vec![
            0x01, 0x00, // version
            0x00, 0x00, // flag
            0x00, 0x00, // compression
            0x00, 0x00, // last_modify_time
            0x00, 0x00, // last_modify_date
            0x00, 0x00, 0x00, 0x00, // crc32
            0x00, 0x00, 0x00, 0x00, // compressed_size
            0x00, 0x00, 0x00, 0x00, // uncompressed_size
            0x0A, 0x00, // file_name_length
            0x00,  // extra_field_length
        ];
        match LocalFileHeader::from_bytes(bytes) {
            Ok(header) => {
                println!("{:?}", header);
                panic!("");

            },
            Err(err) => {
                println!("Error: {}", err);
                assert_eq!("Insufficient bytes to create LocalFileHeader", err);

            },
        }
    }
    #[test]
    fn test_transmute_copy() {
        use std::mem;
        #[repr(packed)]
        struct Foo {
            bar: u8,
        }

        let foo_array = [10u8];

        unsafe {
            // Copy the data from 'foo_array' and treat it as a 'Foo'
            let mut foo_struct: Foo = mem::transmute_copy(&foo_array);
            assert_eq!(foo_struct.bar, 10);

            // Modify the copied data
            foo_struct.bar = 20;
            assert_eq!(foo_struct.bar, 20);
        }

        // The contents of 'foo_array' should not have changed
        assert_eq!(foo_array, [10]);
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
        let mut store_zip_reader = StoreZipReader::from_file(&file_path);
        // store_zip_reader.open(&file_path)`e
    }
}
