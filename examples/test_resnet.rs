use image::imageops::resize;
use image::io::Reader as ImageReader;
use kuiper_infer::{data::SharedTensor, runtime::RuntimeGraph, Tensor};

use std::path::Path;
fn pre_process_image(image: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> SharedTensor<f32> {
    // 调整输入的大小
    let resized_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        resize(image, 224, 224, image::imageops::FilterType::Nearest);
    let mut image_tensor = Tensor::<f32>::new(&[3, 224, 224]);
    let (width, height) = resized_img.dimensions();
    for y in 0..height {
        for x in 0..width {
            let pixel = resized_img.get_pixel(x, y);
            let [r, g, b] = pixel.0;
            image_tensor.data[[0, y as usize, x as usize]] = r as f32;
            image_tensor.data[[1, y as usize, x as usize]] = g as f32;
            image_tensor.data[[2, y as usize, x as usize]] = b as f32;
        }
    }
    let (mean_r, mean_g, mean_b) = (0.485, 0.456, 0.406);
    let (var_r, var_g, var_b) = (0.229, 0.224, 0.225);
    for y in 0..height {
        for x in 0..width {
            let r = image_tensor.data[[0, y as usize, x as usize]];
            let g = image_tensor.data[[1, y as usize, x as usize]];
            let b = image_tensor.data[[2, y as usize, x as usize]];
            image_tensor.data[[0, y as usize, x as usize]] = (r - mean_r) / var_r;
            image_tensor.data[[1, y as usize, x as usize]] = (g - mean_g) / var_g;
            image_tensor.data[[2, y as usize, x as usize]] = (b - mean_b) / var_b;
        }
    }
    image_tensor.shared_self()
}
fn main() {
    let param_path = "model_file/resnet18_batch1.param".to_string();
    let bin_path = "model_file/resnet18_batch1.pnnx.bin".to_string();
    let mut runtime_grpah: RuntimeGraph = RuntimeGraph::new(param_path, bin_path);

    runtime_grpah.init();

    runtime_grpah.build("pnnx_input_0".to_string(), "pnnx_output_0".to_string());

    let img_path = "model_file/car.jpg";

    let img = ImageReader::open(&Path::new(img_path))
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();
    let tensor = pre_process_image(&img);
    let inputs = vec![tensor];
    let outputs = runtime_grpah.forward(&inputs).unwrap();
    println!("{:?}", outputs);
}
