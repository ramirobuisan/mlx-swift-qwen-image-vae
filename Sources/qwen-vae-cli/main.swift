// qwen-vae <qwen_image_vae.safetensors> <input image> <output.png>
// Encodes the image to the 16-channel latent and decodes it back; prints timings and PSNR.
import Foundation
import MLX
import QwenImageVAE

let args = Array(CommandLine.arguments.dropFirst())
guard args.count == 3 else {
    print("usage: qwen-vae <qwen_image_vae.safetensors> <input image> <output.png>")
    exit(2)
}
do {
    let vae = try QwenImageVAE(url: URL(fileURLWithPath: args[0]))
    guard let img = loadRGB(args[1]) else { throw NSError(domain: "qwen-vae", code: 1, userInfo: [NSLocalizedDescriptionKey: "can't read \(args[1])"]) }
    // the VAE works on multiples of 8: crop
    let (w, h) = (img.width / 8 * 8, img.height / 8 * 8)
    let full = MLXArray(img.pixels, [img.height, img.width, 3])
    let x = (full[0 ..< h, 0 ..< w].asType(.float32) / 127.5 - 1).expandedDimensions(axis: 0)
    for run in 1 ... 2 {   // the first run includes Metal kernel compilation
        let t0 = Date()
        let z = vae.encode(x)
        eval(z)
        let t1 = Date()
        let y = vae.decode(z)
        eval(y)
        let t2 = Date()
        let mse = ((y.asType(.float32) - x) * 0.5).square().mean().item(Float.self)
        print(String(format: "run %d, %dx%d: encode %.2fs, decode %.2fs, latent %@, round-trip PSNR %.2f dB", run, w, h,
                     t1.timeIntervalSince(t0), t2.timeIntervalSince(t1), z.shape.description, 10 * log10(1 / mse)))
        if run == 2 {
            let out = clip((y[0].asType(.float32) + 1) * 127.5, min: 0, max: 255).round().asType(.uint8)
            try savePNG(out.asArray(UInt8.self), width: w, height: h, to: args[2])
        }
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
    exit(1)
}
