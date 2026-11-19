# mlx-swift-qwen-image-vae

The **Qwen-Image VAE** (the Wan 2.1 video VAE, also used by Anima / WAI-ANIMA) for single images, in Swift with
[MLX](https://github.com/ml-explore/mlx-swift). It encodes RGB images to the 16-channel latent and decodes them
back, 100-200× faster than stable-diffusion.cpp's Metal backend on the same machine.

```swift
import MLX
import QwenImageVAE

let vae = try QwenImageVAE(url: URL(fileURLWithPath: "qwen_image_vae.safetensors"))
let latent = vae.encode(image)    // NHWC in -1...1 -> [N, H/8, W/8, 16] (raw mean, not normalised)
let image2 = vae.decode(latent)   // -> NHWC in -1...1
```

```bash
swift run -c release qwen-vae qwen_image_vae.safetensors input.png roundtrip.png
```

Weights: `qwen_image_vae.safetensors` from [Comfy-Org](https://huggingface.co/Comfy-Org) (original Wan 2.1
key names, `encoder.downsamples.N...`, 3D kernels), bf16 by default.

## Speed

Apple M5 Max, bf16:

| | MLX (this) | stable-diffusion.cpp (ggml Metal) |
|---|---|---|
| decode 1024×1344 | **0.27 s** | ~45 s |
| decode 1536×2016 | **0.68 s** | ~135 s |
| encode 1536×2016 | **0.59 s** | ~85 s |

- Encode → decode round trip: 44.5-45 dB PSNR.
- A full Anima txt2img where only the VAE differs: 53.9 dB against stable-diffusion.cpp's image.
- stable-diffusion.cpp's VAE tiling and `vae_conv_direct` options both made it slower.

## Why a 3D VAE runs as a 2D one

The Wan VAE is causal in time: every 3×3×3 convolution pads two frames in front, and the down/upsample
blocks have extra temporal convolutions that start working from the second frame. For a single image the
causal cache is empty (zeros), so

- each 3×3×3 kernel only ever multiplies the image with its **last temporal slice**: it is a 3×3 2D conv,
- the `time_conv` layers are skipped on the first frame,
- `RMS_norm` (`F.normalize(x, dim=channels) * sqrt(C) * gamma`) is exactly an RMS norm over channels.

What is left is an ordinary 2D VAE: the loader keeps `weight[:, :, -1]` of each 5D kernel and drops
`time_conv`. The middle attention block is single-head over all pixels and runs through MLX's fused attention.

## Using it from stable-diffusion.cpp

`patches/sdcpp-vae-hook.patch` (for stable-diffusion.cpp commit `532d2c1`) adds

```c
typedef bool (*sd_vae_hook_t)(int direction, const float* input, const int64_t* input_shape,
                              float* output, const int64_t* output_shape, void* data);
void sd_set_vae_hook(sd_ctx_t* ctx, sd_vae_hook_t hook, void* data);
```

With a hook set, every single-image VAE encode (`direction` 0: image NCHW in 0...1 → raw latent) and decode
(`direction` 1: raw latent → image in 0...1) of that context goes to your callback. That covers txt2img,
img2img, hi-res fix and the ADetailer inpainting. The model's latent mean/std normalisation stays in
stable-diffusion.cpp. Returning `false` falls back to the built-in VAE. `encode(nchw:...)` and `decode(nchw:...)`
in this package take exactly those buffers. [WAI Gen](https://github.com/ramirobuisan/wai-gen) uses it this way.

## License

MIT
