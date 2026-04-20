# Why a U-Net for Resolv

## The Problem

Resolv produces paired images: a noisy low-sample-count Cycles render and a clean high-sample-count + OIDN reference. The denoising model must take a noisy image and output a clean one — a **pixel-to-pixel regression** task where the output has the same spatial dimensions as the input.

## Why U-Net Is Ideal

### 1. It's built for dense pixel-wise prediction

U-Net was designed for tasks where every pixel in the output matters (originally biomedical segmentation). Denoising is exactly this kind of task — the network must produce a value for every single pixel, not just a class label or a bounding box.

### 2. Multi-scale context with fine-grained detail

Monte Carlo noise in path-traced renders appears at multiple scales. At very low spp (2–4 samples), you get large colour splotches and fireflies; at moderate spp (16–64), noise is finer-grained. U-Net's encoder captures context at progressively larger receptive fields (understanding "this region is a shadow" or "this is a bright highlight"), while the decoder reconstructs fine spatial detail. This multi-scale reasoning is essential for distinguishing noise from actual scene features.

### 3. Skip connections preserve structure

The defining feature of U-Net is its skip connections between encoder and decoder at each resolution level. For denoising, this is critical: the overall structure of the image (edges, object boundaries, texture patterns) is already present in the noisy input — it just needs to be cleaned up. Skip connections let the decoder access the original high-resolution features directly, so the network doesn't have to "re-learn" spatial layout from a compressed bottleneck. The result is sharper edges and better preservation of fine detail.

### 4. The input and output are nearly identical

In denoising, the noisy image is already close to the target — unlike tasks such as generation or style transfer. U-Net effectively learns a **residual correction** on top of the input structure. The skip connections make this natural: the network can pass most information through unchanged and focus its learned capacity on removing the noise.

### 5. Efficient for high-resolution images

Our renders range from 640x480 to 1920x1080. U-Net is memory-efficient compared to architectures that maintain full resolution throughout (like plain convolutional stacks), because the encoder progressively downsamples. Computation-heavy processing happens at reduced resolution in the bottleneck, while skip connections recover spatial detail cheaply.

## U-Net Structure

```
Input (noisy image)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│                     ENCODER                          │
│                                                      │
│  Level 1: Conv → Conv → ReLU          ──────┐       │
│     │                                        │ skip  │
│     ▼ MaxPool (downsample ½)                 │       │
│                                              │       │
│  Level 2: Conv → Conv → ReLU          ──────┐│      │
│     │                                        ││skip  │
│     ▼ MaxPool (downsample ½)                 ││      │
│                                              ││      │
│  Level 3: Conv → Conv → ReLU          ──────┐││     │
│     │                                        │││skip │
│     ▼ MaxPool (downsample ½)                 │││     │
│                                              │││     │
│  Level 4: Conv → Conv → ReLU          ──────┐│││    │
│     │                                        ││││    │
│     ▼ MaxPool (downsample ½)                 ││││    │
└──────────────────────────────────────────────┼┼┼┼────┘
                                               ││││
┌──────────────────────────────────────────────┼┼┼┼────┐
│                  BOTTLENECK                   ││││    │
│                                              ││││    │
│  Conv → Conv → ReLU  (smallest resolution,   ││││    │
│                        largest channel count) ││││    │
└──────────────────────────────────────────────┼┼┼┼────┘
                                               ││││
┌──────────────────────────────────────────────┼┼┼┼────┐
│                     DECODER                   ││││    │
│                                              ││││    │
│  UpConv (upsample ×2) + Concat ◄─────────────┘│││   │
│  Level 4: Conv → Conv → ReLU                   │││   │
│     │                                          │││   │
│  UpConv (upsample ×2) + Concat ◄───────────────┘││  │
│  Level 3: Conv → Conv → ReLU                    ││   │
│     │                                           ││   │
│  UpConv (upsample ×2) + Concat ◄────────────────┘│  │
│  Level 2: Conv → Conv → ReLU                      │  │
│     │                                             │  │
│  UpConv (upsample ×2) + Concat ◄─────────────────┘  │
│  Level 1: Conv → Conv → ReLU                         │
│     │                                                │
│     ▼                                                │
│  1×1 Conv (map channels → output channels)           │
└──────────────────────────────────────────────────────┘
  │
  ▼
Output (denoised image)
```

### Component Breakdown

- **Encoder (contracting path):** Each level applies two 3x3 convolutions (each followed by batch normalisation and ReLU), then a 2x2 max pool to halve spatial dimensions. Channel count doubles at each level (e.g. 64 → 128 → 256 → 512). This builds up increasingly abstract, large-receptive-field features.

- **Bottleneck:** The deepest level — smallest spatial resolution, most channels (e.g. 1024). This is where the network reasons about global context: overall illumination, large-scale colour patterns, scene-level structure.

- **Decoder (expansive path):** Each level upsamples by 2x (transposed convolution or bilinear interpolation + conv), concatenates the corresponding encoder features via the skip connection, then applies two 3x3 convolutions. Channel count halves at each level. The concatenation is the key — it combines the decoder's upsampled coarse prediction with the encoder's fine-grained spatial information.

- **Final layer:** A 1x1 convolution maps the last feature map to the desired output channels (3 for RGB).

### Why Each Part Matters for Denoising

| Component | Role in Denoising |
|-----------|-------------------|
| Encoder downsampling | Aggregates local pixel neighbourhoods to distinguish noise (random) from signal (structured) |
| Bottleneck | Global reasoning — understands that a dark region is a shadow, not noise to be brightened |
| Skip connections | Preserves edges, texture detail, and spatial layout that would be lost through the bottleneck |
| Decoder upsampling | Reconstructs full-resolution output guided by both global context and local detail |

## Summary

U-Net's encoder-decoder structure with skip connections is a natural fit for image denoising: it processes information at multiple scales (matching the multi-scale nature of Monte Carlo noise), preserves spatial structure through skip connections (critical since noisy and clean images share the same geometry), and is efficient enough for high-resolution renders. It is the standard backbone for learned denoising and the right starting architecture for Resolv.
