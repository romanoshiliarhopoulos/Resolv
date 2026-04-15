# Resolv
 
Resolv is a machine learning research project focused on image denoising, targeting real-world camera sensor noise.
 
## Goals
 
- Generate a large-scale synthetic paired dataset of clean and noisy images using photorealistic 3D rendering, with emphasis on optically complex scenes involving reflections, refractions, and caustics
- Develop and train deep learning models capable of recovering clean signal from noisy input across a range of noise conditions
- Support simulated camera sensor noise (shot noise, read noise) within a unified pipeline
- Evaluate model performance rigorously against established image quality benchmarks
- Produce models that generalize well to unseen scenes and noise levels, not just the training distribution
 
## Scope
 
Resolv covers the full pipeline from raw data generation through model training and evaluation. This includes scene generation and rendering, noise simulation, dataset management, model architecture, training infrastructure, and qualitative and quantitative evaluation.
 
The project targets a research-grade implementation suitable for experimentation and iteration, with clean abstractions between each stage of the pipeline to allow components to be swapped or improved independently.