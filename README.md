# rotating_cube_ppm
A minimal C++ implementation​ that renders a rotating 3D cube​ and outputs each frame as a PPM image sequence. 
No external dependencies—pure standard library code (C++17 or later).
Ideal for learning software rendering, 3D transformation basics, or generating input for video/animation pipelines😄.
# Key Features:​
Renders a wireframe/color cube with smooth rotation.
Outputs frames as PPM (P6 binary format recommended)​ to a specified directory.
# Use Cases:​
Learning how 3D rendering works under the hood (no OpenGL/Vulkan!).
Generating PPM sequences for conversion to MP4/GIF (with FFmpeg).
Quick prototyping of software rasterization concepts.
# Quick Start:​
Clone & build (```g++ main.cpp -o2 -o main.exe```).
Run the executable.
Convert PPMs to video: ```ffmpeg -i output_%03d.ppm -r 40 out.mp4```
Tech Stack:​ Pure C++, no external libs.
Output:​ PPM image sequence → easily shareable/renderable.