# Running CPU Inference on the Board

To run inference on the board using the CPU demo, you will need:

1. **A .rgb input file**
   - The input image must be in raw RGB format (`.rgb`).
   - You can generate a `.rgb` file from a `.jpg` image using [ffmpeg](https://ffmpeg.org/), which can be installed via [Homebrew](https://brew.sh/) on macOS:
     ```sh
     brew install ffmpeg
     ffmpeg -i bell.jpg -vf scale=224:224,format=rgb24 -f rawvideo -y bell.rgb
     ```
   - This command converts `bell.jpg` to a 224x224 raw RGB file named `bell.rgb`.

2. **The .tflite model**
   - The TensorFlow Lite model should be generated in the [`../model`](../model) directory (see that directory's README for details).

3. **The compiled runner**
   - The CPU inference executable, which you build from the source in this directory.

Place the `.rgb` input file, the `.tflite` model, and the compiled runner on your board. You can then run inference using these files.

---
