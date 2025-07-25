
# 1. Build models
```sh
./build-linux.sh <demo_name>
```

For example, to build the MobileNet demo:

```sh
./build-linux.sh mobilenet
```


### Requirements

- The demo you want to build must exist under the [`models/`](./models/) directory.
- Inside your demo directory (e.g., [`models/mobilenet/`](./models/mobilenet/)), there must be a subdirectory called [`inference_npu/`](./models/mobilenet/inference_npu/) containing a [`CMakeLists.txt`](./models/mobilenet/inference_npu/CMakeLists.txt) and the source code for the demo.

---

## Project Directory and Toolchain Setup

All of the paths and routes in this project are designed with the expectation that the top-level `rv1103` directory is located under:

```
/toolchain/Projects/rv1103
```

- The cross-compiler for rv1103/rv1106 is expected to be available at:
  ```
  /toolchain/Projects/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc
  ```
- The TensorFlow build (tf-build) and other required tools should also be placed under `/toolchain/Projects/rv1103`.

If you need the cross-compiler and tf-build, you can download them from the following link:

[Google Drive: rv1103 toolchain and tf-build](https://drive.google.com/drive/folders/1Ofj4RRBqKj-Q-BhMlN6Pdzs47BuJpX1y?usp=drive_link)

If your setup is different, you may need to adjust the paths in the scripts accordingly


