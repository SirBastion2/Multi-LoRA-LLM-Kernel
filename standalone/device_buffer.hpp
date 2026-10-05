#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

class DeviceBuffer {
 public:
  DeviceBuffer() : ptr_(nullptr), bytes_(0) {}
  explicit DeviceBuffer(size_t bytes) : ptr_(nullptr), bytes_(0) { allocate(bytes); }

  ~DeviceBuffer() { free(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }

  void allocate(size_t bytes) {
    free();
    if (bytes == 0) {
      return;
    }
    cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    bytes_ = bytes;
  }

  void free() {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      bytes_ = 0;
    }
  }

  void* data() { return ptr_; }
  const void* data() const { return ptr_; }
  size_t size() const { return bytes_; }

 private:
  void* ptr_;
  size_t bytes_;
};
