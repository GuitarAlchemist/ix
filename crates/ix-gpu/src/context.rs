//! GPU context: device initialization and buffer management.
//!
//! Handles WGPU device/queue creation and provides helpers for
//! buffer allocation and data transfer.

use wgpu::*;

/// GPU compute context — holds the device and queue.
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub adapter_info: AdapterInfo,
}

impl GpuContext {
    /// Initialize a GPU context. Prefers discrete GPU, falls back to integrated.
    ///
    /// This uses `pollster::block_on` internally — call from sync code.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    /// Async GPU initialization.
    pub async fn new_async() -> Result<Self, GpuError> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("ix-gpu"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: MemoryHints::Performance,
                experimental_features: ExperimentalFeatures::default(),
                trace: Trace::Off,
            })
            .await
            .map_err(|e: RequestDeviceError| GpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            adapter_info: info,
        })
    }

    /// Get a human-readable description of the GPU.
    pub fn gpu_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get the backend being used (Vulkan, DX12, Metal, etc.).
    pub fn backend(&self) -> Backend {
        self.adapter_info.backend
    }

    /// Create a storage buffer initialized with f32 data.
    pub fn create_buffer_init(&self, label: &str, data: &[f32]) -> Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
    }

    /// Create a read-back buffer for retrieving results from GPU.
    pub fn create_readback_buffer(&self, label: &str, size: u64) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a storage buffer for output.
    pub fn create_output_buffer(&self, label: &str, size: u64) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Read f32 values back from a buffer.
    pub fn read_buffer(&self, buffer: &Buffer, size: u64) -> Vec<f32> {
        let readback = self.create_readback_buffer("readback", size);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = self.device.poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();

        result
    }

    /// Create a compute pipeline from WGSL shader source.
    pub fn create_compute_pipeline(
        &self,
        label: &str,
        shader_source: &str,
        entry_point: &str,
    ) -> ComputePipeline {
        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        self.device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // Auto layout
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }
}

/// GPU errors.
#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    DeviceRequest(String),
    ShaderCompilation(String),
    BufferMapping(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "No compatible GPU adapter found"),
            Self::DeviceRequest(e) => write!(f, "Failed to create GPU device: {}", e),
            Self::ShaderCompilation(e) => write!(f, "Shader compilation failed: {}", e),
            Self::BufferMapping(e) => write!(f, "Buffer mapping failed: {}", e),
        }
    }
}

impl std::error::Error for GpuError {}
