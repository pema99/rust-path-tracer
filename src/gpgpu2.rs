use std::{marker::PhantomData, sync::Arc, time::Duration};

use pollster::FutureExt;
use wgpu::util::DeviceExt;

pub const GPU_BUFFER_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_truncate(
    wgpu::BufferUsages::STORAGE.bits()
        | wgpu::BufferUsages::COPY_SRC.bits()
        | wgpu::BufferUsages::COPY_DST.bits(),
);
pub const GPU_UNIFORM_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_truncate(
    wgpu::BufferUsages::UNIFORM.bits() | wgpu::BufferUsages::COPY_DST.bits(),
);
pub const GPU_CONST_IMAGE_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_truncate(
    wgpu::TextureUsages::TEXTURE_BINDING.bits() | wgpu::TextureUsages::COPY_DST.bits(),
);

pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
}

pub struct GpuBuffer<'fw, T> {
    fw: &'fw GpuContext,
    buffer: wgpu::Buffer,
    size: u64,
    marker: PhantomData<T>,
}

pub struct GpuFloatImage {
    texture_view: wgpu::TextureView,
}

pub struct GpuSampler {
    sampler: wgpu::Sampler,
    filter_mode: wgpu::FilterMode,
}

pub struct GpuKernelBuilder<'fw, 'res> {
    fw: &'fw GpuContext,
    shader: wgpu::ShaderModule,
    entry_point: String,
    layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    bind_entries: Vec<wgpu::BindGroupEntry<'res>>,
}

pub struct GpuKernel<'fw> {
    fw: &'fw GpuContext,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    entry_point: String,
}

impl Default for GpuContext {
    fn default() -> Self {
        let power_preference = wgpu::util::power_preference_from_env()
            .unwrap_or(wgpu::PowerPreference::HighPerformance);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                ..Default::default()
            })
            .block_on()
            .expect("Failed at adapter creation.");
        Self::new(adapter, Duration::from_millis(1)).block_on()
    }
}

impl GpuContext {
    pub async fn new(adapter: wgpu::Adapter, polling_time: Duration) -> Self {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: adapter.features(),
                    limits: adapter.limits(),
                },
                None,
            )
            .await
            .expect("Failed at device creation.");

        let device = Arc::new(device);
        let polling_device = Arc::clone(&device);

        Self { device, queue }
    }
}

impl GpuSampler {
    pub fn new(
        fw: &GpuContext,
        wrap_mode: wgpu::AddressMode,
        filter_mode: wgpu::FilterMode,
    ) -> Self {
        let sampler = fw.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wrap_mode,
            address_mode_v: wrap_mode,
            address_mode_w: wrap_mode,
            mag_filter: filter_mode,
            min_filter: filter_mode,
            mipmap_filter: filter_mode,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            compare: None,
            anisotropy_clamp: None,
            border_color: None,
        });
        Self {
            sampler,
            filter_mode,
        }
    }
}

impl<'fw> GpuKernel<'fw> {
    pub fn enqueue(&self, x: u32, y: u32, z: u32) {
        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

            cpass.set_pipeline(&self.pipeline);

            cpass.set_bind_group(0, &self.bind_group, &[]);

            cpass.dispatch_workgroups(x, y, z);
        }

        self.fw.queue.submit(Some(encoder.finish()));
    }

    pub fn begin_work(&self) {

    }
}

impl<'fw, 'res> GpuKernelBuilder<'fw, 'res> {
    pub fn new(fw: &'fw GpuContext, spirv: &[u8], entry_point: &str) -> Self {
        let source = wgpu::util::make_spirv(spirv);
        let shader = fw
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source,
            });

        Self {
            fw,
            shader,
            entry_point: entry_point.into(),
            layout_entries: Vec::new(),
            bind_entries: Vec::new(),
        }
    }

    pub fn bind_uniform_buffer<T: bytemuck::Pod>(
        mut self,
        uniform_buffer: &'res GpuBuffer<T>,
    ) -> Self {
        let bind_id = self.layout_entries.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Uniform,
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: uniform_buffer.buffer.as_entire_binding(),
        };

        self.layout_entries.push(bind_entry);
        self.bind_entries.push(bind);

        self
    }

    pub fn bind_buffer<T: bytemuck::Pod>(
        mut self,
        storage_buffer: &'res GpuBuffer<T>,
        writable: bool,
    ) -> Self {
        let bind_id = self.layout_entries.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage {
                    read_only: !writable,
                },
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: storage_buffer.buffer.as_entire_binding(),
        };

        self.layout_entries.push(bind_entry);
        self.bind_entries.push(bind);

        self
    }

    pub fn bind_image(mut self, img: &'res GpuFloatImage) -> Self {
        let bind_id = self.layout_entries.len() as u32;

        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: wgpu::BindingResource::TextureView(&img.texture_view),
        };

        self.layout_entries.push(bind_entry);
        self.bind_entries.push(bind);

        self
    }

    pub fn bind_sampler(mut self, sampler: &'res crate::gpgpu2::GpuSampler) -> Self {
        let bind_id = self.layout_entries.len() as u32;
        let sampler_binding_type = match sampler.filter_mode {
            wgpu::FilterMode::Linear => wgpu::SamplerBindingType::Filtering,
            wgpu::FilterMode::Nearest => wgpu::SamplerBindingType::NonFiltering,
        };
        let bind_entry = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(sampler_binding_type),
            count: None,
        };

        let bind = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: wgpu::BindingResource::Sampler(&sampler.sampler),
        };

        self.layout_entries.push(bind_entry);
        self.bind_entries.push(bind);

        self
    }

    pub fn build(self) -> GpuKernel<'fw> {
        let bind_group_layout =
            self.fw
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &self.layout_entries,
                });

        let bind_group = self
            .fw
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &self.bind_entries,
            });

        let pipeline_layout =
            self.fw
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .fw
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                module: &self.shader,
                entry_point: &self.entry_point,
                layout: Some(&pipeline_layout),
            });

        GpuKernel {
            fw: self.fw,
            pipeline,
            bind_group,
            entry_point: self.entry_point,
        }
    }
}

impl<'fw, T: bytemuck::Pod> GpuBuffer<'fw, T> {
    pub fn read(&self, buf: &mut [T]) -> u64 {
        let output_size = (buf.len() * std::mem::size_of::<T>()) as u64;
        let download_size = if output_size > self.size {
            self.size
        } else {
            output_size
        };

        let (tx, rx) = futures::channel::oneshot::channel();
        wgpu::util::DownloadBuffer::read_buffer(
            &self.fw.device,
            &self.fw.queue,
            &self.buffer.slice(..download_size as u64),
            move |result| {
                tx.send(result)
                    .unwrap_or_else(|_| panic!("Failed to download buffer."));
            },
        );
        self.fw.device.poll(wgpu::Maintain::Wait);
        let download = rx.block_on().unwrap().unwrap();

        buf.copy_from_slice(bytemuck::cast_slice(&download));

        download_size
    }

    pub fn write(&self, buf: &[T]) -> u64 {
        let input_size = (buf.len() * std::mem::size_of::<T>()) as u64;
        let upload_size = if input_size > self.size {
            self.size
        } else {
            input_size
        };

        self.fw
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(buf));

        let encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.fw.queue.submit(Some(encoder.finish()));

        upload_size
    }

    pub fn new(
        fw: &'fw crate::gpgpu2::GpuContext,
        slice: &[T],
        usages: wgpu::BufferUsages,
    ) -> Self {
        let size = (slice.len() * std::mem::size_of::<T>()) as u64;
        let buf = fw
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(slice),
                usage: usages,
            });

        Self {
            fw,
            buffer: buf,
            size,
            marker: PhantomData,
        }
    }
}

impl GpuFloatImage {
    pub fn from_bytes(
        fw: &crate::gpgpu2::GpuContext,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture = fw.device.create_texture_with_data(
            &fw.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size,
                dimension: wgpu::TextureDimension::D2,
                mip_level_count: 1,
                sample_count: 1,
                format,
                usage: GPU_CONST_IMAGE_USAGES,
                view_formats: &[format]
            },
            data,
        );

        let texture_view = texture.create_view(&Default::default());

        Self { texture_view }
    }
}
