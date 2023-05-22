// Heavily modified, trimmed down version of gpgpu-rs (https://github.com/UpsettingBoy/gpgpu-rs)

// Original license:
/*
EUROPEAN UNION PUBLIC LICENCE v. 1.2

EUPL © the European Union 2007, 2016

This European Union Public Licence (the ‘EUPL’) applies to the Work (as defined below) which is provided under the terms of this Licence. Any use of the Work, other than as authorised under this Licence is prohibited (to the extent such use is covered by a right of the copyright holder of the Work).

The Work is provided under the terms of this Licence when the Licensor (as defined below) has placed the following notice immediately following the copyright notice for the Work:

Licensed under the EUPL

or has expressed by any other means his willingness to license under the EUPL.
1. Definitions

In this Licence, the following terms have the following meaning:

— ‘The Licence’: this Licence.

— ‘The Original Work’: the work or software distributed or communicated by the Licensor under this Licence, available as Source Code and also as Executable Code as the case may be.

— ‘Derivative Works’: the works or software that could be created by the Licensee, based upon the Original Work or modifications thereof. This Licence does not define the extent of modification or dependence on the Original Work required in order to classify a work as a Derivative Work; this extent is determined by copyright law applicable in the country mentioned in Article 15.

— ‘The Work’: the Original Work or its Derivative Works.

— ‘The Source Code’: the human-readable form of the Work which is the most convenient for people to study and modify.

— ‘The Executable Code’: any code which has generally been compiled and which is meant to be interpreted by a computer as a program.

— ‘The Licensor’: the natural or legal person that distributes or communicates the Work under the Licence.

— ‘Contributor(s)’: any natural or legal person who modifies the Work under the Licence, or otherwise contributes to the creation of a Derivative Work.

— ‘The Licensee’ or ‘You’: any natural or legal person who makes any usage of the Work under the terms of the Licence.

— ‘Distribution’ or ‘Communication’: any act of selling, giving, lending, renting, distributing, communicating, transmitting, or otherwise making available, online or offline, copies of the Work or providing access to its essential functionalities at the disposal of any other natural or legal person.
2. Scope of the rights granted by the Licence

The Licensor hereby grants You a worldwide, royalty-free, non-exclusive, sublicensable licence to do the following, for the duration of copyright vested in the Original Work:

— use the Work in any circumstance and for all usage,

— reproduce the Work,

— modify the Work, and make Derivative Works based upon the Work,

— communicate to the public, including the right to make available or display the Work or copies thereof to the public and perform publicly, as the case may be, the Work,

— distribute the Work or copies thereof,

— lend and rent the Work or copies thereof,

— sublicense rights in the Work or copies thereof.

Those rights can be exercised on any media, supports and formats, whether now known or later invented, as far as the applicable law permits so.

In the countries where moral rights apply, the Licensor waives his right to exercise his moral right to the extent allowed by law in order to make effective the licence of the economic rights here above listed.

The Licensor grants to the Licensee royalty-free, non-exclusive usage rights to any patents held by the Licensor, to the extent necessary to make use of the rights granted on the Work under this Licence.
3. Communication of the Source Code

The Licensor may provide the Work either in its Source Code form, or as Executable Code. If the Work is provided as Executable Code, the Licensor provides in addition a machine-readable copy of the Source Code of the Work along with each copy of the Work that the Licensor distributes or indicates, in a notice following the copyright notice attached to the Work, a repository where the Source Code is easily and freely accessible for as long as the Licensor continues to distribute or communicate the Work.
4. Limitations on copyright

Nothing in this Licence is intended to deprive the Licensee of the benefits from any exception or limitation to the exclusive rights of the rights owners in the Work, of the exhaustion of those rights or of other applicable limitations thereto.
5. Obligations of the Licensee

The grant of the rights mentioned above is subject to some restrictions and obligations imposed on the Licensee. Those obligations are the following:

    Attribution right: The Licensee shall keep intact all copyright, patent or trademarks notices and all notices that refer to the Licence and to the disclaimer of warranties. The Licensee must include a copy of such notices and a copy of the Licence with every copy of the Work he/she distributes or communicates. The Licensee must cause any Derivative Work to carry prominent notices stating that the Work has been modified and the date of modification.
    Copyleft clause: If the Licensee distributes or communicates copies of the Original Works or Derivative Works, this Distribution or Communication will be done under the terms of this Licence or of a later version of this Licence unless the Original Work is expressly distributed only under this version of the Licence — for example by communicating ‘EUPL v. 1.2 only’. The Licensee (becoming Licensor) cannot offer or impose any additional terms or conditions on the Work or Derivative Work that alter or restrict the terms of the Licence.
    Compatibility clause: If the Licensee Distributes or Communicates Derivative Works or copies thereof based upon both the Work and another work licensed under a Compatible Licence, this Distribution or Communication can be done under the terms of this Compatible Licence. For the sake of this clause, ‘Compatible Licence’ refers to the licences listed in the appendix attached to this Licence. Should the Licensee's obligations under the Compatible Licence conflict with his/her obligations under this Licence, the obligations of the Compatible Licence shall prevail.
    Provision of Source Code: When distributing or communicating copies of the Work, the Licensee will provide a machine-readable copy of the Source Code or indicate a repository where this Source will be easily and freely available for as long as the Licensee continues to distribute or communicate the Work.
    Legal Protection: This Licence does not grant permission to use the trade names, trademarks, service marks, or names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the copyright notice.

6. Chain of Authorship

The original Licensor warrants that the copyright in the Original Work granted hereunder is owned by him/her or licensed to him/her and that he/she has the power and authority to grant the Licence.

Each Contributor warrants that the copyright in the modifications he/she brings to the Work are owned by him/her or licensed to him/her and that he/she has the power and authority to grant the Licence.

Each time You accept the Licence, the original Licensor and subsequent Contributors grant You a licence to their contri­ butions to the Work, under the terms of this Licence.
7. Disclaimer of Warranty

The Work is a work in progress, which is continuously improved by numerous Contributors. It is not a finished work and may therefore contain defects or ‘bugs’ inherent to this type of development.

For the above reason, the Work is provided under the Licence on an ‘as is’ basis and without warranties of any kind concerning the Work, including without limitation merchantability, fitness for a particular purpose, absence of defects or errors, accuracy, non-infringement of intellectual property rights other than copyright as stated in Article 6 of this Licence.

This disclaimer of warranty is an essential part of the Licence and a condition for the grant of any rights to the Work.
8. Disclaimer of Liability

Except in the cases of wilful misconduct or damages directly caused to natural persons, the Licensor will in no event be liable for any direct or indirect, material or moral, damages of any kind, arising out of the Licence or of the use of the Work, including without limitation, damages for loss of goodwill, work stoppage, computer failure or malfunction, loss of data or any commercial damage, even if the Licensor has been advised of the possibility of such damage. However, the Licensor will be liable under statutory product liability laws as far such laws apply to the Work.
9. Additional agreements

While distributing the Work, You may choose to conclude an additional agreement, defining obligations or services consistent with this Licence. However, if accepting obligations, You may act only on your own behalf and on your sole responsibility, not on behalf of the original Licensor or any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against such Contributor by the fact You have accepted any warranty or additional liability.
10. Acceptance of the Licence

The provisions of this Licence can be accepted by clicking on an icon ‘I agree’ placed under the bottom of a window displaying the text of this Licence or by affirming consent in any other similar way, in accordance with the rules of applicable law. Clicking on that icon indicates your clear and irrevocable acceptance of this Licence and all of its terms and conditions.

Similarly, you irrevocably accept this Licence and all of its terms and conditions by exercising any rights granted to You by Article 2 of this Licence, such as the use of the Work, the creation by You of a Derivative Work or the Distribution or Communication by You of the Work or copies thereof.
11. Information to the public

In case of any Distribution or Communication of the Work by means of electronic communication by You (for example, by offering to download the Work from a remote location) the distribution channel or media (for example, a website) must at least provide to the public the information requested by the applicable law regarding the Licensor, the Licence and the way it may be accessible, concluded, stored and reproduced by the Licensee.
12. Termination of the Licence

The Licence and the rights granted hereunder will terminate automatically upon any breach by the Licensee of the terms of the Licence.

Such a termination will not terminate the licences of any person who has received the Work from the Licensee under the Licence, provided such persons remain in full compliance with the Licence.
13. Miscellaneous

Without prejudice of Article 9 above, the Licence represents the complete agreement between the Parties as to the Work.

If any provision of the Licence is invalid or unenforceable under applicable law, this will not affect the validity or enforceability of the Licence as a whole. Such provision will be construed or reformed so as necessary to make it valid and enforceable.

The European Commission may publish other linguistic versions or new versions of this Licence or updated versions of the Appendix, so far this is required and reasonable, without reducing the scope of the rights granted by the Licence. New versions of the Licence will be published with a unique version number.

All linguistic versions of this Licence, approved by the European Commission, have identical value. Parties can take advantage of the linguistic version of their choice.
14. Jurisdiction

Without prejudice to specific agreement between parties,

— any litigation resulting from the interpretation of this License, arising between the European Union institutions, bodies, offices or agencies, as a Licensor, and any Licensee, will be subject to the jurisdiction of the Court of Justice of the European Union, as laid down in article 272 of the Treaty on the Functioning of the European Union,

— any litigation arising between other parties and resulting from the interpretation of this License, will be subject to the exclusive jurisdiction of the competent court where the Licensor resides or conducts its primary business.
15. Applicable Law

Without prejudice to specific agreement between parties,

— this Licence shall be governed by the law of the European Union Member State where the Licensor has his seat, resides or has his registered office,

— this licence shall be governed by Belgian law if the Licensor has no seat, residence or registered office inside a European Union Member State.
Appendix

‘Compatible Licences’ according to Article 5 EUPL are:

— GNU General Public License (GPL) v. 2, v. 3

— GNU Affero General Public License (AGPL) v. 3

— Open Software License (OSL) v. 2.1, v. 3.0

— Eclipse Public License (EPL) v. 1.0

— CeCILL v. 2.0, v. 2.1

— Mozilla Public Licence (MPL) v. 2

— GNU Lesser General Public Licence (LGPL) v. 2.1, v. 3

— Creative Commons Attribution-ShareAlike v. 3.0 Unported (CC BY-SA 3.0) for works other than software

— European Union Public Licence (EUPL) v. 1.1, v. 1.2

— Québec Free and Open-Source Licence — Reciprocity (LiLiQ-R) or Strong Reciprocity (LiLiQ-R+)

The European Commission may update this Appendix to later versions of the above licences without producing a new version of the EUPL, as long as they provide the rights granted in Article 2 of this Licence and protect the covered Source Code from exclusive appropriation.

All other changes or additions to this Appendix require the production of a new EUPL version.
*/

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
