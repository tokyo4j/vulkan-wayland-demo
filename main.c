// SPDX-License-Identifier: MIT
// based on simple-vulkan.c in weston by Erico Nunes
#include "matrix.h"
#include "shaders/main_frag.spv.h"
#include "shaders/main_vert.spv.h"
#include "util.h"
#include "xdg-shell-protocol.h"
#include <assert.h>
#include <linux/input.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <wayland-client.h>
#include <wayland-cursor.h>

#define VK_USE_PLATFORM_WAYLAND_KHR
#define VK_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_wayland.h>

#define MAX_NUM_IMAGES 5
#define MAX_CONCURRENT_FRAMES 2

#define VK_CHECK(func)                                                         \
	{                                                                      \
		const VkResult __result = func;                                \
		if (__result != VK_SUCCESS) {                                  \
			fprintf(stderr,                                        \
				"%s %d %s Error: %s failed with VkResult "     \
				"%d\n",                                        \
				__FILE__, __LINE__, __func__, #func,           \
				__result);                                     \
			abort();                                               \
		}                                                              \
	}
#define MASK_CONTAINS(mask, flag) (((mask) & (flag)) == (flag))

struct window;
struct seat;

struct display {
	struct wl_display *display;
	struct wl_registry *registry;
	struct wl_compositor *compositor;
	struct xdg_wm_base *wm_base;
	struct wl_seat *seat;
	struct wl_pointer *pointer;
	struct wl_touch *touch;
	struct wl_keyboard *keyboard;
	struct wl_shm *shm;
	struct wl_cursor_theme *cursor_theme;
	struct wl_cursor *default_cursor;
	struct wl_surface *cursor_surface;
	struct wp_tearing_control_manager_v1 *tearing_manager;
	struct wp_viewporter *viewporter;
	struct wp_fractional_scale_manager_v1 *fractional_scale_manager;
	struct window *window;

	struct wl_list outputs; /* struct output::link */
};

struct geometry {
	int width, height;
};

struct window_image {
	VkImageView image_view;
	VkImage image;
	VkSemaphore render_done;
};

struct window_buffer {
	VkBuffer buffer;
	VkDeviceMemory mem;
	void *map;
};

struct window_frame {
	VkSemaphore image_acquired;
	VkFence fence;
	VkCommandBuffer cmd_buffer;

	VkDescriptorSet descriptor_set;
	struct window_buffer ubo_buffer;
};

struct window_vulkan_pipeline {
	VkDescriptorSetLayout descriptor_set_layout;
	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;
};

struct window_vulkan {
	VkInstance inst;
	VkPhysicalDevice phys_dev;
	VkDevice dev;

	VkQueue queue;
	uint32_t queue_family;

	VkDescriptorPool descriptor_pool;
	VkCommandPool cmd_pool;

	struct window_vulkan_pipeline pipeline;

	VkSwapchainKHR swapchain;
	VkPresentModeKHR present_mode;
	VkSurfaceKHR surface;

	VkFormat format;
	uint32_t image_count;
	struct window_image images[MAX_NUM_IMAGES];
	uint32_t frame_index;
	struct window_frame frames[MAX_CONCURRENT_FRAMES];

	struct window_buffer vertex_buffer;

	PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR
		get_wayland_presentation_support;
	PFN_vkCreateWaylandSurfaceKHR create_wayland_surface;
};

struct window {
	struct display *display;
	struct geometry window_size;
	struct geometry logical_size;
	struct geometry buffer_size;
	int32_t buffer_scale;
	enum wl_output_transform buffer_transform;
	bool needs_buffer_geometry_update;

	uint32_t frames;
	uint32_t initial_frame_time;
	uint32_t benchmark_time;
	struct wl_surface *surface;
	struct xdg_surface *xdg_surface;
	struct xdg_toplevel *xdg_toplevel;
	int fullscreen, maximized, opaque, delay;
	bool fullscreen_ratio;
	bool wait_for_configure;

	struct window_vulkan vk;

	struct wl_list window_outputs; /* struct window_output::link */
};

struct output {
	struct display *display;
	struct wl_output *wl_output;
	uint32_t name;
	struct wl_list link; /* struct display::outputs */
	enum wl_output_transform transform;
	int32_t scale;
};

struct window_output {
	struct output *output;
	struct wl_list link; /* struct window::window_outputs */
};

static void
pnext(void *base, void *next)
{
	VkBaseOutStructure *b = base;
	VkBaseOutStructure *n = next;
	n->pNext = b->pNext;
	b->pNext = n;
}

static int running = 1;

static int32_t
compute_buffer_scale(struct window *window)
{
	struct window_output *window_output;
	int32_t scale = 1;

	wl_list_for_each(window_output, &window->window_outputs, link) {
		if (window_output->output->scale > scale)
			scale = window_output->output->scale;
	}

	return scale;
}

static enum wl_output_transform
compute_buffer_transform(struct window *window)
{
	struct window_output *window_output;
	enum wl_output_transform transform = WL_OUTPUT_TRANSFORM_NORMAL;

	wl_list_for_each(window_output, &window->window_outputs, link) {
		/* If the surface spans over multiple outputs the optimal
		 * transform value can be ambiguous. Thus just return the value
		 * from the oldest entered output.
		 */
		transform = window_output->output->transform;
		break;
	}

	return transform;
}

static void
create_image_view(VkDevice device, VkImage image, VkFormat format,
	VkImageView *image_view)
{
	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.subresourceRange.baseMipLevel = 0,
		.subresourceRange.levelCount = 1,
		.subresourceRange.baseArrayLayer = 0,
		.subresourceRange.layerCount = 1,
	};

	VK_CHECK(vkCreateImageView(device, &view_info, NULL, image_view));
}

static void
create_swapchain(struct window *window)
{
	VkSurfaceCapabilitiesKHR surface_caps;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(window->vk.phys_dev,
		window->vk.surface, &surface_caps);

	VkBool32 supported;
	vkGetPhysicalDeviceSurfaceSupportKHR(window->vk.phys_dev, 0,
		window->vk.surface, &supported);
	assert(supported);

	uint32_t present_mode_count;
	vkGetPhysicalDeviceSurfacePresentModesKHR(window->vk.phys_dev,
		window->vk.surface, &present_mode_count, NULL);
	VkPresentModeKHR present_modes[present_mode_count];
	vkGetPhysicalDeviceSurfacePresentModesKHR(window->vk.phys_dev,
		window->vk.surface, &present_mode_count, present_modes);

	assert(window->vk.present_mode >= 0 && window->vk.present_mode < 4);
	supported = false;
	for (size_t i = 0; i < present_mode_count; ++i) {
		if (present_modes[i] == window->vk.present_mode) {
			supported = true;
			break;
		}
	}

	if (!supported) {
		fprintf(stderr, "Present mode %d unsupported\n",
			window->vk.present_mode);
		abort();
	}

	uint32_t min_image_count = 2;
	if (min_image_count < surface_caps.minImageCount)
		min_image_count = surface_caps.minImageCount;

	if (surface_caps.maxImageCount > 0
		&& min_image_count > surface_caps.maxImageCount)
		min_image_count = surface_caps.maxImageCount;

	VkSwapchainCreateInfoKHR swapchain_create_info = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.flags = 0,
		.surface = window->vk.surface,
		.minImageCount = min_image_count,
		.imageFormat = window->vk.format,
		.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
		.imageExtent =
			(VkExtent2D){
				.width = window->buffer_size.width,
				.height = window->buffer_size.height,
			},
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = &window->vk.queue_family,
		.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
	};
	if (surface_caps.supportedCompositeAlpha
			& VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR
		&& !window->opaque) {
		swapchain_create_info.compositeAlpha =
			VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
	} else {
		swapchain_create_info.compositeAlpha =
			VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	}

	swapchain_create_info.presentMode = window->vk.present_mode;
	vkCreateSwapchainKHR(window->vk.dev, &swapchain_create_info, NULL,
		&window->vk.swapchain);

	vkGetSwapchainImagesKHR(window->vk.dev, window->vk.swapchain,
		&window->vk.image_count, NULL);
	assert(window->vk.image_count > 0);
	VkImage swapchain_images[window->vk.image_count];
	vkGetSwapchainImagesKHR(window->vk.dev, window->vk.swapchain,
		&window->vk.image_count, swapchain_images);

	assert(window->vk.image_count <= ARRAY_SIZE(window->vk.images));
	for (uint32_t i = 0; i < window->vk.image_count; i++) {
		create_image_view(window->vk.dev, swapchain_images[i],
			window->vk.format, &window->vk.images[i].image_view);
		window->vk.images[i].image = swapchain_images[i];

		const VkSemaphoreCreateInfo semaphore_create_info = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};
		VK_CHECK(vkCreateSemaphore(window->vk.dev,
			&semaphore_create_info, NULL,
			&window->vk.images[i].render_done));
	}
}

static void
destroy_swapchain(struct window *window)
{
	vkDeviceWaitIdle(window->vk.dev);

	for (uint32_t i = 0; i < window->vk.image_count; i++) {
		vkDestroySemaphore(window->vk.dev,
			window->vk.images[i].render_done, NULL);
		vkDestroyImageView(window->vk.dev,
			window->vk.images[i].image_view, NULL);
	}

	vkDestroySwapchainKHR(window->vk.dev, window->vk.swapchain, NULL);
}

static void
recreate_swapchain(struct window *window)
{
	destroy_swapchain(window);
	create_swapchain(window);
}

static void
update_buffer_geometry(struct window *window)
{
	enum wl_output_transform new_buffer_transform;
	struct geometry new_buffer_size;

	new_buffer_transform = compute_buffer_transform(window);
	if (window->buffer_transform != new_buffer_transform) {
		window->buffer_transform = new_buffer_transform;
		wl_surface_set_buffer_transform(window->surface,
			window->buffer_transform);
	}

	switch (window->buffer_transform) {
	case WL_OUTPUT_TRANSFORM_NORMAL:
	case WL_OUTPUT_TRANSFORM_180:
	case WL_OUTPUT_TRANSFORM_FLIPPED:
	case WL_OUTPUT_TRANSFORM_FLIPPED_180:
		new_buffer_size.width = window->logical_size.width;
		new_buffer_size.height = window->logical_size.height;
		break;
	case WL_OUTPUT_TRANSFORM_90:
	case WL_OUTPUT_TRANSFORM_270:
	case WL_OUTPUT_TRANSFORM_FLIPPED_90:
	case WL_OUTPUT_TRANSFORM_FLIPPED_270:
		new_buffer_size.width = window->logical_size.height;
		new_buffer_size.height = window->logical_size.width;
		break;
	}

	int32_t new_buffer_scale = compute_buffer_scale(window);
	if (window->buffer_scale != new_buffer_scale) {
		window->buffer_scale = new_buffer_scale;
		wl_surface_set_buffer_scale(window->surface,
			window->buffer_scale);
	}

	new_buffer_size.width *= window->buffer_scale;
	new_buffer_size.height *= window->buffer_scale;

	if (window->fullscreen && window->fullscreen_ratio) {
		int new_buffer_size_min =
			MIN(new_buffer_size.width, new_buffer_size.height);
		new_buffer_size.width = new_buffer_size_min;
		new_buffer_size.height = new_buffer_size_min;
	}

	if (window->buffer_size.width != new_buffer_size.width
		|| window->buffer_size.height != new_buffer_size.height) {
		window->buffer_size = new_buffer_size;
	}

	window->needs_buffer_geometry_update = false;
}

static VkFormat
choose_surface_format(struct window *window)
{
	uint32_t num_formats = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(window->vk.phys_dev,
		window->vk.surface, &num_formats, NULL);
	assert(num_formats > 0);

	VkSurfaceFormatKHR formats[num_formats];

	vkGetPhysicalDeviceSurfaceFormatsKHR(window->vk.phys_dev,
		window->vk.surface, &num_formats, formats);

	VkFormat format = VK_FORMAT_UNDEFINED;
	for (int i = 0; i < (int)num_formats; i++) {
		switch (formats[i].format) {
		case VK_FORMAT_B8G8R8A8_UNORM:
			format = formats[i].format;
			break;
		default:
			continue;
		}
	}

	assert(format != VK_FORMAT_UNDEFINED);

	return format;
}

static int
find_memory_type(struct window *window, uint32_t allowed,
	VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(window->vk.phys_dev,
		&mem_properties);

	for (unsigned int i = 0; i < mem_properties.memoryTypeCount; ++i) {
		bool is_allowed = (allowed & (1u << i));
		bool has_properties = MASK_CONTAINS(
			mem_properties.memoryTypes[i].propertyFlags,
			properties);
		if (is_allowed && has_properties) {
			return i;
		}
	}
	return -1;
}

static void
create_buffer(struct window *window, VkDeviceSize size,
	VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
	VkBuffer *buffer, VkDeviceMemory *buffer_memory)
{
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VK_CHECK(vkCreateBuffer(window->vk.dev, &buffer_info, NULL, buffer));

	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(window->vk.dev, *buffer,
		&mem_requirements);

	int memory_type = find_memory_type(window,
		mem_requirements.memoryTypeBits, properties);
	assert(memory_type >= 0);

	const VkMemoryAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_requirements.size,
		.memoryTypeIndex = memory_type,
	};

	VK_CHECK(vkAllocateMemory(window->vk.dev, &alloc_info, NULL,
		buffer_memory));

	VK_CHECK(vkBindBufferMemory(window->vk.dev, *buffer, *buffer_memory,
		0));
}

static void
create_descriptor_set_layout(struct window *window)
{
	struct window_vulkan_pipeline *pipeline = &window->vk.pipeline;

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings =
			(VkDescriptorSetLayoutBinding[]){
				{
					.binding = 0,
					.descriptorType =
						VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags =
						VK_SHADER_STAGE_VERTEX_BIT,
				},
			},
	};
	VK_CHECK(vkCreateDescriptorSetLayout(window->vk.dev, &layout_info, NULL,
		&pipeline->descriptor_set_layout));
}

static void
create_descriptor_set(struct window *window, struct window_frame *frame)
{
	struct window_vulkan_pipeline *pipeline = &window->vk.pipeline;

	const VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = window->vk.descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &pipeline->descriptor_set_layout,
	};
	VK_CHECK(vkAllocateDescriptorSets(window->vk.dev,
		&descriptor_set_allocate_info, &frame->descriptor_set));

	struct window_buffer *ubo_buffer = &frame->ubo_buffer;

	const VkDescriptorBufferInfo descriptor_buffer_info = {
		.buffer = ubo_buffer->buffer,
		.range = VK_WHOLE_SIZE,
	};
	const VkWriteDescriptorSet descriptor_writes[] = {{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = frame->descriptor_set,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &descriptor_buffer_info,
	}};

	vkUpdateDescriptorSets(window->vk.dev, ARRAY_SIZE(descriptor_writes),
		descriptor_writes, 0, NULL);
}

static void
create_pipeline(struct window *window)
{
	struct window_vulkan_pipeline *pipeline = &window->vk.pipeline;

	VkShaderModule vs_module;
	const VkShaderModuleCreateInfo vs_shader_module_create_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(main_vert),
		.pCode = (uint32_t *)main_vert,
	};
	VK_CHECK(vkCreateShaderModule(window->vk.dev,
		&vs_shader_module_create_info, NULL, &vs_module));

	VkShaderModule fs_module;
	const VkShaderModuleCreateInfo fs_shader_module_create_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(main_frag),
		.pCode = (uint32_t *)main_frag,
	};
	VK_CHECK(vkCreateShaderModule(window->vk.dev,
		&fs_shader_module_create_info, NULL, &fs_module));

	const VkPipelineVertexInputStateCreateInfo pipeline_vertex_input_state_create_info = {
		.sType =
			VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 2,
		.pVertexBindingDescriptions =
			(VkVertexInputBindingDescription[]){
				{
					.binding = 0,
					.stride = 3 * sizeof(float),
					.inputRate =
						VK_VERTEX_INPUT_RATE_VERTEX,
				},
				{
					.binding = 1,
					.stride = 3 * sizeof(float),
					.inputRate =
						VK_VERTEX_INPUT_RATE_VERTEX,
				},
			},
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions =
			(VkVertexInputAttributeDescription[]){
				{
					.location = 0,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = 0,
				},
				{
					.location = 1,
					.binding = 1,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = 0,
				},
			},
	};
	const VkPipelineInputAssemblyStateCreateInfo
		pipeline_input_assembly_state_create_info = {
			.sType =
				VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = false,
		};
	const VkPipelineViewportStateCreateInfo
		pipeline_viewport_state_create_info = {
			.sType =
				VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1,
		};
	const VkPipelineRasterizationStateCreateInfo
		pipeline_rasterization_state_create_info = {
			.sType =
				VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.rasterizerDiscardEnable = false,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = VK_FALSE,
			.depthClampEnable = VK_FALSE,
			.lineWidth = 1.0f,
		};
	const VkPipelineMultisampleStateCreateInfo
		pipeline_multisample_state_create_info = {
			.sType =
				VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = 1,
		};
	const VkPipelineColorBlendStateCreateInfo
		pipeline_color_blend_state_create_info = {
			.sType =
				VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments =
				(VkPipelineColorBlendAttachmentState[]){{
					.colorWriteMask =
						VK_COLOR_COMPONENT_A_BIT
						| VK_COLOR_COMPONENT_R_BIT
						| VK_COLOR_COMPONENT_G_BIT
						| VK_COLOR_COMPONENT_B_BIT,
				}}};
	const VkPipelineDynamicStateCreateInfo pipeline_dynamic_state_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates =
			(VkDynamicState[]){
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR,
			},
	};

	const VkPipelineRenderingCreateInfo pipeline_rendering_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.viewMask = 0,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = (VkFormat[]){window->vk.format},
	};

	const VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &pipeline->descriptor_set_layout,
	};
	VK_CHECK(vkCreatePipelineLayout(window->vk.dev,
		&pipeline_layout_create_info, NULL,
		&pipeline->pipeline_layout));

	const VkGraphicsPipelineCreateInfo graphics_pipeline_create_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages =
			(VkPipelineShaderStageCreateInfo[]){
				{
					.sType =
						VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_VERTEX_BIT,
					.module = vs_module,
					.pName = "main",
				},
				{
					.sType =
						VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
					.module = fs_module,
					.pName = "main",
				},
			},
		.pVertexInputState = &pipeline_vertex_input_state_create_info,
		.pInputAssemblyState =
			&pipeline_input_assembly_state_create_info,
		.pViewportState = &pipeline_viewport_state_create_info,
		.pRasterizationState =
			&pipeline_rasterization_state_create_info,
		.pMultisampleState = &pipeline_multisample_state_create_info,
		.pColorBlendState = &pipeline_color_blend_state_create_info,
		.pDynamicState = &pipeline_dynamic_state_create_info,

		.pNext = &pipeline_rendering_info,
		.flags = 0,
		.layout = pipeline->pipeline_layout,
	};
	VK_CHECK(vkCreateGraphicsPipelines(window->vk.dev, VK_NULL_HANDLE, 1,
		&graphics_pipeline_create_info, NULL, &pipeline->pipeline));

	vkDestroyShaderModule(window->vk.dev, fs_module, NULL);
	vkDestroyShaderModule(window->vk.dev, vs_module, NULL);
}

static void
create_vertex_buffer(struct window *window)
{
	// clang-format off
	const float vertices[] = {
		-0.5f,  0.5f, 0.0,
		 0.5f,  0.5f, 0.0,
		 0.0f, -0.5f, 0.0,
	};
	const float colors[] = {
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
	};
	// clang-format on

	uint32_t vertex_buffer_size = sizeof(vertices) + sizeof(colors);

	create_buffer(window, vertex_buffer_size,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
			| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&window->vk.vertex_buffer.buffer,
		&window->vk.vertex_buffer.mem);

	VK_CHECK(vkMapMemory(window->vk.dev, window->vk.vertex_buffer.mem, 0,
		vertex_buffer_size, 0, &window->vk.vertex_buffer.map));

	memcpy(window->vk.vertex_buffer.map, vertices, sizeof(vertices));
	memcpy(window->vk.vertex_buffer.map + sizeof(vertices), colors,
		sizeof(colors));
}

static void
create_descriptor_pool(struct window *window, VkDescriptorPool *descriptor_pool,
	uint32_t base_count, uint32_t maxsets)
{
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.poolSizeCount = 1,
		.pPoolSizes = (VkDescriptorPoolSize[]){{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 * base_count,
		}},
		.maxSets = maxsets,
	};

	VK_CHECK(vkCreateDescriptorPool(window->vk.dev, &pool_info, NULL,
		descriptor_pool));
}

static bool
check_extension(const VkExtensionProperties *avail, uint32_t avail_len,
	const char *name)
{
	for (size_t i = 0; i < avail_len; i++) {
		if (strcmp(avail[i].extensionName, name) == 0) {
			return true;
		}
	}
	return false;
}

static void
load_inst_proc(struct window *window, const char *func, void *proc_ptr)
{
	void *proc = (void *)vkGetInstanceProcAddr(window->vk.inst, func);
	if (proc == NULL) {
		char err[256];
		snprintf(err, sizeof(err),
			"Failed to vkGetInstanceProcAddr %s\n", func);
		err[sizeof(err) - 1] = '\0';
		fprintf(stderr, "%s", err);
		abort();
	}

	*(void **)proc_ptr = proc;
}

static void
create_instance(struct window *window)
{
	uint32_t num_avail_inst_extns;
	VK_CHECK(vkEnumerateInstanceExtensionProperties(NULL,
		&num_avail_inst_extns, NULL));
	assert(num_avail_inst_extns > 0);
	VkExtensionProperties avail_inst_extns[num_avail_inst_extns];
	VK_CHECK(vkEnumerateInstanceExtensionProperties(NULL,
		&num_avail_inst_extns, avail_inst_extns));

	const char *inst_extns[] = {
		VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
		VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
		VK_KHR_SURFACE_EXTENSION_NAME,
		VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
	};

	for (uint32_t i = 0; i < ARRAY_SIZE(inst_extns); i++) {
		if (!check_extension(avail_inst_extns, num_avail_inst_extns,
			    inst_extns[i])) {
			fprintf(stderr, "Unsupported instance extension: %s\n",
				inst_extns[i]);
			abort();
		}
	}

	const VkApplicationInfo app_info = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "simple-vulkan",
		.apiVersion = VK_MAKE_VERSION(1, 3, 0),
	};

	const VkInstanceCreateInfo inst_create_info = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app_info,
		.ppEnabledExtensionNames = inst_extns,
		.enabledExtensionCount = ARRAY_SIZE(inst_extns),
	};

	VK_CHECK(vkCreateInstance(&inst_create_info, NULL, &window->vk.inst));

	load_inst_proc(window, "vkCreateWaylandSurfaceKHR",
		&window->vk.create_wayland_surface);
	load_inst_proc(window,
		"vkGetPhysicalDeviceWaylandPresentationSupportKHR",
		&window->vk.get_wayland_presentation_support);
}

static void
choose_physical_device(struct window *window)
{
	uint32_t n_phys_devs;

	VK_CHECK(vkEnumeratePhysicalDevices(window->vk.inst, &n_phys_devs,
		NULL));
	assert(n_phys_devs != 0);
	VkPhysicalDevice phys_devs[n_phys_devs];
	VK_CHECK(vkEnumeratePhysicalDevices(window->vk.inst, &n_phys_devs,
		phys_devs));

	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	/* Pick the first one */
	for (uint32_t i = 0; i < n_phys_devs; ++i) {
		VkPhysicalDeviceProperties props;

		vkGetPhysicalDeviceProperties(phys_devs[i], &props);

		if (physical_device == VK_NULL_HANDLE) {
			physical_device = phys_devs[i];
			break;
		}
	}

	if (physical_device == VK_NULL_HANDLE) {
		fprintf(stderr, "Unable to find a suitable physical device\n");
		abort();
	}

	window->vk.phys_dev = physical_device;
}

static void
choose_queue_family(struct window *window)
{
	uint32_t n_props = 0;

	vkGetPhysicalDeviceQueueFamilyProperties(window->vk.phys_dev, &n_props,
		NULL);
	VkQueueFamilyProperties props[n_props];
	vkGetPhysicalDeviceQueueFamilyProperties(window->vk.phys_dev, &n_props,
		props);

	uint32_t family_idx = UINT32_MAX;
	/* Pick the first graphics queue */
	for (uint32_t i = 0; i < n_props; ++i) {
		if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			&& props[i].queueCount > 0) {
			family_idx = i;
			break;
		}
	}

	if (family_idx == UINT32_MAX) {
		fprintf(stderr,
			"Physical device exposes no queue with graphics\n");
		abort();
	}

	window->vk.queue_family = family_idx;
}

static void
create_device(struct window *window)
{
	uint32_t num_avail_device_extns;
	VK_CHECK(vkEnumerateDeviceExtensionProperties(window->vk.phys_dev, NULL,
		&num_avail_device_extns, NULL));
	VkExtensionProperties avail_device_extns[num_avail_device_extns];
	VK_CHECK(vkEnumerateDeviceExtensionProperties(window->vk.phys_dev, NULL,
		&num_avail_device_extns, avail_device_extns));

	const char *device_extns[] = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_INCREMENTAL_PRESENT_EXTENSION_NAME,
	};

	for (uint32_t i = 0; i < ARRAY_SIZE(device_extns); i++) {
		if (!check_extension(avail_device_extns, num_avail_device_extns,
			    device_extns[i])) {
			fprintf(stderr, "Unsupported device extension: %s\n",
				device_extns[i]);
			abort();
		}
	}

	const VkDeviceQueueCreateInfo device_queue_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = window->vk.queue_family,
		.queueCount = 1,
		.pQueuePriorities = (float[]){1.0f},
	};

	VkPhysicalDeviceDynamicRenderingFeatures dyn_enable = {
		.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
		.dynamicRendering = VK_TRUE,
	};
	VkPhysicalDeviceFeatures2 features2_enable = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &dyn_enable,
	};

	const VkDeviceCreateInfo device_create_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &device_queue_info,
		.enabledExtensionCount = ARRAY_SIZE(device_extns),
		.ppEnabledExtensionNames = device_extns,
		.pNext = &features2_enable,
	};

	VK_CHECK(vkCreateDevice(window->vk.phys_dev, &device_create_info, NULL,
		&window->vk.dev));
}

static void
init_vulkan(struct window *window)
{
	if (window->needs_buffer_geometry_update) {
		update_buffer_geometry(window);
	}

	create_instance(window);

	choose_physical_device(window);

	choose_queue_family(window);

	create_device(window);

	vkGetDeviceQueue(window->vk.dev, 0, 0, &window->vk.queue);

	if (!window->vk.get_wayland_presentation_support(window->vk.phys_dev, 0,
		    window->display->display)) {
		fprintf(stderr,
			"Vulkan not supported on given Wayland surface");
	}

	const VkWaylandSurfaceCreateInfoKHR wayland_surface_create_info = {
		.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
		.display = window->display->display,
		.surface = window->surface,
	};
	VK_CHECK(window->vk.create_wayland_surface(window->vk.inst,
		&wayland_surface_create_info, NULL, &window->vk.surface));

	window->vk.format = choose_surface_format(window);

	create_descriptor_set_layout(window);
	create_pipeline(window);

	create_vertex_buffer(window);

	create_descriptor_pool(window, &window->vk.descriptor_pool,
		MAX_CONCURRENT_FRAMES, MAX_CONCURRENT_FRAMES);

	const VkCommandPoolCreateInfo cmd_pool_create_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
			| VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = window->vk.queue_family,
	};
	VK_CHECK(vkCreateCommandPool(window->vk.dev, &cmd_pool_create_info,
		NULL, &window->vk.cmd_pool));

	for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
		struct window_frame *frame = &window->vk.frames[i];

		const VkFenceCreateInfo fence_create_info = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		VK_CHECK(vkCreateFence(window->vk.dev, &fence_create_info, NULL,
			&frame->fence));

		const VkCommandBufferAllocateInfo cmd_alloc_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = window->vk.cmd_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VK_CHECK(vkAllocateCommandBuffers(window->vk.dev,
			&cmd_alloc_info, &frame->cmd_buffer));

		const VkSemaphoreCreateInfo semaphore_create_info = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};
		VK_CHECK(vkCreateSemaphore(window->vk.dev,
			&semaphore_create_info, NULL, &frame->image_acquired));

		struct window_buffer *ubo_buffer = &frame->ubo_buffer;

		const uint32_t ubo_size = sizeof(float[16]);

		create_buffer(window, ubo_size,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
				| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&ubo_buffer->buffer, &ubo_buffer->mem);

		VK_CHECK(vkMapMemory(window->vk.dev, ubo_buffer->mem, 0,
			ubo_size, 0, &ubo_buffer->map));

		create_descriptor_set(window, frame);
	}
}

static void
fini_vulkan(struct window *window)
{
	for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
		struct window_frame *frame = &window->vk.frames[i];
		vkDestroySemaphore(window->vk.dev, frame->image_acquired, NULL);
		vkFreeCommandBuffers(window->vk.dev, window->vk.cmd_pool, 1,
			&frame->cmd_buffer);
		vkDestroyFence(window->vk.dev, frame->fence, NULL);

		struct window_buffer *ubo_buffer = &frame->ubo_buffer;
		vkUnmapMemory(window->vk.dev, ubo_buffer->mem);
		vkDestroyBuffer(window->vk.dev, ubo_buffer->buffer, NULL);
		vkFreeMemory(window->vk.dev, ubo_buffer->mem, NULL);
	}

	struct window_vulkan_pipeline *pipeline = &window->vk.pipeline;
	vkDestroyPipelineLayout(window->vk.dev, pipeline->pipeline_layout,
		NULL);
	vkDestroyPipeline(window->vk.dev, pipeline->pipeline, NULL);
	vkDestroyDescriptorSetLayout(window->vk.dev,
		pipeline->descriptor_set_layout, NULL);

	vkDestroyDescriptorPool(window->vk.dev, window->vk.descriptor_pool,
		NULL);

	vkUnmapMemory(window->vk.dev, window->vk.vertex_buffer.mem);
	vkDestroyBuffer(window->vk.dev, window->vk.vertex_buffer.buffer, NULL);
	vkFreeMemory(window->vk.dev, window->vk.vertex_buffer.mem, NULL);

	vkDestroyCommandPool(window->vk.dev, window->vk.cmd_pool, NULL);

	vkDestroyDevice(window->vk.dev, NULL);

	vkDestroySurfaceKHR(window->vk.inst, window->vk.surface, NULL);
	vkDestroyInstance(window->vk.inst, NULL);
}

static void
handle_surface_configure(void *data, struct xdg_surface *surface,
	uint32_t serial)
{
	struct window *window = data;

	xdg_surface_ack_configure(surface, serial);

	window->wait_for_configure = false;
}

static const struct xdg_surface_listener xdg_surface_listener = {
	handle_surface_configure,
};

static void
handle_toplevel_configure(void *data, struct xdg_toplevel *toplevel,
	int32_t width, int32_t height, struct wl_array *states)
{
	struct window *window = data;
	uint32_t *p;

	window->fullscreen = 0;
	window->maximized = 0;
	wl_array_for_each(p, states) {
		uint32_t state = *p;
		switch (state) {
		case XDG_TOPLEVEL_STATE_FULLSCREEN:
			window->fullscreen = 1;
			break;
		case XDG_TOPLEVEL_STATE_MAXIMIZED:
			window->maximized = 1;
			break;
		}
	}

	if (width > 0 && height > 0) {
		if (!window->fullscreen && !window->maximized) {
			window->window_size.width = width;
			window->window_size.height = height;
		}
		window->logical_size.width = width;
		window->logical_size.height = height;
	} else if (!window->fullscreen && !window->maximized) {
		window->logical_size = window->window_size;
	}

	window->needs_buffer_geometry_update = true;
}

static void
handle_toplevel_close(void *data, struct xdg_toplevel *xdg_toplevel)
{
	running = 0;
}

static const struct xdg_toplevel_listener xdg_toplevel_listener = {
	handle_toplevel_configure,
	handle_toplevel_close,
};

static void
add_window_output(struct window *window, struct wl_output *wl_output)
{
	struct output *output;
	struct output *output_found = NULL;
	struct window_output *window_output;

	wl_list_for_each(output, &window->display->outputs, link) {
		if (output->wl_output == wl_output) {
			output_found = output;
			break;
		}
	}

	if (!output_found) {
		return;
	}

	window_output = znew(*window_output);
	window_output->output = output_found;

	wl_list_insert(window->window_outputs.prev, &window_output->link);
	window->needs_buffer_geometry_update = true;
}

static void
destroy_window_output(struct window *window, struct wl_output *wl_output)
{
	struct window_output *window_output;
	struct window_output *window_output_found = NULL;

	wl_list_for_each(window_output, &window->window_outputs, link) {
		if (window_output->output->wl_output == wl_output) {
			window_output_found = window_output;
			break;
		}
	}

	if (window_output_found) {
		wl_list_remove(&window_output_found->link);
		free(window_output_found);
		window->needs_buffer_geometry_update = true;
	}
}

static void
draw_triangle(struct window *window, struct window_frame *frame,
	struct window_image *image)
{
	VkCommandBuffer cmd_buffer = frame->cmd_buffer;

	const VkCommandBufferBeginInfo command_buffer_begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = 0};
	VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &command_buffer_begin_info));

	const VkImageMemoryBarrier barrier_to_color = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image->image,
		.subresourceRange =
			{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	};
	vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0,
		NULL, 1, &barrier_to_color);

	const VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 0.5f}}};
	const VkRenderingAttachmentInfo color_attachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = image->image_view,
		.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.resolveMode = VK_RESOLVE_MODE_NONE,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = clear_color,
	};
	const VkRenderingInfo rendering_info = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea =
			{
				.offset = {0, 0},
				.extent =
					{
						window->buffer_size.width,
						window->buffer_size.height,
					},
			},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &color_attachment,
	};
	vkCmdBeginRendering(cmd_buffer, &rendering_info);

	const VkBuffer buffers[] = {
		window->vk.vertex_buffer.buffer,
		window->vk.vertex_buffer.buffer,
	};
	const VkDeviceSize offsets[] = {
		0,
		3 * sizeof(float[3]),
	};
	vkCmdBindVertexBuffers(cmd_buffer, 0, ARRAY_SIZE(buffers), buffers,
		offsets);

	struct window_vulkan_pipeline *pipeline = &window->vk.pipeline;
	vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipeline->pipeline);

	vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipeline->pipeline_layout, 0, 1, &frame->descriptor_set, 0,
		NULL);

	const VkViewport viewport = {
		.x = 0,
		.y = 0,
		.width = window->buffer_size.width,
		.height = window->buffer_size.height,
		.minDepth = 0,
		.maxDepth = 1,
	};
	vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);

	const VkRect2D scissor = {
		.offset = {0, 0},
		.extent =
			{
				window->buffer_size.width,
				window->buffer_size.height,
			},
	};
	vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

	vkCmdDraw(cmd_buffer, 3, 1, 0, 0);

	vkCmdEndRendering(cmd_buffer);

	const VkImageMemoryBarrier barrier_to_present = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image->image,
		.subresourceRange =
			{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		.dstAccessMask = 0,
	};
	vkCmdPipelineBarrier(cmd_buffer,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1,
		&barrier_to_present);

	VK_CHECK(vkEndCommandBuffer(cmd_buffer));

	const VkPipelineStageFlags wait_stages[] = {
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	};
	const VkSubmitInfo submit_info = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &frame->image_acquired,
		.pWaitDstStageMask = wait_stages,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd_buffer,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &image->render_done,
	};

	VK_CHECK(vkQueueSubmit(window->vk.queue, 1, &submit_info,
		frame->fence));
}

static void
surface_enter(void *data, struct wl_surface *wl_surface,
	struct wl_output *wl_output)
{
	struct window *window = data;

	add_window_output(window, wl_output);
}

static void
surface_leave(void *data, struct wl_surface *wl_surface,
	struct wl_output *wl_output)
{
	struct window *window = data;

	destroy_window_output(window, wl_output);
}

static const struct wl_surface_listener surface_listener = {
	surface_enter,
	surface_leave,
};

static void
create_surface(struct window *window)
{
	struct display *display = window->display;

	window->surface = wl_compositor_create_surface(display->compositor);
	wl_surface_add_listener(window->surface, &surface_listener, window);

	window->xdg_surface =
		xdg_wm_base_get_xdg_surface(display->wm_base, window->surface);
	xdg_surface_add_listener(window->xdg_surface, &xdg_surface_listener,
		window);

	window->xdg_toplevel = xdg_surface_get_toplevel(window->xdg_surface);
	xdg_toplevel_add_listener(window->xdg_toplevel, &xdg_toplevel_listener,
		window);

	xdg_toplevel_set_title(window->xdg_toplevel, "simple-vulkan");
	xdg_toplevel_set_app_id(window->xdg_toplevel,
		"org.freedesktop.weston.simple-vulkan");

	if (window->fullscreen) {
		xdg_toplevel_set_fullscreen(window->xdg_toplevel, NULL);
	} else if (window->maximized) {
		xdg_toplevel_set_maximized(window->xdg_toplevel);
	}

	window->wait_for_configure = true;
	wl_surface_commit(window->surface);
}

static void
destroy_surface(struct window *window)
{
	if (window->xdg_toplevel) {
		xdg_toplevel_destroy(window->xdg_toplevel);
	}
	if (window->xdg_surface) {
		xdg_surface_destroy(window->xdg_surface);
	}
	wl_surface_destroy(window->surface);
}

static void
redraw(struct window *window)
{
	float angle;
	struct mat4f rotation;
	static const uint32_t speed_div = 5, benchmark_interval = 5;
	struct timeval tv;
	VkResult result;

	if (window->needs_buffer_geometry_update) {
		update_buffer_geometry(window);
		recreate_swapchain(window);
	}

	gettimeofday(&tv, NULL);
	uint32_t time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	if (window->frames == 0) {
		window->initial_frame_time = time;
		window->benchmark_time = time;
	}
	if (time - window->benchmark_time > (benchmark_interval * 1000)) {
		printf("%d frames in %d seconds: %f fps\n", window->frames,
			benchmark_interval,
			(float)window->frames / benchmark_interval);
		window->benchmark_time = time;
		window->frames = 0;
	}

	mat4f_init(&rotation);

	angle = ((time - window->initial_frame_time) / speed_div) % 360 * M_PI
		/ 180.0;

	rotation.col[0].el[0] = cos(angle);
	rotation.col[0].el[2] = sin(angle);
	rotation.col[2].el[0] = -sin(angle);
	rotation.col[2].el[2] = cos(angle);

	/* Keep it inside the Vulkan clip volume (z 0..1) */
	mat4f_translate(&rotation, 0.0, 0.0, 0.5);

	switch (window->buffer_transform) {
	default:
	case WL_OUTPUT_TRANSFORM_NORMAL:
	case WL_OUTPUT_TRANSFORM_FLIPPED:
		break;
	case WL_OUTPUT_TRANSFORM_90:
	case WL_OUTPUT_TRANSFORM_FLIPPED_90:
		mat4f_rotate_xy(&rotation, 0, 1);
		break;
	case WL_OUTPUT_TRANSFORM_180:
	case WL_OUTPUT_TRANSFORM_FLIPPED_180:
		mat4f_rotate_xy(&rotation, -1, 0);
		break;
	case WL_OUTPUT_TRANSFORM_270:
	case WL_OUTPUT_TRANSFORM_FLIPPED_270:
		mat4f_rotate_xy(&rotation, 0, -1);
		break;
	}

	struct window_frame *frame = &window->vk.frames[window->vk.frame_index];

	memcpy(frame->ubo_buffer.map, &rotation.colmaj,
		sizeof(rotation.colmaj));

	assert(window->vk.frame_index < ARRAY_SIZE(window->vk.frames));
	vkWaitForFences(window->vk.dev, 1, &frame->fence, VK_TRUE, UINT64_MAX);
	vkResetFences(window->vk.dev, 1, &frame->fence);

	uint32_t image_index;
	result = vkAcquireNextImageKHR(window->vk.dev, window->vk.swapchain,
		UINT64_MAX, frame->image_acquired, VK_NULL_HANDLE,
		&image_index);
	if (result == VK_SUBOPTIMAL_KHR) {
		recreate_swapchain(window);
		return;
	}
	assert(result == VK_SUCCESS);

	assert(image_index < ARRAY_SIZE(window->vk.images));

	struct window_image *image = &window->vk.images[image_index];

	draw_triangle(window, frame, image);

	VkPresentInfoKHR present_info = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &image->render_done,
		.swapchainCount = 1,
		.pSwapchains = &window->vk.swapchain,
		.pImageIndices = &image_index,
		.pResults = NULL,
	};

	const VkRectLayerKHR rect = {
		.offset =
			{
				window->buffer_size.width / 4 - 1,
				window->buffer_size.height / 4 - 1,
			},
		.extent =
			{
				window->buffer_size.width / 2 + 2,
				window->buffer_size.height / 2 + 2,
			},
	};
	const VkPresentRegionKHR region = {
		.rectangleCount = 1,
		.pRectangles = &rect,
	};
	VkPresentRegionsKHR present_regions = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR,
		.swapchainCount = 1,
		.pRegions = &region,
	};
	pnext(&present_info, &present_regions);

	result = vkQueuePresentKHR(window->vk.queue, &present_info);
	if (result != VK_SUCCESS) {
		return;
	}

	window->frames++;
	window->vk.frame_index =
		(window->vk.frame_index + 1) % MAX_CONCURRENT_FRAMES;
}

static void
pointer_handle_enter(void *data, struct wl_pointer *pointer, uint32_t serial,
	struct wl_surface *surface, wl_fixed_t sx, wl_fixed_t sy)
{
	struct display *display = data;
	struct wl_buffer *buffer;
	struct wl_cursor *cursor = display->default_cursor;
	struct wl_cursor_image *image;

	if (display->window->fullscreen) {
		wl_pointer_set_cursor(pointer, serial, NULL, 0, 0);
	} else if (cursor) {
		image = display->default_cursor->images[0];
		buffer = wl_cursor_image_get_buffer(image);
		if (!buffer) {
			return;
		}
		wl_pointer_set_cursor(pointer, serial, display->cursor_surface,
			image->hotspot_x, image->hotspot_y);
		wl_surface_attach(display->cursor_surface, buffer, 0, 0);
		wl_surface_damage(display->cursor_surface, 0, 0, image->width,
			image->height);
		wl_surface_commit(display->cursor_surface);
	}
}

static void
pointer_handle_leave(void *data, struct wl_pointer *pointer, uint32_t serial,
	struct wl_surface *surface)
{
}

static void
pointer_handle_motion(void *data, struct wl_pointer *pointer, uint32_t time,
	wl_fixed_t sx, wl_fixed_t sy)
{
}

static void
pointer_handle_button(void *data, struct wl_pointer *wl_pointer,
	uint32_t serial, uint32_t time, uint32_t button, uint32_t state)
{
	struct display *display = data;

	if (!display->window->xdg_toplevel) {
		return;
	}

	if (button == BTN_LEFT && state == WL_POINTER_BUTTON_STATE_PRESSED) {
		xdg_toplevel_move(display->window->xdg_toplevel, display->seat,
			serial);
	}
}

static void
pointer_handle_axis(void *data, struct wl_pointer *wl_pointer, uint32_t time,
	uint32_t axis, wl_fixed_t value)
{
}

static const struct wl_pointer_listener pointer_listener = {
	pointer_handle_enter,
	pointer_handle_leave,
	pointer_handle_motion,
	pointer_handle_button,
	pointer_handle_axis,
};

static void
touch_handle_down(void *data, struct wl_touch *wl_touch, uint32_t serial,
	uint32_t time, struct wl_surface *surface, int32_t id, wl_fixed_t x_w,
	wl_fixed_t y_w)
{
	struct display *d = (struct display *)data;

	if (!d->wm_base) {
		return;
	}

	xdg_toplevel_move(d->window->xdg_toplevel, d->seat, serial);
}

static void
touch_handle_up(void *data, struct wl_touch *wl_touch, uint32_t serial,
	uint32_t time, int32_t id)
{
}

static void
touch_handle_motion(void *data, struct wl_touch *wl_touch, uint32_t time,
	int32_t id, wl_fixed_t x_w, wl_fixed_t y_w)
{
}

static void
touch_handle_frame(void *data, struct wl_touch *wl_touch)
{
}

static void
touch_handle_cancel(void *data, struct wl_touch *wl_touch)
{
}

static const struct wl_touch_listener touch_listener = {
	touch_handle_down,
	touch_handle_up,
	touch_handle_motion,
	touch_handle_frame,
	touch_handle_cancel,
};

static void
keyboard_handle_keymap(void *data, struct wl_keyboard *keyboard,
	uint32_t format, int fd, uint32_t size)
{
	/* Just so we don’t leak the keymap fd */
	close(fd);
}

static void
keyboard_handle_enter(void *data, struct wl_keyboard *keyboard, uint32_t serial,
	struct wl_surface *surface, struct wl_array *keys)
{
}

static void
keyboard_handle_leave(void *data, struct wl_keyboard *keyboard, uint32_t serial,
	struct wl_surface *surface)
{
}

static void
keyboard_handle_key(void *data, struct wl_keyboard *keyboard, uint32_t serial,
	uint32_t time, uint32_t key, uint32_t state)
{
	struct display *d = data;

	if (!d->wm_base) {
		return;
	}

	if (key == KEY_F11 && state) {
		if (d->window->fullscreen) {
			xdg_toplevel_unset_fullscreen(d->window->xdg_toplevel);
		} else {
			xdg_toplevel_set_fullscreen(d->window->xdg_toplevel,
				NULL);
		}
	} else if (key == KEY_ESC && state) {
		running = 0;
	}
}

static void
keyboard_handle_modifiers(void *data, struct wl_keyboard *keyboard,
	uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched,
	uint32_t mods_locked, uint32_t group)
{
}

static const struct wl_keyboard_listener keyboard_listener = {
	keyboard_handle_keymap,
	keyboard_handle_enter,
	keyboard_handle_leave,
	keyboard_handle_key,
	keyboard_handle_modifiers,
};

static void
seat_handle_capabilities(void *data, struct wl_seat *seat,
	enum wl_seat_capability caps)
{
	struct display *d = data;

	if ((caps & WL_SEAT_CAPABILITY_POINTER) && !d->pointer) {
		d->pointer = wl_seat_get_pointer(seat);
		wl_pointer_add_listener(d->pointer, &pointer_listener, d);
	} else if (!(caps & WL_SEAT_CAPABILITY_POINTER) && d->pointer) {
		wl_pointer_destroy(d->pointer);
		d->pointer = NULL;
	}

	if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !d->keyboard) {
		d->keyboard = wl_seat_get_keyboard(seat);
		wl_keyboard_add_listener(d->keyboard, &keyboard_listener, d);
	} else if (!(caps & WL_SEAT_CAPABILITY_KEYBOARD) && d->keyboard) {
		wl_keyboard_destroy(d->keyboard);
		d->keyboard = NULL;
	}

	if ((caps & WL_SEAT_CAPABILITY_TOUCH) && !d->touch) {
		d->touch = wl_seat_get_touch(seat);
		wl_touch_set_user_data(d->touch, d);
		wl_touch_add_listener(d->touch, &touch_listener, d);
	} else if (!(caps & WL_SEAT_CAPABILITY_TOUCH) && d->touch) {
		wl_touch_destroy(d->touch);
		d->touch = NULL;
	}
}

static const struct wl_seat_listener seat_listener = {
	seat_handle_capabilities,
};

static void
xdg_wm_base_ping(void *data, struct xdg_wm_base *shell, uint32_t serial)
{
	xdg_wm_base_pong(shell, serial);
}

static const struct xdg_wm_base_listener wm_base_listener = {
	xdg_wm_base_ping,
};

static void
display_handle_geometry(void *data, struct wl_output *wl_output, int32_t x,
	int32_t y, int32_t physical_width, int32_t physical_height,
	int32_t subpixel, const char *make, const char *model,
	int32_t transform)
{
	struct output *output = data;

	output->transform = transform;
	output->display->window->needs_buffer_geometry_update = true;
}

static void
display_handle_mode(void *data, struct wl_output *wl_output, uint32_t flags,
	int32_t width, int32_t height, int32_t refresh)
{
}

static void
display_handle_done(void *data, struct wl_output *wl_output)
{
}

static void
display_handle_scale(void *data, struct wl_output *wl_output, int32_t scale)
{
	struct output *output = data;

	output->scale = scale;
	output->display->window->needs_buffer_geometry_update = true;
}

static const struct wl_output_listener output_listener = {
	display_handle_geometry,
	display_handle_mode,
	display_handle_done,
	display_handle_scale,
};

static void
display_add_output(struct display *d, uint32_t name)
{
	struct output *output;

	output = znew(*output);
	output->display = d;
	output->scale = 1;
	output->wl_output =
		wl_registry_bind(d->registry, name, &wl_output_interface, 2);
	output->name = name;
	wl_list_insert(d->outputs.prev, &output->link);

	wl_output_add_listener(output->wl_output, &output_listener, output);
}

static void
display_destroy_output(struct display *d, struct output *output)
{
	destroy_window_output(d->window, output->wl_output);
	wl_output_destroy(output->wl_output);
	wl_list_remove(&output->link);
	free(output);
}

static void
display_destroy_outputs(struct display *d)
{
	struct output *tmp;
	struct output *output;

	wl_list_for_each_safe(output, tmp, &d->outputs, link)
		display_destroy_output(d, output);
}

static void
registry_handle_global(void *data, struct wl_registry *registry, uint32_t name,
	const char *interface, uint32_t version)
{
	struct display *d = data;

	if (strcmp(interface, wl_compositor_interface.name) == 0) {
		d->compositor = wl_registry_bind(registry, name,
			&wl_compositor_interface, MIN(version, 4));
	} else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
		d->wm_base = wl_registry_bind(registry, name,
			&xdg_wm_base_interface, 1);
		xdg_wm_base_add_listener(d->wm_base, &wm_base_listener, d);
	} else if (strcmp(interface, wl_seat_interface.name) == 0) {
		d->seat =
			wl_registry_bind(registry, name, &wl_seat_interface, 1);
		wl_seat_add_listener(d->seat, &seat_listener, d);
	} else if (strcmp(interface, wl_shm_interface.name) == 0) {
		d->shm = wl_registry_bind(registry, name, &wl_shm_interface, 1);
		d->cursor_theme = wl_cursor_theme_load(NULL, 32, d->shm);
		if (!d->cursor_theme) {
			fprintf(stderr, "unable to load default theme\n");
			return;
		}
		d->default_cursor =
			wl_cursor_theme_get_cursor(d->cursor_theme, "left_ptr");
		if (!d->default_cursor) {
			fprintf(stderr,
				"unable to load default left pointer\n");
			// TODO: abort ?
		}
	} else if (strcmp(interface, wl_output_interface.name) == 0
		&& version >= 2) {
		display_add_output(d, name);
	}
}

static void
registry_handle_global_remove(void *data, struct wl_registry *registry,
	uint32_t name)
{
	struct display *d = data;
	struct output *output;

	wl_list_for_each(output, &d->outputs, link) {
		if (output->name == name) {
			display_destroy_output(d, output);
			break;
		}
	}
}

static const struct wl_registry_listener registry_listener = {
	registry_handle_global,
	registry_handle_global_remove,
};

static void
signal_int(int signum)
{
	running = 0;
}

static void
usage(int error_code)
{
	fprintf(stderr,
		"Usage: simple-vulkan [OPTIONS]\n\n"
		"  -d <us>\tBuffer swap delay in microseconds\n"
		"  -p <presentation mode>\tSet presentation mode\n"
		"     immediate = 0\n"
		"     mailbox = 1\n"
		"     fifo = 2 (default)\n"
		"     fifo_relaxed = 3\n"
		"  -f\tRun in fullscreen mode\n"
		"  -r\tUse fixed width/height ratio when run in fullscreen "
		"mode\n"
		"  -m\tRun in maximized mode\n"
		"  -o\tCreate an opaque surface\n"
		"  -h\tThis help text\n\n");

	exit(error_code);
}

int
main(int argc, char **argv)
{
	struct sigaction sigint;
	struct display display = {};
	struct window window = {};
	int i, ret = 0;

	window.display = &display;
	display.window = &window;
	window.buffer_size.width = 250;
	window.buffer_size.height = 250;
	window.window_size = window.buffer_size;
	window.buffer_scale = 1;
	window.buffer_transform = WL_OUTPUT_TRANSFORM_NORMAL;
	window.needs_buffer_geometry_update = false;
	window.delay = 0;
	window.fullscreen_ratio = false;
	window.vk.present_mode = VK_PRESENT_MODE_FIFO_KHR;

	wl_list_init(&display.outputs);
	wl_list_init(&window.window_outputs);

	for (i = 1; i < argc; i++) {
		if (strcmp("-d", argv[i]) == 0 && i + 1 < argc) {
			window.delay = atoi(argv[++i]);
		} else if (strcmp("-p", argv[i]) == 0 && i + 1 < argc) {
			window.vk.present_mode = atoi(argv[++i]);
			assert(window.vk.present_mode >= 0
				&& window.vk.present_mode < 4);
		} else if (strcmp("-f", argv[i]) == 0) {
			window.fullscreen = 1;
		} else if (strcmp("-r", argv[i]) == 0) {
			window.fullscreen_ratio = true;
		} else if (strcmp("-m", argv[i]) == 0) {
			window.maximized = 1;
		} else if (strcmp("-o", argv[i]) == 0) {
			window.opaque = 1;
		} else if (strcmp("-h", argv[i]) == 0) {
			usage(EXIT_SUCCESS);
		} else {
			usage(EXIT_FAILURE);
		}
	}

	display.display = wl_display_connect(NULL);
	assert(display.display);

	display.registry = wl_display_get_registry(display.display);
	wl_registry_add_listener(display.registry, &registry_listener,
		&display);

	wl_display_roundtrip(display.display);

	if (!display.wm_base) {
		fprintf(stderr,
			"xdg-shell support required. simple-vulkan exiting\n");
		goto out_no_xdg_shell;
	}

	create_surface(&window);

	/* we already have wait_for_configure set after create_surface() */
	while (running && ret != -1 && window.wait_for_configure) {
		ret = wl_display_dispatch(display.display);

		/* wait until xdg_surface::configure acks the new dimensions */
		if (window.wait_for_configure) {
			continue;
		}

		init_vulkan(&window);
	}

	create_swapchain(&window);

	display.cursor_surface =
		wl_compositor_create_surface(display.compositor);

	sigint.sa_handler = signal_int;
	sigemptyset(&sigint.sa_mask);
	sigint.sa_flags = SA_RESETHAND;
	sigaction(SIGINT, &sigint, NULL);

	while (running && ret != -1) {
		ret = wl_display_dispatch_pending(display.display);
		redraw(&window);
	}

	destroy_surface(&window);
	destroy_swapchain(&window);
	fini_vulkan(&window);

	wl_surface_destroy(display.cursor_surface);
out_no_xdg_shell:
	display_destroy_outputs(&display);

	if (display.cursor_theme) {
		wl_cursor_theme_destroy(display.cursor_theme);
	}
	if (display.shm) {
		wl_shm_destroy(display.shm);
	}
	if (display.pointer) {
		wl_pointer_destroy(display.pointer);
	}
	if (display.keyboard) {
		wl_keyboard_destroy(display.keyboard);
	}
	if (display.touch) {
		wl_touch_destroy(display.touch);
	}
	if (display.seat) {
		wl_seat_destroy(display.seat);
	}
	if (display.wm_base) {
		xdg_wm_base_destroy(display.wm_base);
	}
	if (display.compositor) {
		wl_compositor_destroy(display.compositor);
	}

	wl_registry_destroy(display.registry);
	wl_display_flush(display.display);
	wl_display_disconnect(display.display);

	return 0;
}
