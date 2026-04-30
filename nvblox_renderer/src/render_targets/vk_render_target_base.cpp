/*
Copyright 2026 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "nvblox/renderer/render_targets/vk_render_target_base.h"

#include <array>

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"
#include "nvblox/renderer/core/vk_utils.h"

namespace nvblox {
namespace renderer {

void VkRenderTargetBase::initBase(VkDevice device,
                                  VkPhysicalDevice physical_device) {
  device_ = device;
  physical_device_ = physical_device;
}

VkFramebuffer VkRenderTargetBase::framebuffer(uint32_t index) const {
  if (framebuffers_.empty()) {
    LOG(ERROR) << "No framebuffers available, index: " << index;
    return VK_NULL_HANDLE;
  }
  if (index >= framebuffers_.size()) {
    LOG(ERROR) << "Invalid framebuffer index: " << index
               << " (max: " << framebuffers_.size() - 1 << ")";
    return VK_NULL_HANDLE;
  }
  return framebuffers_[index];
}

bool VkRenderTargetBase::createDepthResources() {
  // Use helper to create depth image with memory and view
  Image2DCreateInfo create_info;
  create_info.width = extent_.width;
  create_info.height = extent_.height;
  create_info.format = depth_format_;
  create_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (needsDepthReadback()) {
    create_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }
  create_info.memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  Image2DResult result;
  if (!createImage2D(device_, physical_device_, create_info,
                     VK_IMAGE_ASPECT_DEPTH_BIT, &result)) {
    LOG(ERROR) << "Failed to create depth resources";
    return false;
  }

  depth_image_ = result.image;
  depth_memory_ = result.memory;
  depth_view_ = result.view;
  return true;
}

bool VkRenderTargetBase::createRenderPass() {
  VkAttachmentDescription color_attachment{};
  color_attachment.format = color_format_;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = colorFinalLayout();

  VkAttachmentDescription depth_attachment{};
  depth_attachment.format = depth_format_;
  depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp = needsDepthReadback()
                                 ? VK_ATTACHMENT_STORE_OP_STORE
                                 : VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_attachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkAttachmentReference depth_ref{
      1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_ref;
  subpass.pDepthStencilAttachment = &depth_ref;

  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  std::array<VkAttachmentDescription, 2> attachments = {color_attachment,
                                                        depth_attachment};
  VkRenderPassCreateInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
  render_pass_info.pAttachments = attachments.data();
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  checkVkErrors(
      vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_));
  return true;
}

bool VkRenderTargetBase::createFramebuffers() {
  const auto& views = imageViews();
  framebuffers_.resize(views.size());

  for (size_t i = 0; i < views.size(); ++i) {
    std::array<VkImageView, 2> attachments = {views[i], depth_view_};

    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = render_pass_;
    fb_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    fb_info.pAttachments = attachments.data();
    fb_info.width = extent_.width;
    fb_info.height = extent_.height;
    fb_info.layers = 1;

    checkVkErrors(
        vkCreateFramebuffer(device_, &fb_info, nullptr, &framebuffers_[i]));
  }

  return true;
}

void VkRenderTargetBase::destroyDepthResources() {
  if (depth_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_, depth_view_, nullptr);
    depth_view_ = VK_NULL_HANDLE;
  }
  if (depth_image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device_, depth_image_, nullptr);
    depth_image_ = VK_NULL_HANDLE;
  }
  if (depth_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, depth_memory_, nullptr);
    depth_memory_ = VK_NULL_HANDLE;
  }
}

void VkRenderTargetBase::destroyFramebuffers() {
  for (auto fb : framebuffers_) {
    if (fb != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(device_, fb, nullptr);
    }
  }
  framebuffers_.clear();
}

void VkRenderTargetBase::destroyRenderPass() {
  if (render_pass_ != VK_NULL_HANDLE) {
    vkDestroyRenderPass(device_, render_pass_, nullptr);
    render_pass_ = VK_NULL_HANDLE;
  }
}

}  // namespace renderer
}  // namespace nvblox
