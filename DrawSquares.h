#pragma once

#include "vulkan/vulkan.hpp"

#include "GpuInterface/Types/DeviceManager.h"


float vertices[]{-1.0f, -1.0f, 0.5f, -1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 0.5f, 1.0f, -1.0f, 0.5f};

u32 vertexIndices[] = {0, 1, 2, 0, 2, 3};


KE::VK::Buffer vertexBuffer;
KE::VK::Buffer indexBuffer;

void Draw(vk::CommandBuffer cmd, vk::Pipeline pipeline)
{

    auto swapchainOutput = KE::VK::ContextManager::GetSwapchain(0).GetSwapchainOutputHandle().GetResourceRef<KE::VK::SwapchainOutput>();

    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    barrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;

    barrier.oldLayout        = swapchainOutput.GetLayout();
    barrier.newLayout        = vk::ImageLayout::eColorAttachmentOptimal;
    barrier.image            = swapchainOutput.GetImage();
    barrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    vk::DependencyInfo depInfo{};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers    = &barrier;

    cmd.pipelineBarrier2(depInfo);

    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.imageView   = swapchainOutput.GetImageView();
    colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAttachment.loadOp      = vk::AttachmentLoadOp::eNone;
    colorAttachment.storeOp     = vk::AttachmentStoreOp::eStore;
    colorAttachment.clearValue  = vk::ClearColorValue{0.1f, 0.1f, 0.1f, 1.0f};

    vk::RenderingInfo renderingInfo{};
    renderingInfo.renderArea           = vk::Rect2D{{0, 0}, swapchainOutput.GetExtent()};
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;


    cmd.beginRendering(renderingInfo);

    vk::Viewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = swapchainOutput.GetExtent().width;
    viewport.height   = swapchainOutput.GetExtent().height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    cmd.setViewport(0, viewport);

    vk::Rect2D scissor{{0, 0}, swapchainOutput.GetExtent()};
    cmd.setScissor(0, scissor);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    std::array<vk::DeviceSize, 1> offsets = {0};

    std::array<vk::Buffer, 2> buffers = {vertexBuffer.GetVkBuffer()};

    std::array<vk::VertexInputBindingDescription2EXT, 1> bindings;

    bindings[0].binding = 0;
    bindings[0].stride  = 12;
    bindings[0].divisor = 1;

    std::array<vk::VertexInputAttributeDescription2EXT, 1> attributeDescs;

    attributeDescs[0].offset = 0;
    attributeDescs[0].format = vk::Format::eR32G32B32Sfloat;
    attributeDescs[0].location = 0;


    vk::Buffer vertBuff = vertexBuffer.GetVkBuffer();

    cmd.setVertexInputEXT(bindings, attributeDescs);

    cmd.bindVertexBuffers(0, 1, &vertBuff, offsets.data());
    cmd.bindIndexBuffer(indexBuffer.GetVkBuffer(), 0, vk::IndexType::eUint32);

    cmd.bindResourceHeapEXT(KE::VK::ContextManager::GetDevice(0).GetResourceHeapBindInfoPtr());
    cmd.bindSamplerHeapEXT(KE::VK::ContextManager::GetDevice(0).GetSamplerHeapBindInfoPtr());


    cmd.drawIndexed(6, 1, 0, 0, 0);

    cmd.endRendering();
}