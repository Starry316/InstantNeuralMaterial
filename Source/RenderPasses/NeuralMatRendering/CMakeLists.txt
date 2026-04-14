add_plugin(NeuralMatRendering)


target_include_directories(NeuralMatRendering PUBLIC  ${CUDA_INCLUDE_DIRS})
target_link_libraries(NeuralMatRendering PUBLIC  ${CUDA_LIBRARIES} )

target_sources(NeuralMatRendering PRIVATE
    NeuralMatRendering.cpp
    NeuralMatRendering.h
    MinimalPathTracer.rt.slang
    Tools/Utils.cpp
    Tools/Utils.h
)
set_target_properties(NeuralMatRendering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_copy_shaders(NeuralMatRendering RenderPasses/NeuralMatRendering)
target_source_group(NeuralMatRendering "RenderPasses")
