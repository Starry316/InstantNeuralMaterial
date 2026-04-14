from falcor import *

def render_graph_NeuralMatRendering():
    g = RenderGraph("NeuralMatRendering")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    NeuralMatRendering = createPass("NeuralMatRendering", {'maxBounces': 3})
    g.addPass(NeuralMatRendering, "NeuralMatRendering")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("NeuralMatRendering.color", "AccumulatePass.input")
    g.markOutput("ToneMapper.dst")
    return g

NeuralMatRendering = render_graph_NeuralMatRendering()
try: m.addGraph(NeuralMatRendering)
except NameError: None
