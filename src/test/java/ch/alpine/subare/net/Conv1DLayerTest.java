// code by jph
package ch.alpine.subare.net;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.RepetitionInfo;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.io.Primitives;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.UniformDistribution;

class Conv1DLayerTest {
  @RepeatedTest(4)
  void test(RepetitionInfo repetitionInfo) {
    int k = repetitionInfo.getCurrentRepetition();
    Conv1D conv1d = new Conv1D(k);
    Tensor vweights = RandomVariate.of(UniformDistribution.unit(), k);
    conv1d.weights = Primitives.toDoubleArray(vweights);
    conv1d.bias = 2;
    int n = 20;
    Tensor vx = RandomVariate.of(UniformDistribution.unit(), n);
    double[] x = Primitives.toDoubleArray(vx);
    Tensor forward1 = Tensors.vectorDouble(conv1d.forward(x));
    // ---
    Conv1DLayer conv1dLayer = new Conv1DLayer();
    conv1dLayer.w = Tensors.vectorDouble(conv1d.weights);
    conv1dLayer.b = RealScalar.of(conv1d.bias);
    Tensor forward2 = conv1dLayer.forward(Tensors.vectorDouble(x));
    Tolerance.CHOP.requireClose(forward1, forward2);
    // ---
    Tensor gOut = RandomVariate.of(UniformDistribution.unit(), n - k + 1);
    double[] gradOut = Primitives.toDoubleArray(gOut);
    assertEquals(forward1.length(), gradOut.length);
    double[] backward = conv1d.backward(gradOut, 1);
    Tensor backward1 = Tensors.vectorDouble(backward);
    Tensor back = conv1dLayer.back(gOut);
    // ---
    assertEquals(RealScalar.of(conv1d.biasGradient), conv1dLayer.gb);
    Tolerance.CHOP.requireClose(backward1, back);
    Tensor wG1 = Tensors.vectorDouble(conv1d.weightsGradient);
    Tolerance.CHOP.requireClose(wG1, conv1dLayer.gW);
  }
}
