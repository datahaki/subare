// code by jph
package ch.alpine.subare.net;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.RepetitionInfo;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.io.Primitives;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.UniformDistribution;

class MaxPooling1DLayerTest {
  @RepeatedTest(3)
  void test(RepetitionInfo repetitionInfo) {
    int k = repetitionInfo.getCurrentRepetition();
    int n = 5;
    MaxPooling1D maxPooling1D = new MaxPooling1D(k);
    Tensor vinput = RandomVariate.of(UniformDistribution.unit(), k * n);
    double[] input = Primitives.toDoubleArray(vinput);
    // new double[] { 1, 2, 10, 9, -3, -2 };
    maxPooling1D.forward(input);
    Tensor maxInd1 = Tensors.vectorInt(maxPooling1D.lastMaxIndices);
    MaxPooling1DLayer mp1dl = new MaxPooling1DLayer(k);
    // Tensor forward =
    mp1dl.forward(Tensors.vectorDouble(input));
    Tensor maxInd2 = Tensors.vectorInt(mp1dl.lastMaxIndices);
    assertEquals(maxInd1, maxInd2);
    // ---
    Tensor vgrdO = RandomVariate.of(UniformDistribution.unit(), n);
    double[] gradOut = Primitives.toDoubleArray(vgrdO);
    double[] backward = maxPooling1D.backward(gradOut, input.length);
    Tensor back = mp1dl.back(Tensors.vectorDouble(gradOut));
    Tolerance.CHOP.requireClose(Tensors.vectorDouble(backward), back);
  }
}
