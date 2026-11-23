// code by jph
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.subare.net.NetOutput;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.mat.MatrixQ;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.d.BernoulliDistribution;

class ScalarLayerTest {
  @Test
  void test() {
    NetLayer netLayer = NetChains.binary(NetChains.linear(ElementwiseLayers.GELU, 2, 5, 1));
    NetLayers.initialize(netLayer);
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    assertInstanceOf(Scalar.class, tensor);
  }

  @Test
  void testVector() {
    NetLayer netLayer = NetChains.binary(NetChains.linear(ElementwiseLayers.GELU, 2, 8, 1));
    NetLayers.initialize(netLayer);
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    assertInstanceOf(Scalar.class, tensor);
    NetOutput netOutput = NetLayers.output(netLayer);
    NetLoss netError = netOutput.error(RealScalar.ONE);
    Tensor gradOutput = netError.gradOutput();
    IO.println(gradOutput);
    // assertInstanceOf(Scalar.class, gradOutput);
  }

  @ParameterizedTest
  @ValueSource(ints = { 2, 3, 5, 10 })
  void testMatrix(int batch_size) {
    NetLayer netLayer = NetChains.binary(NetChains.linear(ElementwiseLayers.GELU, 2, 8, 1));
    NetLayers.initialize(netLayer);
    Tensor X = RandomVariate.of(NormalDistribution.standard(), batch_size, 2);
    Tensor Y = netLayer.forwardBatch(X);
    VectorQ.requireLength(Y, batch_size);
    Tensor T = RandomVariate.of(BernoulliDistribution.of(.3), batch_size);
    NetOutput netOutput = NetLayers.output(netLayer);
    NetLoss netError = netOutput.errorBatch(T);
    Tensor gradOutput = netError.gradOutput();
    MatrixQ.requireSize(gradOutput, batch_size, 1);
  }
}
