// code by jph
package ch.alpine.subare.net;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.mat.MatrixQ;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.d.BernoulliDistribution;

class BinaryLayerTest {
  @Test
  void test() {
    NetLayer netLayer = NetChains.binary(NetChains.linear(() -> ElementwiseLayer.gelu(), 2, 5, 1));
    NetInitialize.of(netLayer);
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    VectorQ.requireLength(tensor, 1);
  }

  @Test
  void testVector() {
    NetLayer netLayer = NetChains.binary(NetChains.linear(() -> ElementwiseLayer.gelu(), 2, 8, 1));
    NetInitialize.of(netLayer);
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    VectorQ.requireLength(tensor, 1);
    NetError netError = netLayer.error(Tensors.vector(1));
    Tensor gradOutput = netError.gradOutput();
    VectorQ.requireLength(gradOutput, 1);
  }

  @ParameterizedTest
  @ValueSource(ints = { 2, 3, 5, 10 })
  void testMatrix(int batch_size) {
    NetLayer netLayer = NetChains.binary(NetChains.linear(() -> ElementwiseLayer.gelu(), 2, 8, 1));
    NetInitialize.of(netLayer);
    Tensor X = RandomVariate.of(NormalDistribution.standard(), batch_size, 2);
    Tensor Y = netLayer.forward(X);
    MatrixQ.requireSize(Y, batch_size, 1);
    Tensor T = RandomVariate.of(BernoulliDistribution.of(.3), batch_size, 1);
    NetError netError = netLayer.error(T);
    Tensor gradOutput = netError.gradOutput();
    MatrixQ.requireSize(gradOutput, batch_size, 1);
  }
}
