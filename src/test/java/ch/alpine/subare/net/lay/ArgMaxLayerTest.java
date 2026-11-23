// code by jph
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.NetModels;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.chq.ExactTensorQ;
import ch.alpine.tensor.mat.HilbertMatrix;
import ch.alpine.tensor.mat.MatrixQ;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.d.BernoulliDistribution;

class ArgMaxLayerTest {
  @ParameterizedTest
  @ValueSource(ints = { 2, 3, 5, 10 })
  void testVector(int labels) {
    NetModel netModel = NetModels.argMax(NetChains.linear(ElementwiseLayers.GELU, 2, 8, labels));
    NetLayer netLayer = netModel.netLayer();
    LossFunction lossFunction = netModel.lossFunction();
    NetLayers.initialize(netLayer);
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    assertInstanceOf(Scalar.class, tensor);
    ExactTensorQ.require(tensor);
    NetLoss netError = lossFunction.compute(tensor, RealScalar.ONE);
    Tensor gradOutput = netError.gradOutput();
    VectorQ.requireLength(gradOutput, labels);
  }

  @ParameterizedTest
  @ValueSource(ints = { 2, 3, 5, 10 })
  void testMatrix(int labels) {
    NetModel netModel = NetModels.argMax(NetChains.linear(ElementwiseLayers.GELU, 2, 8, labels));
    NetLayer netLayer = netModel.netLayer();
    LossFunction lossFunction = netModel.lossFunction();
    NetLayers.initialize(netLayer);
    int batch_size = 5;
    Tensor X = RandomVariate.of(NormalDistribution.standard(), batch_size, 2);
    Tensor Y = netLayer.forwardBatch(X);
    VectorQ.requireLength(Y, batch_size);
    ExactTensorQ.require(Y);
    Tensor T = RandomVariate.of(BernoulliDistribution.of(.3), batch_size);
    NetLoss netError = lossFunction.computeBatch(Y, T);
    Tensor gradOutput = netError.gradOutput();
    MatrixQ.requireSize(gradOutput, batch_size, labels);
  }

  @Test
  void testNonVector() {
    ArgMaxLayer argMaxLayer = new ArgMaxLayer();
    assertThrows(Exception.class, () -> argMaxLayer.forward(HilbertMatrix.of(2, 3)));
  }
}
