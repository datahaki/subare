// code by jph
package ch.alpine.subare.net.opt;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import ch.alpine.subare.net.lay.MatrixLayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class NetOptimizersTest {
  @ParameterizedTest
  @EnumSource
  void test(NetOptimizers netOptimizers) {
    MatrixLayer matrixLayer = new MatrixLayer(1, 2);
    Tensor y = matrixLayer.forward(Tensors.vector(1));
    matrixLayer.back(y);
    NetOptimizer netOptimizer = netOptimizers.create(matrixLayer, RealScalar.of(0.003), RealScalar.of(0.9), RealScalar.of(0.99));
    netOptimizer.update();
  }
}
