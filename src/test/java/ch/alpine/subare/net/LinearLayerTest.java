package ch.alpine.subare.net;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.jet.JetScalar;
import ch.alpine.tensor.mat.DiagonalMatrix;

class LinearLayerTest {
  @Test
  void test() {
    LinearLayer linearLayer = new LinearLayer();
    linearLayer.W = DiagonalMatrix.full(Tensors.vector(3));
    linearLayer.b = Tensors.vector(5);
    JetScalar js = JetScalar.of(RealScalar.of(4), 2);
    IO.println(js);
    Tensor x = Tensors.of(js);
    Tensor y = linearLayer.forward(x);
    Tensor d = linearLayer.back(y);
    IO.println(d);
  }
}
