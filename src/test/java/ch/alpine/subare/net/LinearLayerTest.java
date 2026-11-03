// code by jph
package ch.alpine.subare.net;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Drop;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.jet.JetScalar;
import ch.alpine.tensor.mat.DiagonalMatrix;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.qty.Quantity;

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
    IO.println(y);
    Tensor d = linearLayer.back(y);
    IO.println(d);
  }

  @ParameterizedTest
  @ValueSource(ints = { 5, 10, 20 })
  void testTraining(int d_out) {
    Integer d_in = 10;
    Integer samples = 30;
    Scalar learningRate = RealScalar.of(0.05);
    Distribution distribution = NormalDistribution.standard();
    TensorUnaryOperator tuo;
    {
      LinearLayer linearLayer = new LinearLayer();
      linearLayer.W = RandomVariate.of(distribution, d_out, d_in);
      linearLayer.b = RandomVariate.of(distribution, d_out);
      NetChain netChain = NetChain.of(linearLayer);
      tuo = netChain::forward;
    }
    Tensor xdata = RandomVariate.of(distribution, samples, d_in);
    Tensor ydata = tuo.slash(xdata);
    // ---
    NetChain netChain = NetChain.of(LinearLayer.xavier(d_out, d_in));
    NetTrain netTrain = new NetTrain(netChain, xdata, ydata);
    netTrain.run(learningRate, Quantity.of(1, "s"), 100, 1);
    TableBuilder tloss = netTrain.tloss;
    int n = tloss.getRowCount();
    Tensor row = tloss.getRow(n - 1);
    Tensor loss = Drop.head(row, 1);
    Tolerance.CHOP.requireAllZero(loss);
  }
}
