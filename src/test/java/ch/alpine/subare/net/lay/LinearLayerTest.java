// code by jph
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.err.NetLoss;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetSimpleGradientDescent;
import ch.alpine.subare.net.run.NetInitialize;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.jet.JetScalar;
import ch.alpine.tensor.mat.DiagonalMatrix;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.sca.exp.Exp;

class LinearLayerTest {
  @Test
  void test() {
    LinearLayer linearLayer = new LinearLayer(1, 1);
    linearLayer.W = DiagonalMatrix.full(Tensors.vector(3));
    linearLayer.b = Tensors.vector(5);
    JetScalar js = JetScalar.of(RealScalar.of(4), 2);
    // IO.println(js);
    Tensor x = Tensors.of(js);
    Tensor y = linearLayer.forward(x);
    // IO.println(y);
    Tensor d = linearLayer.back(y);
    // IO.println(d);
  }

  @ParameterizedTest
  @ValueSource(ints = { 5, 10, 20 })
  void testTrainingBatch(int d_out) {
    Integer d_in = 3;
    Integer samples = 30;
    Scalar learningRate = RealScalar.of(0.02);
    Distribution distribution = NormalDistribution.standard();
    TensorUnaryOperator tuo;
    NetLayer netLayer = NetChain.of(new LinearLayer(d_in, d_out));
    NetInitialize.of(netLayer);
    tuo = netLayer::forward;
    Tensor xdata = RandomVariate.of(distribution, samples, d_in);
    Tensor ydata = tuo.slash(xdata);
    TrainingData trainingData = new TrainingData(xdata, ydata);
    // ---
    NetInitialize.of(netLayer);
    NetOptimizer netOptimize = new NetSimpleGradientDescent(netLayer, learningRate);
    NetTrain netTrain = new NetTrain(netLayer, netOptimize);
    netTrain.runAsBatches(trainingData, 2, 100, Quantity.of(1, "s"));
    TableBuilder tloss = netTrain.tloss;
    Tolerance.CHOP.requireAllZero(Exp.FUNCTION.apply(netTrain.log_loss()));
  }

  @Test
  void testGrad11() {
    LinearLayer linearLayer = new LinearLayer(1, 1);
    linearLayer.initialize(ThreadLocalRandom.current());
    Tensor f = linearLayer.forward(Tensors.vector(1));
    NetLoss netError = linearLayer.error(f);
    Tensor grad = netError.gradOutput();
    Tensor back = linearLayer.back(grad);
    assertEquals(Dimensions.of(back), List.of(1));
  }

  @Test
  void testGrad23() {
    LinearLayer linearLayer = new LinearLayer(2, 3);
    linearLayer.initialize(ThreadLocalRandom.current());
    Tensor y = linearLayer.forward(Tensors.vector(0.1, 0.2));
    assertEquals(Dimensions.of(y), List.of(3));
    NetLoss netError = linearLayer.error(Tensors.vector(0.2, -0.3, 0.4));
    Tensor grad = netError.gradOutput();
    assertEquals(Dimensions.of(grad), List.of(3));
    Tensor back = linearLayer.back(grad);
    assertEquals(Dimensions.of(back), List.of(2));
  }
}
