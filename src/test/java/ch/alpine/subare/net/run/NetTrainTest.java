// code by jph
package ch.alpine.subare.net.run;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.ThreadLocalRandom;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.api.NetOptimizer;
import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.alg.Subdivide;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.ply.Polynomial;

class NetTrainTest {
  @Test
  void test() {
    LinearLayer linearLayer = new LinearLayer(1, 1);
    Polynomial polynomial = Polynomial.of(Tensors.vector(1, 0.5));
    Tensor xdata = Range.of(-3, 3 + 1).maps(Tensors::of);
    Tensor ydata = xdata.maps(polynomial);
    linearLayer.initialize(ThreadLocalRandom.current());
    NetOptimizer netOptimize = NetOptimizers.SIMPLE.create(linearLayer, RealScalar.of(0.02));
    NetTrain netTrain = new NetTrain(linearLayer, netOptimize);
    TrainingData trainingData = new TrainingData(xdata, ydata);
    netTrain.setMaxEpochs(1000);
    netTrain.runAsBatches(trainingData, 3);
    Tensor affineTuple = linearLayer.parameters();
    Chop._10.requireClose(affineTuple.get(0), Tensors.fromString("{{0.5}}"));
    Chop._10.requireClose(affineTuple.get(1), Tensors.fromString("{1}"));
    Scalar log_loss = netTrain.log_loss();
    assertTrue(Scalars.lessThan(log_loss, RealScalar.of(-20)));
    assertEquals(netTrain.tloss.getRowCount(), 1000);
  }

  @Test
  void testMassive() {
    LinearLayer linearLayer = new LinearLayer(1, 1);
    Polynomial polynomial = Polynomial.of(Tensors.vector(1, 0.5));
    Tensor xdata = Subdivide.of(-3, 3, 100).maps(Tensors::of);
    Tensor ydata = xdata.maps(polynomial);
    linearLayer.initialize(ThreadLocalRandom.current());
    NetOptimizer netOptimize = NetOptimizers.SIMPLE.create(linearLayer, RealScalar.of(0.02));
    NetTrain netTrain = new NetTrain(linearLayer, netOptimize);
    TrainingData trainingData = new TrainingData(xdata, ydata);
    netTrain.setMaxEpochs(1000);
    netTrain.runAsBatches(trainingData, 10);
    Tensor affineTuple = linearLayer.parameters();
    Chop._10.requireClose(affineTuple.get(0), Tensors.fromString("{{0.5}}"));
    Chop._10.requireClose(affineTuple.get(1), Tensors.fromString("{1}"));
    Scalar log_loss = netTrain.log_loss();
    assertTrue(Scalars.lessThan(log_loss, RealScalar.of(-20)));
  }
}
