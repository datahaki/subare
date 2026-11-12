// code by jph
package ch.alpine.subare.net.run;

import static org.junit.jupiter.api.Assertions.assertTrue;

import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetSimpleGradientDescent;
import ch.alpine.subare.net.run.NetInitialize;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.qty.Quantity;

public enum NetTrainableQ {
  ;
  public static void check(NetLayer netLayer, int d_in) {
    int samples = 100;
    int batch_size = 20;
    Tensor X = RandomVariate.of(NormalDistribution.standard(), samples, d_in);
    NetInitialize.of(netLayer);
    Tensor Y = netLayer.forward(X);
    NetInitialize.of(netLayer);
    NetOptimizer netOptimize = new NetSimpleGradientDescent(netLayer,RealScalar.of(0.1));
    NetTrain netTrain = new NetTrain(netLayer, netOptimize);
    TrainingData trainingData = new TrainingData(X, Y);
    netTrain.runAsBatches(trainingData, batch_size, 1000, Quantity.of(2, "s"));
    Scalar loss = netTrain.log_loss();
    boolean lessThan = Scalars.lessThan(loss, RealScalar.of(-3));
    assertTrue(lessThan);
  }
}
