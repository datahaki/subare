// code by jph
package ch.alpine.subare.net.run;

import static org.junit.jupiter.api.Assertions.assertTrue;

import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

public enum NetTrainableQ {
  ;
  public static void check(NetLayer netLayer, int d_in) {
    int samples = 100;
    int batch_size = 20;
    Tensor X = RandomVariate.of(NormalDistribution.standard(), samples, d_in);
    NetInitialize.of(netLayer);
    Tensor Y = netLayer.forwardBatch(X);
    NetInitialize.of(netLayer);
    NetOptimizer netOptimize = NetOptimizers.SIMPLE.create(netLayer, RealScalar.of(0.1));
    NetTrain netTrain = new NetTrain(netLayer, netOptimize);
    TrainingData trainingData = new TrainingData(X, Y);
    netTrain.setMaxEpochs(1000);
    netTrain.runAsBatches(trainingData, batch_size);
    Scalar loss = netTrain.log_loss();
    boolean lessThan = Scalars.lessThan(loss, RealScalar.of(-3));
    assertTrue(lessThan);
  }
}
