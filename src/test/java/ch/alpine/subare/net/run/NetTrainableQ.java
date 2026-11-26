// code by jph
package ch.alpine.subare.net.run;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

public enum NetTrainableQ {
  ;
  public static void check(NetLayer netLayer, int d_in) {
    int samples = 100;
    int batch_size = 20;
    Tensor X = RandomVariate.of(NormalDistribution.standard(), samples, d_in);
    NetLayers.initialize(netLayer);
    Tensor Y = netLayer.forwardBatch(X);
    NetLayers.initialize(netLayer);
    NetOptimizer netOptimize = NetOptimizers.SIMPLE.with(RealScalar.of(0.1)).apply(netLayer);
    // NetTrain netTrain = new NetTrain(netLayer, netOptimize);
    // TrainingData trainingData = new TrainingData(X, Y);
    // netTrain.setMaxEpochs(1000);
    // netTrain.runAsBatches(trainingData, batch_size);
    // Scalar loss = netTrain.log_loss();
    // boolean lessThan = Scalars.lessThan(loss, RealScalar.of(-3));
    // assertTrue(lessThan);
  }
}
