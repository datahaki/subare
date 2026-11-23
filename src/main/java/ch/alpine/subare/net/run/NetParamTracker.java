// code by jph
package ch.alpine.subare.net.run;

import ch.alpine.subare.net.NetLayers;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.TableBuilder;

/** ONLY FOR TESTING ON SMALL NETWORKS */
public class NetParamTracker implements NetTrainListener {
  public final TableBuilder tparam = new TableBuilder();

  @Override
  public boolean epoch(NetTrain netTrain) {
    int epoch = netTrain.getTotalEpochs();
    tparam.appendRow(RealScalar.of(epoch), Tensor.of(NetLayers.parameters(netTrain.netModel().netLayer())));
    return true;
  }
}
