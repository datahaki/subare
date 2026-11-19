// code by jph
package ch.alpine.subare.net;

import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.io.TableBuilder;

/** ONLY FOR TESTING ON SMALL NETWORKS */
public class NetParamTracker implements NetTrainListener {
  public final TableBuilder tparam = new TableBuilder();

  @Override
  public boolean epoch(NetTrain netTrain) {
    int epoch = netTrain.getTotalEpochs();
    tparam.appendRow(RealScalar.of(epoch), netTrain.netLayer().parameters());
    return true;
  }
}
