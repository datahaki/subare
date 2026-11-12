// code by jph
package ch.alpine.subare.net;

import ch.alpine.subare.net.run.NetExtract;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.TableBuilder;

public class NetParamTracker implements NetTrainListener {
  public final TableBuilder tparam = new TableBuilder();

  @Override
  public boolean epoch(NetTrain netTrain, int epoch) {
    tparam.appendRow(RealScalar.of(epoch), Tensor.of(new NetExtract(netTrain.netLayer).all()));
    return true;
  }
}
