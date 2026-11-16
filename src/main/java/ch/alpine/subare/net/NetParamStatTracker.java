// code by jph
package ch.alpine.subare.net;

import ch.alpine.subare.net.run.NetExtract;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.red.ScalarSummaryStatistics;

public class NetParamStatTracker implements NetTrainListener {
  public static final String[] LABELS = { "count", "min", "max", "mean", "stddev" };
  private final TableBuilder tparam = new TableBuilder();

  @Override
  public boolean epoch(NetTrain netTrain) {
    int epoch = netTrain.getTotalEpochs();
    ScalarSummaryStatistics scalarSummaryStatistics = new NetExtract(netTrain.netLayer).all().collect(ScalarSummaryStatistics.collector());
    tparam.appendRow(RealScalar.of(epoch), Tensors.of( //
        scalarSummaryStatistics.getMin(), //
        scalarSummaryStatistics.getMax(), //
        scalarSummaryStatistics.getMean(), //
        scalarSummaryStatistics.getStandardDeviation()));
    return true;
  }

  public TableBuilder getTableCopy() { // TODO misnomer
    return tparam;
  }
}
