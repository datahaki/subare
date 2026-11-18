// code by jph
package ch.alpine.subare.net;

import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.red.ScalarSummaryStatistics;

public class NetParamStatTracker implements NetTrainListener {
  public static final String[] LABELS = { "min", "max", "mean", "stddev" };
  private final TableBuilder tableBuilder = new TableBuilder();

  @Override
  public boolean epoch(NetTrain netTrain) {
    int epoch = netTrain.getTotalEpochs();
    ScalarSummaryStatistics scalarSummaryStatistics = new NetExtract(netTrain.netLayer()).all().collect(ScalarSummaryStatistics.collector());
    tableBuilder.appendRow(RealScalar.of(epoch), Tensors.of( //
        scalarSummaryStatistics.getMin(), //
        scalarSummaryStatistics.getMax(), //
        scalarSummaryStatistics.getMean(), //
        scalarSummaryStatistics.getStandardDeviation()));
    return true;
  }

  public TableBuilder tableBuilder() {
    return tableBuilder;
  }
}
