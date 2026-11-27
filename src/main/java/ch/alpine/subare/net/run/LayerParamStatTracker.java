// code by jph
package ch.alpine.subare.net.run;

import java.util.LinkedList;
import java.util.List;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.red.ScalarSummaryStatistics;

public class LayerParamStatTracker implements NetTrainListener {
  public static final String[] LABELS = { "min", "max", "mean", "stddev" };
  private final List<ScalarSummaryStatistics[]> list = new LinkedList<>();

  @Override
  public boolean epoch(NetTrain netTrain) {
    NetLayer netLayer = netTrain.netModel().netLayer();
    ScalarSummaryStatistics[] array = //
        NetLayers.parameters(netLayer) //
            .map(Flatten::scalars) //
            .map(s -> s.collect(ScalarSummaryStatistics.collector())).toArray(ScalarSummaryStatistics[]::new);
    list.add(array);
    return true;
  }
  // public TableBuilder tableBuilder() {
  // TableBuilder tableBuilder = new TableBuilder();
  // Int myI = new Int();
  // list.stream() //
  // .forEach(scalarSummaryStatistics -> tableBuilder.appendRow( //
  // RealScalar.of(myI.getAndIncrement()), //
  // Tensors.of( //
  // scalarSummaryStatistics.getMin(), //
  // scalarSummaryStatistics.getMax(), //
  // scalarSummaryStatistics.getMean(), //
  // scalarSummaryStatistics.getStandardDeviation())));
  // return tableBuilder;
  // }
  // public TableBuilder tableBuilderForLogPlot() {
  // TableBuilder tableBuilder = new TableBuilder();
  // Int myI = new Int();
  // list.stream() //
  // .forEach(scalarSummaryStatistics -> tableBuilder.appendRow( //
  // RealScalar.of(myI.getAndIncrement()), //
  // Tensors.of( //
  // Max.of( //
  // Abs.FUNCTION.apply(scalarSummaryStatistics.getMin()), //
  // Abs.FUNCTION.apply(scalarSummaryStatistics.getMax())), //
  // Abs.FUNCTION.apply(scalarSummaryStatistics.getMean()), //
  // scalarSummaryStatistics.getStandardDeviation())));
  // return tableBuilder;
  // }
}
