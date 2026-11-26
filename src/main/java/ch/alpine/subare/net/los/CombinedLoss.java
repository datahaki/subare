// code by jph
package ch.alpine.subare.net.los;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import ch.alpine.subare.net.los.NetLossCollectors.AccumulatedNetLoss;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.MathematicaFormat;

public class CombinedLoss implements LossFunction, Serializable {
  private final List<LossFunction> list;

  public CombinedLoss(List<LossFunction> list) {
    this.list = list;
  }

  @Override
  public NetLoss compute(Tensor prediction, Tensor target) {
    Integers.requireEquals(list.size(), prediction.length());
    int index = 0;
    List<NetLoss> netLoss = new ArrayList<>(list.size());
    for (LossFunction lossFunction : list) {
      netLoss.add(lossFunction.compute( //
          prediction.get(index), //
          target.get(index)));
      ++index;
    }
    AccumulatedNetLoss result = netLoss.stream().collect(NetLossCollectors.toAccumulated());
    // Tensor grad = Tensor.of(netLoss.stream().map(NetLoss::gradOutput), netLoss.size());
    // Tolerance.CHOP.requireClose(here, grad);
    return new NetLoss(result.loss(), result.compile());
  }

  @Override
  public NetLoss computeBatch(Tensor prediction, Tensor target) {
    Integers.requireEquals(list.size(), prediction.get(0).length());
    int index = 0;
    List<NetLoss> netLoss = new ArrayList<>(list.size());
    for (LossFunction lossFunction : list) {
      netLoss.add(lossFunction.computeBatch( //
          prediction.get(Tensor.ALL, index), //
          target.get(Tensor.ALL, index)));
      ++index;
    }
    AccumulatedNetLoss result = netLoss.stream().collect(NetLossCollectors.toAccumulated());
    // Tensor grad = Transpose.of(Tensor.of(netLoss.stream().map(NetLoss::gradOutput), netLoss.size()));
    // Tolerance.CHOP.requireClose(here, grad);
    return new NetLoss( //
        result.loss(), //
        result.compileBatch());
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("CombinedLoss", list);
  }
}
