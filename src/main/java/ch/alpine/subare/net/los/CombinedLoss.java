// code by jph
package ch.alpine.subare.net.los;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public class CombinedLoss implements LossFunction, Serializable {
  private final List<LossFunction> list;

  public CombinedLoss(List<LossFunction> list) {
    this.list = list;
  }

  @Override
  public NetLoss compute(Tensor prediction, Tensor target) {
    int index = 0;
    List<NetLoss> netLoss = new ArrayList<>(list.size());
    for (LossFunction lossFunction : list) {
      netLoss.add(lossFunction.compute( //
          prediction.get(index), //
          target.get(index)));
      ++index;
    }
    return new NetLoss( //
        netLoss.stream() //
            .map(NetLoss::loss) //
            .reduce(Scalar::add) //
            .orElse(RealScalar.ZERO), //
        Tensor.of(netLoss.stream().map(NetLoss::gradOutput), netLoss.size()));
  }

  @Override
  public NetLoss computeBatch(Tensor prediction, Tensor target) {
    // TODO Auto-generated method stub
    return null;
  }
}
