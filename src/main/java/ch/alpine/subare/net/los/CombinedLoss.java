package ch.alpine.subare.net.los;

import java.util.ArrayList;
import java.util.List;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public class CombinedLoss implements LossFunction {
  private final List<LossFunction> list;

  public CombinedLoss(List<LossFunction> list) {
    this.list = list;
  }

  @Override
  public NetLoss compute(Tensor prediction, Tensor target) {
    int index = 0;
    List<NetLoss> outs = new ArrayList<>(list.size());
    for (LossFunction lossFunction : list) {
      outs.add(lossFunction.compute( //
          prediction.get(index), //
          target.get(index)));
      ++index;
    }
    Scalar loss = outs.stream() //
        .map(NetLoss::loss) //
        .reduce(Scalar::add).orElse(RealScalar.ZERO);
    return new NetLoss(loss, null);
  }

  @Override
  public NetLoss computeBatch(Tensor prediction, Tensor target) {
    // TODO Auto-generated method stub
    return null;
  }
}
