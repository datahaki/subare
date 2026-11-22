// code by jph
package ch.alpine.subare.net.lay;

import java.util.List;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.ArrayReshape;

// TODO UNFINISHED UNTESTED WORK IN PROGRESS !!!
class ReshapeLayer implements NetLayer {
  private transient List<Integer> list;

  @Override
  public Tensor forward(Tensor x) {
    return ArrayReshape.of(x, list);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    // TODO Auto-generated method stub
    return null;
  }
}
