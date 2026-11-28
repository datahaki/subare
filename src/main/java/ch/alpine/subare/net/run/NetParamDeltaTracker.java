// code by jph
package ch.alpine.subare.net.run;

import java.util.Objects;
import java.util.Optional;

import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.BoundedLinkedList;

public class NetParamDeltaTracker implements NetTrainListener {
  private Tensor prev = null;
  private BoundedLinkedList<Tensor> deltas = new BoundedLinkedList<>(2);

  @Override
  public synchronized boolean epoch(NetTrain netTrain) {
    Tensor curr = Tensor.of(NetLayers.parameters(netTrain.netModel().netLayer()));
    if (Objects.nonNull(prev))
      deltas.add(curr.subtract(prev));
    prev = curr;
    return true;
  }

  public synchronized Optional<Tensor> getDelta() {
    return Optional.ofNullable(deltas.peekLast());
  }

  public synchronized Optional<Tensor> getDelta2nd() {
    return 1 < deltas.size() //
        ? Optional.ofNullable(deltas.get(1).subtract(deltas.get(0)))
        : Optional.empty();
  }
}
