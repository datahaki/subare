// code by jph
package ch.alpine.subare.net;

import java.util.List;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.ext.MergeIllegal;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetChain.html">NetChain</a> */
public class NetChain {
  public static NetChain of(Layer... layers) {
    return new NetChain(List.of(layers));
  }

  public static NetChain of(List<Layer> layers) {
    return new NetChain(layers.stream().toList());
  }

  // ---
  private final List<Layer> list;

  private NetChain(List<Layer> list) {
    this.list = list;
  }

  public Tensor forward(Tensor x) {
    return list.stream().reduce(x, Layer.forward(), MergeIllegal.operator());
  }

  public Tensor back(Tensor d) {
    return list.reversed().stream().reduce(d, Layer.back(), MergeIllegal.operator());
  }

  public void update() {
    list.forEach(Layer::update);
  }

  public Tensor parameters() {
    return Tensor.of(list.stream().map(Layer::parameters).flatMap(tensor -> Flatten.stream(tensor, -1)));
  }

  public Tensor error(Tensor tensor) {
    return list.getLast().error(tensor);
  }
}
