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

  public Tensor forward(Tensor x0) {
    return list.stream().reduce(x0, (x, layer) -> layer.forward(x), MergeIllegal.operator());
  }

  public Tensor back(Tensor grad0) {
    return list.reversed().stream().reduce(grad0, (grad, layer) -> layer.back(grad), MergeIllegal.operator());
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
