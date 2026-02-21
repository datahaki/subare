// code by jph
package ch.alpine.subare.net;

import java.util.List;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.ext.MergeIllegal;
import ch.alpine.tensor.sca.Sign;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetChain.html">NetChain</a> */
public record NetChain(List<Layer> list) {
  public static NetChain of(Layer... layers) {
    return new NetChain(List.of(layers));
  }

  public static NetChain of(List<Layer> layers) {
    return new NetChain(layers.stream().toList());
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

  public void setL2(Scalar l2) {
    Sign.requirePositiveOrZero(l2);
    list.stream() //
        .filter(linearLayer -> linearLayer instanceof LinearLayer) //
        .map(LinearLayer.class::cast) //
        .forEach(linearLayer -> linearLayer.l2 = l2);
  }
}
