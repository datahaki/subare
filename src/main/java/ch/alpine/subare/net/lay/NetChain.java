// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Stream;

import ch.alpine.subare.net.err.NetLoss;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.MergeIllegal;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetChain.html">NetChain</a> */
public record NetChain(List<NetLayer> list) implements NetLayer, Serializable {
  public static NetLayer of(NetLayer... layers) {
    return new NetChain(List.of(layers));
  }

  @Override
  public Tensor forward(Tensor x0) {
    return list.stream() //
        .reduce(x0, (x, layer) -> layer.forward(x), MergeIllegal.operator());
  }

  @Override
  public Tensor back(Tensor grad0) {
    return list.reversed().stream() //
        .reduce(grad0, (grad, layer) -> layer.back(grad), MergeIllegal.operator());
  }

  @Override
  public NetLoss error(Tensor target) {
    return list.getLast().error(target);
  }
  // // TODO probably obsolete
  // public void setL2(Scalar l2) {
  // Sign.requirePositiveOrZero(l2);
  // list.stream() //
  // .filter(linearLayer -> linearLayer instanceof LinearLayer) //
  // .map(LinearLayer.class::cast) //
  // .forEach(linearLayer -> linearLayer.l2 = l2);
  // }

  @Override
  public Stream<NetLayer> stream() {
    return list.stream().flatMap(l -> l.stream());
  }
}
