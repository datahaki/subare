// code by jph
package ch.alpine.subare.net;

import java.util.List;
import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.MergeIllegal;
import ch.alpine.tensor.sca.Sign;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetChain.html">NetChain</a> */
public record NetChain(List<NetLayer> list) implements NetLayer {
  public static NetLayer of(NetLayer... layers) {
    return new NetChain(List.of(layers));
  }

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    list.forEach(netLayer -> netLayer.initialize(randomGenerator));
  }

  @Override
  public Stream<Scalar> parameters() {
    return list.stream().flatMap(NetLayer::parameters);
  }

  @Override
  public Tensor forward(Tensor x0) {
    return list.stream().reduce(x0, (x, layer) -> layer.forward(x), MergeIllegal.operator());
  }

  @Override
  public Tensor back(Tensor grad0) {
    return list.reversed().stream().reduce(grad0, (grad, layer) -> layer.back(grad), MergeIllegal.operator());
  }

  @Override
  public void update() {
    list.forEach(NetLayer::update);
  }

  @Override
  public Tensor error(Tensor tensor) {
    return list.getLast().error(tensor);
  }

  // TODO probably obsolete
  public void setL2(Scalar l2) {
    Sign.requirePositiveOrZero(l2);
    list.stream() //
        .filter(linearLayer -> linearLayer instanceof LinearLayer) //
        .map(LinearLayer.class::cast) //
        .forEach(linearLayer -> linearLayer.l2 = l2);
  }
}
