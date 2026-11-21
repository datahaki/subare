// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.api.NetLoss;
import ch.alpine.subare.net.api.NetOptimizer;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.MergeIllegal;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetChain.html">NetChain</a> */
public record NetChain(List<NetLayer> list) implements NetLayer, Serializable {
  /** join all given net layers in the order given.
   * if any of the given net layers are instances of NetChain
   * the sequence of that net chain are brought down onto the first level
   * of the resulting net chain
   * 
   * @param netLayers
   * @return */
  public static NetLayer join(NetLayer... netLayers) {
    return new NetChain(Stream.of(netLayers) //
        .flatMap(netLayer -> netLayer instanceof NetChain netChain //
            ? netChain.list().stream()
            : Stream.of(netLayer))
        .toList());
  }

  @Override
  public Tensor forward(Tensor x0) {
    return list.stream() // fold
        .reduce(x0, (x, layer) -> layer.forward(x), MergeIllegal.operator());
  }

  @Override
  public Tensor forwardBatch(Tensor X0) {
    return list.stream() // fold
        .reduce(X0, (X, layer) -> layer.forwardBatch(X), MergeIllegal.operator());
  }

  @Override
  public NetLoss error(Tensor target) {
    return list.getLast().error(target);
  }

  @Override
  public Tensor back(Tensor grad0) {
    return list.reversed().stream() // fold
        .reduce(grad0, (grad, layer) -> layer.back(grad), MergeIllegal.operator());
  }

  @Override
  public Tensor backBatch(Tensor grad0) {
    return list.reversed().stream() // fold
        .reduce(grad0, (grad, layer) -> layer.backBatch(grad), MergeIllegal.operator());
  }

  @Override
  public Stream<NetLayer> nested() {
    return list.stream();
  }

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    list.forEach(l -> l.initialize(randomGenerator));
  }

  /** @return */
  @Override
  public Tensor parameters() {
    return Tensor.of(list.stream().map(NetLayer::parameters), list.size());
  }

  /** @param decayInLoss whether to include the gradient from the L2 (!) regularization,
   * i.e. "weight decay", in the gradient descent
   * @return */
  @Override
  public Tensor gradient(boolean decayInLoss) {
    throw new UnsupportedOperationException();
  }

  /** @param decayInLoss whether the gradient from the L2 (!) regularization was considered
   * in the gradient descent
   * @param delta
   * @param eta */
  @Override
  public void increment(boolean decayInLoss, Tensor delta, Scalar eta) {
    throw new UnsupportedOperationException();
  }

  /** note:
   * for certain {@link NetOptimizer} the weight decay does not
   * show in the loss function
   * 
   * @return contribution to loss by weight regularization
   * @see NetOptimizers */
  @Override
  public Scalar reg_loss(boolean decayInLoss) {
    return list.stream() //
        .map(l -> l.reg_loss(decayInLoss)) //
        .reduce(Scalar::add) //
        .orElseThrow();
  }
}
