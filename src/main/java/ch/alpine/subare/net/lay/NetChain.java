// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.MergeIllegal;
import ch.alpine.tensor.io.MathematicaFormat;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetChain.html">NetChain</a> */
public record NetChain(List<NetLayer> list) implements NetLayer, Serializable {
  public NetChain(NetLayer... netLayers) {
    this(Arrays.asList(netLayers));
  }

  /** join all given net layers in the order given.
   * if any of the given net layers are instances of NetChain
   * the sequence of that net chain are brought down onto the first level
   * of the resulting net chain
   * 
   * @param netLayers
   * @return */
  public static NetLayer join(NetLayer... netLayers) {
    return join(Stream.of(netLayers));
  }

  public static NetLayer join(Stream<NetLayer> stream) {
    return new NetChain(stream //
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
  public final String toString() {
    return MathematicaFormat.concise("NetChain", list.size());
  }
}
