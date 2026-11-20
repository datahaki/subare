// code by jph
package ch.alpine.subare.net.lay;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.c.NormalDistribution;

public enum NetChains {
  ;
  /** @param supplier
   * @param dims
   * @return sequence of linear layers dims[0] -> dim[1] -> dim[last]
   * with in-between netLayers provided by supplier */
  public static NetLayer linear(Supplier<NetLayer> supplier, int... dims) {
    List<NetLayer> list = new ArrayList<>();
    for (int i = 0; i < dims.length - 1; ++i) {
      if (0 < i)
        list.add(supplier.get());
      list.add(new LinearLayer(dims[i], dims[i + 1]));
    }
    return new NetChain(list);
  }

  /** @param v
   * @param d
   * @param netLayer that takes as input a vector of length s*d where s is
   * the sequence length, and outputs vector of length 1
   * @return */
  public static NetLayer embedding(int v, int d, NetLayer netLayer) {
    return NetChain.join( //
        new EmbeddingLayer(v, d), //
        new FlattenLayer(), //
        netLayer);
  }

  /** the output of netLayer is a vector of length n
   * where n corresponds to the number of labels for classification
   * 
   * @param netLayer
   * @return layer that outputs vector of length 1 with integer value as entry */
  public static NetLayer argMax(NetLayer netLayer) {
    return NetChain.join( //
        netLayer, //
        SoftmaxLayer.INSTANCE, //
        new ArgMaxLayer());
  }

  /** intended for binary classification tasks
   * even though network layer does not round output to integer values (!)
   * 
   * @param netLayer with final layer that returns a vector of length 1
   * @return */
  public static NetLayer binary(NetLayer netLayer) {
    return NetChain.join( //
        netLayer, //
        ElementwiseLayers.SIGMOID.get(), //
        new BinaryLayer());
  }

  public static NetLayer cnnRelu(int kernelSize) {
    Distribution distribution = NormalDistribution.of(0, 0.1);
    return NetChain.join( //
        Conv1DLayer.of(distribution, ThreadLocalRandom.current(), kernelSize), //
        ElementwiseLayers.RELU.get());
  }
}
