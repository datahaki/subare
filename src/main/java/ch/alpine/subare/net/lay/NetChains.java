// code by jph
package ch.alpine.subare.net.lay;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import ch.alpine.subare.net.reg.NetRegularization;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
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

  public static NetLayer linearWith(List<Integer> dims, Supplier<NetLayer> supplier, NetRegularization netRegularization, Scalar dropout_p) {
    List<NetLayer> list = new ArrayList<>();
    for (int i = 0; i < dims.size() - 1; ++i) {
      if (0 < i) {
        list.add(supplier.get());
        if (Scalars.nonZero(dropout_p))
          list.add(DropoutLayer.of(dropout_p));
      }
      LinearLayer linearLayer = new LinearLayer(dims.get(i), dims.get(i + 1));
      linearLayer.netRegularization = netRegularization;
      list.add(linearLayer);
    }
    return new NetChain(list);
  }

  public static NetLayer cnnRelu(int kernelSize) {
    Distribution distribution = NormalDistribution.of(0, 0.1);
    return NetChain.join( //
        Conv1DLayer.of(distribution, ThreadLocalRandom.current(), kernelSize), //
        ElementwiseLayers.RELU.get());
  }
}
