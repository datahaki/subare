// code by jph
package ch.alpine.subare.net;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.c.NormalDistribution;

public enum NetChains {
  ;
  /** @param INPUT_SIZE
   * @param hiddenSize
   * @param OUTPUT_SIZE
   * @return */
  public static NetLayer binary(int INPUT_SIZE, int hiddenSize, int OUTPUT_SIZE) {
    return NetChain.of( //
        new LinearLayer(hiddenSize, INPUT_SIZE), //
        ElementwiseLayer.logSig(), //
        new LinearLayer(OUTPUT_SIZE, hiddenSize), //
        ElementwiseLayer.logSig(), //
        new BinaryLayer());
  }

  public static NetLayer argMaxMLP(int d_in, int hidden, int dout) {
    return NetChain.of( //
        new LinearLayer(hidden, d_in), //
        ElementwiseLayer.gelu(), //
        new LinearLayer(dout, hidden), //
        new SoftArgMax());
  }

  public static NetLayer cnnRelu(int kernelSize) {
    Distribution distribution = NormalDistribution.of(0, 0.1);
    return NetChain.of( //
        Conv1DLayer.of(distribution, ThreadLocalRandom.current(), kernelSize), //
        ElementwiseLayer.relu());
  }

  public static NetLayer sequence(Supplier<NetLayer> supplier, int... dims) {
    List<NetLayer> list = new ArrayList<>();
    for (int i = 0; i < dims.length - 1; ++i) {
      if (0 < i)
        list.add(supplier.get());
      list.add(new LinearLayer(dims[i + 1], dims[i]));
    }
    return new NetChain(list);
  }
}
