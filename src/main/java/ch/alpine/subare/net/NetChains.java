// code by jph
package ch.alpine.subare.net;

import java.util.concurrent.ThreadLocalRandom;

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

  public static NetLayer feedForward(int d_in, int hidden, int dout) {
    return NetChain.of( //
        new LinearLayer(hidden, d_in), //
        ElementwiseLayer.gelu(), //
        new LinearLayer(dout, hidden), //
        new SoftArgMax());
  }

  public static NetLayer linTanhLin(int d_in, int hidden, int dout) {
    return NetChain.of( //
        new LinearLayer(hidden, d_in), //
        ElementwiseLayer.tanh(), //
        new LinearLayer(dout, hidden));
  }

  public static NetLayer cnnRelu(int kernelSize) {
    Distribution distribution = NormalDistribution.of(0, 0.1);
    return NetChain.of( //
        Conv1DLayer.of(distribution, ThreadLocalRandom.current(), kernelSize), //
        ElementwiseLayer.relu());
  }
}
