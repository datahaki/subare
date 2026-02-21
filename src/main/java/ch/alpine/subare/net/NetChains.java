// code by jph
package ch.alpine.subare.net;

import java.util.concurrent.ThreadLocalRandom;

import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clips;

public enum NetChains {
  ;
  /** @param INPUT_SIZE
   * @param hiddenSize
   * @param OUTPUT_SIZE
   * @return */
  public static NetChain binary(int INPUT_SIZE, int hiddenSize, int OUTPUT_SIZE) {
    Distribution distribution = UniformDistribution.of(Clips.absolute(0.5));
    return NetChain.of( //
        LinearLayer.of(distribution, ThreadLocalRandom.current(), hiddenSize, INPUT_SIZE), //
        ElementwiseLayer.logSig(), //
        LinearLayer.of(distribution, ThreadLocalRandom.current(), OUTPUT_SIZE, hiddenSize), //
        ElementwiseLayer.logSig(), //
        new BinaryLayer());
  }

  public static NetChain argMaxMLP(int d_in, int hidden, int dout) {
    Distribution distribution = NormalDistribution.of(0.0, 0.1);
    return NetChain.of( //
        LinearLayer.of(distribution, ThreadLocalRandom.current(), hidden, d_in), //
        ElementwiseLayer.relu(), //
        LinearLayer.of(distribution, ThreadLocalRandom.current(), dout, hidden), //
        new SoftArgMax());
  }

  public static NetChain linTanhLin(int d_in, int hidden, int dout) {
    return NetChain.of( //
        LinearLayer.xavier(hidden, d_in), //
        ElementwiseLayer.tanh(), //
        LinearLayer.xavier(dout, hidden));
  }
}
