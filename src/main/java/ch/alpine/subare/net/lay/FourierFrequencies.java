package ch.alpine.subare.net.lay;

import java.util.random.RandomGenerator;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.num.Pi;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

public class FourierFrequencies {
  public static Tensor sampleB(int numFrequencies, Scalar sigma, RandomGenerator randomGenerator) {
    Distribution distribution = NormalDistribution.of(sigma.zero(), sigma);
    return RandomVariate.of(distribution, randomGenerator, numFrequencies, 2).multiply(Pi.TWO);
  }
}
