// code by fluric
package ch.alpine.subare.util;

import java.security.SecureRandom;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.d.BernoulliDistribution;
import ch.alpine.tensor.sca.Clips;

public class Coinflip {
  private static final RandomGenerator RANDOM_GENERATOR = new SecureRandom();
  private static final Coinflip FAIR = new Coinflip(RationalScalar.HALF);

  /** @param p_head in the interval [0, 1]
   * @return new instance of Coinflip with given probability p_head that {@link #tossHead()} returns true
   * @throws Exception if given probability is not inside the unit interval */
  public static Coinflip of(Scalar p_head) {
    return new Coinflip(Clips.unit().requireInside(p_head));
  }

  /** Quote from Wikipedia:
   * "a fair coin is an idealized randomizing device with two states
   * (usually named "heads" and "tails") which are equally likely to occur."
   * 
   * @return new instance of Coinflip for which {@link #tossHead()} returns true with probability 1/2 */
  public static Coinflip fair() {
    return FAIR;
  }

  // ---
  private final Distribution distribution;

  private Coinflip(Scalar p_head) {
    distribution = BernoulliDistribution.of(RealScalar.ONE.subtract(p_head));
  }

  /** @param randomGenerator
   * @return whether the coin toss ended up with head */
  public boolean tossHead(RandomGenerator randomGenerator) {
    return Scalars.isZero(RandomVariate.of(distribution, randomGenerator));
  }

  /** @return whether the coin toss ended up with head */
  public boolean tossHead() {
    return tossHead(RANDOM_GENERATOR);
  }
}
