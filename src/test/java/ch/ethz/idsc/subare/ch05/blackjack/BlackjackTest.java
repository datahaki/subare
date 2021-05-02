// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.util.Map;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.red.Tally;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import junit.framework.TestCase;

public class BlackjackTest extends TestCase {
  @SuppressWarnings("unused")
  public void testSimple() {
    Blackjack blackjack = new Blackjack();
    // TODO fail sometimes, correct or wrong?
    {
      Tensor next = blackjack.move(Tensors.vector(0, 18, 7), RealScalar.ONE);
      // assertEquals(next, Tensors.vector(-1));
    }
    {
      Tensor next = blackjack.move(Tensors.vector(0, 21, 7), RealScalar.ZERO);
      // assertEquals(next, Tensors.vector(1));
    }
  }

  public void testEpisodeLength() {
    Blackjack blackjack = new Blackjack();
    Policy pi = EquiprobablePolicy.create(blackjack);
    Tensor tally = Tensors.empty();
    for (int episodes = 0; episodes < 10000; ++episodes) {
      EpisodeInterface ei = EpisodeKickoff.single(blackjack, pi);
      int count = 0;
      while (ei.hasNext()) {
        ei.step();
        ++count;
      }
      tally.append(RealScalar.of(count));
    }
    Map<Tensor, Long> map = Tally.of(tally);
    // {1=6574, 2=2537, 3=759, 4=121, 5=8, 7=1}
    // {2=2497, 1=6623, 6=1, 5=18, 4=138, 3=723}
    assertTrue(5 <= map.size());
    System.out.println("" + Tally.of(tally));
  }
}
