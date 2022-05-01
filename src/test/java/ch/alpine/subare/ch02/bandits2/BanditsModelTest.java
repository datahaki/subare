// code by jph
package ch.alpine.subare.ch02.bandits2;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.red.Variance;
import ch.alpine.tensor.sca.Chop;

class BanditsModelTest {
  @Test
  public void testMean() {
    int num = 10;
    BanditsModel banditsModel = new BanditsModel(num);
    Tensor means = Tensors.vector(k -> banditsModel.expectedReward(BanditsModel.START, RealScalar.of(k)), num);
    Chop._10.requireAllZero(Mean.of(means));
    Tensor starts = banditsModel.startStates();
    assertEquals(starts.length(), 1);
  }

  @Test
  public void testExact() {
    int num = 20;
    BanditsModel banditsModel = new BanditsModel(num);
    DiscreteQsa ref = BanditsHelper.getOptimalQsa(banditsModel);
    Tensor expected = Tensors.vector(i -> ref.value(BanditsModel.START, RealScalar.of(i)), num);
    Scalar mean = (Scalar) Mean.of(expected);
    Chop._10.requireAllZero(mean);
    Scalar var = Variance.ofVector(expected);
    Chop._10.requireClose(var, RealScalar.ONE);
  }
}
