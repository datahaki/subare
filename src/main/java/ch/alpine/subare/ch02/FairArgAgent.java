// code by jph
package ch.alpine.subare.ch02;

import ch.alpine.subare.util.FairArg;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;

public abstract class FairArgAgent extends Agent {
  protected abstract Tensor getQVector();

  @Override
  public final int protected_takeAction() {
    FairArg fairArgMax = FairArg.max(getQVector());
    if (!fairArgMax.isUnique()) {
      int index = Scalars.intValueExact(getCount());
      if (index < openingSequence.size())
        return openingSequence.get(index);
      notifyAboutRandomizedDecision();
    }
    return fairArgMax.nextRandomIndex(); // (2.2)
  }
}
