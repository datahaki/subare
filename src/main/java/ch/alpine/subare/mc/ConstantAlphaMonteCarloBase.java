// code by jph
package ch.alpine.subare.mc;

import java.util.ArrayList;
import java.util.List;

import ch.alpine.subare.api.DiscountFunction;
import ch.alpine.subare.api.EpisodeDigest;
import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.util.LearningRate;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

public abstract class ConstantAlphaMonteCarloBase implements EpisodeDigest {
  protected final DiscountFunction discountFunction;
  protected final LearningRate learningRate;
  protected final StateActionCounter sac;

  protected ConstantAlphaMonteCarloBase(DiscountFunction discountFunction, LearningRate learningRate, StateActionCounter sac) {
    this.discountFunction = discountFunction;
    this.learningRate = learningRate;
    this.sac = sac;
  }

  @Override // from EpisodeDigest
  public final void digest(EpisodeInterface episodeInterface) {
    Tensor rewards = Tensors.empty();
    List<StepRecord> trajectory = new ArrayList<>();
    while (episodeInterface.hasNext()) {
      StepRecord stepInterface = episodeInterface.step();
      rewards.append(stepInterface.reward());
      trajectory.add(stepInterface);
    }
    digest(rewards, trajectory);
  }

  protected abstract void digest(Tensor rewards, List<StepRecord> trajectory);
}
