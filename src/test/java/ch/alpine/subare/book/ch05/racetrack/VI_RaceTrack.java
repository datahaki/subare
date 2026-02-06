// code by jph
package ch.alpine.subare.book.ch05.racetrack;

import java.nio.file.Path;
import java.util.LinkedList;
import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.mc.MonteCarloEpisode;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.img.ImageResize;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.io.Primitives;

/* package */ enum VI_RaceTrack {
  ;
  public static void make(String name, int maxSpeed, Path file) throws Exception {
    Racetrack racetrack = RacetrackHelper.create(name, maxSpeed);
    ValueIteration vi = new ValueIteration(racetrack, racetrack);
    vi.untilBelow(RealScalar.of(10), 5);
    System.out.println("iterations=" + vi.iterations());
    Policy policy = PolicyType.GREEDY.bestEquiprobable(racetrack, vi.vs(), null);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(file, 400, TimeUnit.MILLISECONDS)) {
      for (Tensor start : racetrack.statesStart) {
        System.out.println(start);
        Tensor image = racetrack.image();
        MonteCarloEpisode mce = new MonteCarloEpisode( //
            racetrack, policy, start, new LinkedList<>());
        while (mce.hasNext()) {
          StepRecord stepRecord = mce.step();
          {
            Tensor state = stepRecord.prevState();
            int[] index = Primitives.toIntArray(state);
            image.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
          }
          {
            Tensor state = stepRecord.nextState();
            int[] index = Primitives.toIntArray(state);
            image.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
          }
        }
        animationWriter.write(ImageResize.nearest(image, 6));
      }
    }
    System.out.println("gif created");
  }

  static void main() throws Exception {
    make("track2", 5, HomeDirectory.Pictures.resolve(VI_RaceTrack.class.getSimpleName() + ".gif"));
  }
}
