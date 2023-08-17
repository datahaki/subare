// code by jph
package ch.alpine.subare.book.ch04.rental;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** Example 4.2: Jack's Car Rental
 * Figure 4.2
 * 
 * p.87-88 */
/* package */ enum VI_CarRental {
  ;
  public static void main(String[] args) throws Exception {
    CarRental carRental = new CarRental(20);
    ValueIteration vi = new ValueIteration(carRental);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("carrental_vi.gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int count = 0; count <= 25; ++count) {
        System.out.println(count);
        animationWriter.write(CarRentalHelper.joinAll(carRental, vi.vs()));
        vi.step();
      }
      animationWriter.write(CarRentalHelper.joinAll(carRental, vi.vs()));
    }
  }
}
