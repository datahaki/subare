// code by jph
package ch.alpine.subare.demo.prison;

import java.io.IOException;
import java.util.function.Supplier;

import ch.alpine.subare.book.ch02.Agent;
import ch.alpine.subare.book.ch02.UCBAgent;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Rescale;
import ch.alpine.tensor.alg.Subdivide;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.img.ColorDataGradients;
import ch.alpine.tensor.img.ImageResize;
import ch.alpine.tensor.img.Raster;
import ch.alpine.tensor.io.Export;
import ch.alpine.tensor.io.Put;

/* package */ class UCBsExact extends AbstractExact {
  public UCBsExact(Supplier<Agent> sup1, Supplier<Agent> sup2) {
    super(sup1, sup2);
    // ---
    contribute(new Integer[] { 0 }, new Integer[] { 0 });
    contribute(new Integer[] { 0 }, new Integer[] { 1 });
    contribute(new Integer[] { 1 }, new Integer[] { 0 });
    contribute(new Integer[] { 1 }, new Integer[] { 1 });
  }

  public static void showOne() {
    Supplier<Agent> sup1 = () -> new UCBAgent(2, RationalScalar.of(10, 10));
    Supplier<Agent> sup2 = () -> new UCBAgent(2, RationalScalar.of(8, 10));
    UCBsExact exact = new UCBsExact(sup1, sup2);
    System.out.println(exact.getExpectedRewards());
  }

  public static void main(String[] args) throws IOException {
    Tensor init = Subdivide.of(RationalScalar.of(3, 5), RationalScalar.of(3, 2), 240);
    Tensor expectedRewards = Array.zeros(init.length(), init.length());
    int px = 0;
    Tensor res = Tensors.empty();
    for (Tensor c0 : init) {
      Tensor row = Tensors.empty();
      int py = 0;
      for (Tensor c1 : init) {
        Supplier<Agent> sup1 = () -> new UCBAgent(2, (Scalar) c0);
        Supplier<Agent> sup2 = () -> new UCBAgent(2, (Scalar) c1);
        UCBsExact exact = new UCBsExact(sup1, sup2);
        row.append(Join.of(Tensors.of(c0, c1), exact.getExpectedRewards()));
        expectedRewards.set(exact.getExpectedRewards(), px, py);
        ++py;
      }
      res.append(row);
      ++px;
    }
    {
      Tensor rescale = Rescale.of(expectedRewards.get(Tensor.ALL, Tensor.ALL, 0));
      Tensor image = Raster.of(rescale, ColorDataGradients.CLASSIC);
      Export.of(HomeDirectory.Pictures("ucbs.png"), ImageResize.nearest(image, 2));
    }
    Put.of(HomeDirectory.file("ucb"), res);
  }
}
