// code by jph, gemini
package ch.alpine.subare.net.lay;

import java.util.function.Supplier;

import ch.alpine.subare.net.api.NetLayer;

public enum ElementwiseLayers {
  /** When to Use Sigmoid
   * 
   * The defining characteristic of Sigmoid is that its output is strictly bounded
   * between $0$ and $1$. This makes it ideal for scenarios requiring probabilities
   * or gating percentages.
   * 
   * Binary Classification Output Layers:
   * Use case: In the final layer of a binary classifier, you want a probability
   * score representing the likelihood of a positive class (e.g., spam vs. not spam).
   * Sigmoid maps any real-valued number into a $0$ to $1$ interval.
   * Gating Mechanisms in Recurrent Networks (LSTMs & GRUs):
   * Use case: Inside LSTM and GRU gates (forget gate, input gate, output gate), you
   * need to decide what fraction of information to let through. A value of $0$ means
   * "block everything," $1$ means "let everything through," and values in between
   * act as a proportional filter. */
  SIGMOID {
    @Override
    public Supplier<NetLayer> supplier() {
      return () -> ElementwiseLayer.sigmoid();
    }
  },
  /** When to Use Tanh
   * 
   * Tanh scales Sigmoid's output range to $-1$ to $1$. Because it is zero-centered,
   * negative inputs map to negative outputs, zero maps to zero, and positive inputs
   * map to positive outputs.
   * 
   * Hidden Layers in Recurrent Neural Networks (RNNs):
   * Use case: RNN hidden states benefit greatly from Tanh because its zero-centered
   * nature helps keep the mean of the activations closer to zero across layers,
   * mitigating some of the directional bias during backpropagation compared to
   * Sigmoid.
   * 
   * Gating Cell States in LSTMs:
   * Use case: While LSTMs use Sigmoid for gates, they use Tanh to create candidate
   * values for the cell state, allowing updates to swing freely between negative and
   * positive values.
   * 
   * Generative Adversarial Networks (GANs):
   * Use case: The generator network in some GAN architectures uses Tanh as its final
   * activation function to map pixel values to a $[-1, 1]$ range, which often aligns
   * well with normalized image datasets. */
  TANH {
    @Override
    public Supplier<NetLayer> supplier() {
      return () -> ElementwiseLayer.tanh();
    }
  },
  /** When to Choose ReLU
   * 
   * Inference Latency & Edge Deployment: If you are deploying to mobile devices,
   * microcontrollers (TinyML), or older NPUs, ReLU is significantly faster and
   * consumes less power. Even on GPUs, ReLU requires fewer FLOPs and lower memory
   * bandwidth than GELU.
   * 
   * Convolutional Neural Networks (CNNs): Standard vision backbones (like ResNets
   * or MobileNets) often see negligible accuracy gains from GELU while paying a
   * noticeable latency penalty. (Note: ReLU6 or SiLU/Swish are also common
   * alternatives here).
   * 
   * Sparsity & Pruning: Because ReLU outputs exact zeros for all negative inputs,
   * it naturally creates sparse activations, which can be exploited for model
   * compression and faster matrix multiplications.
   * 
   * Shallow Networks: In networks with fewer than 5–10 layers, the "dying ReLU"
   * problem is rarely severe enough to justify the overhead of GELU. */
  RELU {
    @Override
    public Supplier<NetLayer> supplier() {
      return () -> ElementwiseLayer.relu();
    }
  },
  /** When to Choose GELU
   * 
   * Transformers & Large Language Models: GELU (along with its gated cousin SwiGLU)
   * is the gold standard for self-attention architectures (BERT, GPT, ViT). Its
   * smooth curvature helps stabilize training in very deep networks where subtle
   * gradient signals matter.
   * 
   * Preventing Dead Neurons: Because GELU allows a small, smooth curve into negative
   * values before tapering to zero, neurons don't permanently "die" during aggressive
   * learning rate schedules.
   * 
   * Training Stability on Complex Distributions: When modeling high-dimensional,
   * continuous data (like language embeddings or latent diffusion spaces), GELU's
   * probabilistic gating ($x$ weighted by its standard normal CDF) empirically
   * converges faster and reaches lower loss than ReLU. */
  GELU {
    @Override
    public Supplier<NetLayer> supplier() {
      return () -> ElementwiseLayer.gelu();
    }
  };

  public abstract Supplier<NetLayer> supplier();
}
