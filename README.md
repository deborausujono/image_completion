# Image completion

Given an image with a square region of missing pixels, our goal is to fill in the missing pixels in a visually plausible way. We plan to start with the CIFAR-10 dataset, removing a square region of each image. Time permitting, we will try our method on other datasets, such as ImageNet or the lamprey dataset from our original project.

As our baseline, we will implement together CNNs with MSE loss to do regression on the intensities of the missing pixels. Then, each of us will investigate different methods using generative models to improve the baseline:
- Debora: PixelCNN and PixelRNN [1, 2]
- Grace: GAN and DCGAN [3, 4]
- Yue: Image completion with perceptual and contextual losses [5]

After experimentation, we will compare our results and all of us will work together to optimize one or two models that we find most promising.

We will use RMSE to evaluate our results. Since the completed image does not have to duplicate the original image as long as it looks real and plausible to a human judge, we will also randomly sample images for human judgment.

## References:
[1] Conditional Image Generation with PixelCNN Decoders https://arxiv.org/pdf/1606.05328v2.pdf
[2] Pixel Recurrent Neural Networks https://arxiv.org/pdf/1601.06759v3
[3] Generative Adversarial Nets http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
[4] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks https://arxiv.org/pdf/1511.06434v2
[5] Semantic Image Inpainting with Perceptual and Contextual Losses https://arxiv.org/pdf/1607.07539v2.pdf
