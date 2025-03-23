Character level text generation. Follow the following steps and show your works in a jupyter notebook. The requirements of the notebook are emphasis in bold.

1. Find and download a suitable data for training a character level text generation model.
+ The text material can be English, Chinese, or Code.
+ The training data should contain at least 5000 sentences or comparable amount of data.
2. Gather the statistics of P(xt​∣xt−1​,xt−2​,…,xt−n​) for the training dataset, where xi are characters in the training dataset. Do the followings for at least 2 different n,
+ Store the information in a counter.
+ Show how many distinct tuples (xt−1​,xt−2​,…,xt−n) are there in the training data.
+ Which tuple (xt−1​,xt−2​,…,xt−n) appears most times int the dataset?
+ Show the top 3 candidate's x_t that is most likely to appear right after the above tuple.
+ Generate 3 paragraph of text according to the statistics. You may start from any tuples that is appeared in the dataset.
3. Warmup for neural network and deep learning
+ Choose any of the deep learning framework. Pytorch lightning and keras are recommended.
+ Instead of using a counter, train a neural network to learn P(xt​∣xt−1​,xt−2​,…,xt−n​) or P(xt​∣xt−1​,xt−2​,…)
+ The network can be an RNN, 1D-CNN or an MLP. You can limit x_i to be a subset of the characters, e.g. choose only the top 100~1000 most frequently appeared characters.
+ Generate 3 paragraph of text using this neural network model.

Submit your work in a single jupyter notebook.