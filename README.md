This is a working Python implementation of a latent backdoor attack algorithm on a deep neural network(DNN) trained on the MNIST dataset for digit recognition.
With this algorithm one was is able to poison a neural network indirectly, without having access to it, by poisoning its teacher network. This code aims to test its efficacy and attack success rate.

To run on your own computer:
#1. Download the MNIST dataset and place it into the "MNIST" directory. (Found [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset))
#2. Use the main script to split the dataset into respective student and teacher categories. By default, the teacher model handles digits (0-4) and the student model digits (5-9), feel free to experiment with different setups!
#3. Use the main script to train your teacher model.
#4. Use the PatternGenerate.py script to generate an attack noise pattern using any of the provided algorithms.
#5. Use the Poison function of the main script to poison the teacher model.
#6. Use the transfer learning function of the main script to create the student model and test its accuracy.
#7. Apply the trigger pattern to a randomized subset of student model digits and test the manipulation accuracy.

FUTURE IMPROVEMENTS:
#1. Clean up PatternGenerate script and turn it into an appropriate class.
#2
