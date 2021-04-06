# semisupervised-learning

This project shows a comparison of 3 different techniques of semi-supervised learning. Semi- supervised learning is a field of machine learning that allows 
a model to gain high accuracy with a very small set of labelled training data, and a large unlabelled dataset. This is a very important field of study as 
labelling data is time consuming and expensive, while unlabelled data is very easy to obtain. The 3 different methods I compared are
* Rotations: Each unlabelled image is rotated by a multiple of 90° and a model is trained to recognise the rotation that was performed. Transfer learning 
is then used to train the network on the labelled data.
* Relative Patch Location: Each unlabelled image is split into 9 small patches, and the centre patch along with one other random patch are chosen. 
The model is trained to predict which patch was chosen. Again, transfer learning is then used to train this network on the labelled data.
* Solving Jigsaws: Again, each unlabelled image is split into 9, but this time all the patches are jumbled up. The model is trained to predict the 
permutation used to jumble up the image. Finally, transfer learning is used to train this network on the labelled data.

All three of these methods provided far better accuracy than what can be achieved with just the labelled training data, 
with the solving jigsaw task providing the best performance of the 3.

The STL10 dataset was used which contains 100,000 unlabelled images and 5000 labelled image obtained from the ImageNet dataset. The results of the experiments were
* Baseline - 46.5% Accuracy
* Rotations - 55.0% Accuracy
* Relative Patch Location  - 55.3% Accuracy
* Solving Jigsaws - 60.9% Accuracy

See the PDF document for the full report containing detailed a description of the implementation and results
