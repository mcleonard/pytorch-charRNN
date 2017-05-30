# Character-wise RNN

Here's an implementation of a character-wise recurrent neural network written with PyTorch. The model is inspired by [Andrej Karpathy's](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) excellent blog post on RNNs. This neural network learns about the structure of text one character at a time, then can generate new text one character at time, based on what it learned previously. This repository contains a notebook walking through the code and describing how it works. There are also scripts to train and sample from networks.

## Dependencies

Python 3.6, Numpy, PyTorch 0.1.12

## Script usage

### Training

To train a network on some text run the `train.py` script passing in your text file and the directory where you want to save the model checkpoint.

```bash
python train.py --in_file text.txt --save_dir save
```

You can set the size of the hidden layers with `--rnn_size 512` and the number of hidden layers with `--num_layers 3`. Add the `--gpu` flag to run the training on a GPU with CUDA.

For more options, use `python train.py --help`

### Sampling

For generating text by sampling from a trained model, use

```bash
python sample.py checkpoint.ckpt --num_samples 1000
```

This will load the model from the checkpoint and generate a new text with 1000 characters. Again, for more options, use `python sample.py --help`


## Example generated text

I trained this network on the text from Tolstoy's Anna Karenina with 512 hidden units and 2 layers.

```
Anna, and she seemed to
have a tears of himself, but as though he did not tell them his face. He
was not settled at him, and with whom he had already sat down, and that
it was impossible to say."

"That's not somewhere the point to her." They despring them at that
mistake and him. The marshes was the chincer, was in a glance of
assisted at the corricted service. But when she had so given on the
presence.

"I have been always to the than except into a bare, but there,
as it is to be supposed." Again he they had not all to be
about the mistares and thousands as so towards it, and what the
mather was to the carriage with silence, and walked a love with which
he had true, that he had an acquaintance. The creeming
sense of the colorel sorts, and at the comminterness of his own
her hands, and had belonged to the peasants, he had a country peasant
his breathed handed, he sat sight of the same time as his friends, to see
where the household has taken in out of the same time and his sister
in a long wife.
```

