# Assignment 2, Part 2: Micro-GPT

This folder contains the template code for implementing your own GPT2 model. We will train the model to predict the next characters. The code is structured in the following way:

* `dataset.py`: Contains the implementation of the character-level text dataset and tokenizer. It includes:
  - `CharTokenizer`: A class that handles conversion between characters and indices, with methods for encoding text to indices and decoding indices back to text.
  - `TextDataset`: A PyTorch Dataset class that processes text data in blocks, preparing it for training the GPT model. It loads text data and provides character sequences for training, where each input sequence (x) is paired with its corresponding target sequence (y) shifted by one character.
* `gpt.py`: Contains template classes for the building block of the GPT decoder-only model.
* `cfg.py`: Contains the configuration setup using ArgumentParser. It defines various hyperparameters and settings for the model training, including:
  - Model configuration (text file path, model type, block size, pretrained options)
  - Training parameters (batch sizes, learning rate, optimizer settings, number of epochs)
  - System settings (logging directory, seed, number of workers)
  - Performance options (flash attention, precision, model compilation (torch.compile))
* `train.py`: Contains the training implementation for the GPT model using PyTorch Lightning. It handles model training, evaluation, text generation during training, and supports both training from scratch and fine-tuning pretrained models. The code uses TensorBoard for logging and includes standard training optimizations like gradient clipping and precision options.

* `generate.py`: Contains the implementation for text generation using the trained GPT model. It includes the `generate` function, which takes a prompt, the weights of the model you trained, and various generation parameters, and generates text samples based on the model's predictions.
<!-- * `unittests.py`: Contains unittests for the Encoder, Decoder, Discriminator networks. It will hopefully help you debug your code. Your final code should pass these unittests. -->

Default hyperparameters are provided in the `ArgumentParser` object of the `cfg.py` file. The model should be able to generate decent images with the default hyperparameters.
The training time for descent performance, with the default hyperparameters and architecture, is less than 30 minutes on gpu partition of the Snellius cluster.

## Training
Integrate ``` python train.py ``` into your SLURM script. 

The default configuration uses rotary position embeddings. 
Optionally, you could use ``` python train.py --abs_emb ``` for using learnable absolute position embedding and ```--use_flash_attn``` for using flash attention. 

After completing all the modules, please use the full command to train your model for 5 epochs:

```python train.py --use_flash_attn --compile --num_epochs 5```

