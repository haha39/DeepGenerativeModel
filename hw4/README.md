# Dreambooth Training
## Objective
Create your own dataset and finetune a StableDiffusion model using dreambooth method.

## Instructions
+ Use huggingface diffusers for finetuning.
    + Refer to https://huggingface.co/docs/diffusers/en/training/dreambooth for training instruction.
    + You can use the provided scripts for training, but create your own dataset. See reference scripts here: https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
    + You can choose to train it with or without lora, and can choose to use sdxl, sd 1.x or sd 2.x as the base model.
+ Present your work and results using the notebook
    + Example notebooks can be found here: notebooks/diffusers at main · huggingface/notebooks (github.com)
    + It is acceptable to use these notebooks, but modifications are necessary:
        + Adjust the package installation scripts in order to make it works on your machine.
        + Show some sample text/image pairs of your dataset.
        + Remove the unused parts of the notebook.
    + It is acceptable to call diffusers scripts like train_dreambooth_lora_sdxl_advanced using accelerate from the notebook.
    + It is preferable to run the training on a GPU. Use free GPU resources like colab, kaggle if needed.
## Key requirement of the Homework
+ Show that you create your own dataset.
    + Show some samples of image/text pairs in you dataset.
    + Demonstrates how you create the image/text pairs. Using templates or using blip? Provide codes in the notebook.
+ Show that you successfully run the training process.
    + So there should be some log or progress bar in the notebook output.
+ Show the resulting model works.
    + Show at least 10 sample images generated by the resulting model.
    + Provide some creative prompts to see how well the model generalized to prompts not existed in the training data.

+ Submit your work in a single jupyter notebook.