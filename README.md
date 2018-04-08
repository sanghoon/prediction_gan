# Prediction Optimizer (to stabilize GAN training)
PyTorch Impl. of https://openreview.net/pdf?id=Skj8Kag0Z


### Introduction
This is a PyTorch implementation of the 'prediction method' introduced in the following paper ...

- Abhay Yadav et al., Stabilizing Adversarial Nets with Prediction Methods, ICLR 2018, [Link](https://openreview.net/forum?id=Skj8Kag0Z&noteId=rkLymJTSf)

You can also find one of the author's original impl. at https://github.com/shahsohil/stableGAN .
However, this impl. is compatible with **any optimizers** while the author's one only supports ADAM optimizer.


### How-to-use

#### Instructions
  - Import prediction.py
    - `from prediction import PredOpt`
  - Initialize just like an optimizer
    - `pred = PredOpt(net.parameters())`
  - Run the model in a 'with' block to get results from a model with predicted params.
    - With 'step' argument, you can control lookahead step size (1.0 by default)
    - ```python
      with pred.lookahead(step=1.0):
          output = net(input)
      ``` 
  - Call step() after an update of the network parameters
    - ```python
      optim_net.step()
      pred.step()
      ```

#### Samples
  - You can find a sample code in this repository (example_gan.py)
  - A sample snippet
  - ```python
    import torch.optim as optim
    from prediction import PredOpt
    
    
    # ...
    
    optim_G = optim.Adam(netG.parameters(), lr=0.01)
    optim_D = optim.Adam(netD.parameters(), lr=0.01)
    
    pred_G = PredOpt(netG.parameters())             # Create an prediction optimizer with target parameters
    pred_D = PredOpt(netD.parameters())
    
    
    for i, data in enumerate(dataloader, 0):
        # (1) Training D with samples from predicted generator
        with pred_G.lookahead(step=1.0):            # in the 'with' block, the model works as a 'predicted' model
            fake_predicted = netG(Z)                           
        
            # Compute gradients and loss 
        
            optim_D.step()
        
        
        # (2) Training G        
        with pred_D.lookahead(step=1.0:)
            fake = netG(Z)                          # Draw samples from the real model. (not predicted one)
            D_outs = netD(fake)
    
            # Compute gradients and loss
        
            optim_G.step()
            pred_G.step()                           # You should call PredOpt.step() after each update
    ``` 
    
### Output samples w/ large learning rate (0.01)
#### Cifar-10

You can find more images at https://github.com/sanghoon/prediction_gan/issues/1

##### Vanilla DCGAN
![ep25_cifar_base_lr 0 01](https://user-images.githubusercontent.com/3340388/38464108-fd288880-3b42-11e8-8392-7ac9d4261077.png)
##### DCGAN w/ prediction (step=1.0)
**Wrong results. To be updated**
![ep25_cifar_pred_lr 0 01](https://user-images.githubusercontent.com/3340388/38464113-042961e0-3b43-11e8-85f4-a6827d95344d.png)
 
#### CelebA (50k images only)

You can find more images at https://github.com/sanghoon/prediction_gan/issues/2

##### Vanilla DCGAN
![ep25_celeba_base_lr 0 01](https://user-images.githubusercontent.com/3340388/38464191-43ed0b3c-3b44-11e8-934e-2914a7b581a0.png)
##### DCGAN w/ prediction (step=1.0)
**Wrong results. To be updated**
![ep25_celeba_pred_lr 0 01](https://user-images.githubusercontent.com/3340388/38464196-51cbc860-3b44-11e8-867c-a285afdd0a6f.png)


#####
### TODOs
 
 - [x] : Impl. as an optimizer
 - [ ] : Support pip install
 - [x] : Add some experimental results 
