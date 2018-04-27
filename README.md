# Prediction Optimizer (to stabilize GAN training)

### Introduction
This is a PyTorch implementation of 'prediction method' introduced in the following paper ...

- Abhay Yadav et al., Stabilizing Adversarial Nets with Prediction Methods, ICLR 2018, [Link](https://openreview.net/forum?id=Skj8Kag0Z&noteId=rkLymJTSf)
- (*Just for clarification, I'm not an author of the paper.*)

The authors proposed a simple (but effective) method to stabilize GAN trainings. With this Prediction Optimizer, you can easily apply the method to your existing GAN codes. This impl. is compatible with **most of PyTorch optimizers and network structures**. (Please let me know if you have any issues using this)


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
        with pred_D.lookahead(step=1.0:)            # 'Predicted D'
            fake = netG(Z)                          # Draw samples from the real model. (not predicted one)
            D_outs = netD(fake)
    
            # Compute gradients and loss
        
            optim_G.step()
            pred_G.step()                           # You should call PredOpt.step() after each update
    ``` 
    
### Output samples
You can find more images at the following issues.
- [Cifar-10 (Prediction of G only)](https://github.com/sanghoon/prediction_gan/issues/1)
- [CelebA (Prediction of G only)](https://github.com/sanghoon/prediction_gan/issues/2)
- [Cifar-10 (Prediction of both D and G)](https://github.com/sanghoon/prediction_gan/issues/3)
- [CelebA (Prediction of both D and G)](https://github.com/sanghoon/prediction_gan/issues/4)

#### Training w/ large learning rate (0.01)

| Vanilla DCGAN | DCGAN w/ prediction (step=1.0) |
| --- | --- |
| ![ep25_cifar_base_lr 0 01](https://user-images.githubusercontent.com/3340388/38464108-fd288880-3b42-11e8-8392-7ac9d4261077.png) | ![ep25_cifar_pred_lr 0 01](https://user-images.githubusercontent.com/3340388/38499679-51c0e18e-3c43-11e8-9287-38f780db933c.png) |
| ![ep25_celeba_base_lr 0 01](https://user-images.githubusercontent.com/3340388/38464191-43ed0b3c-3b44-11e8-934e-2914a7b581a0.png) | ![ep25_celeba_pred_lr 0 01](https://user-images.githubusercontent.com/3340388/38499902-ecb3f30c-3c43-11e8-958b-a1b4e2aa6531.png) |

#### Training w/ medium learning rate (1e-4)
| Vanilla DCGAN | DCGAN w/ prediction (step=1.0) |
| --- | --- |
| ![ep25_cifar_base_lr 0 0001](https://user-images.githubusercontent.com/3340388/38464133-4e402fb6-3b43-11e8-9631-3d20e033a4d1.png) | ![ep25_cifar_pred_lr 0 0001](https://user-images.githubusercontent.com/3340388/38499708-64acd58c-3c43-11e8-8939-3d97f1ca5cb9.png) |
| ![ep25_celeba_base_lr 0 0001](https://user-images.githubusercontent.com/3340388/38464203-60690aae-3b44-11e8-855b-afd0f06d0abc.png) | ![ep25_celeba_pred_lr 0 0001](https://user-images.githubusercontent.com/3340388/38499884-dfa48b7c-3c43-11e8-8f90-0b7cac45c771.png) |

#### Training w/ small learning rate (1e-5)

| Vanilla DCGAN | DCGAN w/ prediction (step=1.0) |
| --- | --- |
| ![ep25_cifar_base_lr 0 00001](https://user-images.githubusercontent.com/3340388/38464153-852f2e64-3b43-11e8-937e-aa463b372291.png) | ![ep25_cifar_pred_lr 0 00001](https://user-images.githubusercontent.com/3340388/38499728-737c0de4-3c43-11e8-8c14-6b69e30e7f19.png) |
| ![ep25_celeba_base_lr 0 00001](https://user-images.githubusercontent.com/3340388/38464218-8eb51894-3b44-11e8-9839-1a259a82748a.png) | ![ep25_celeba_pred_lr 0 00001](https://user-images.githubusercontent.com/3340388/38499853-cf520466-3c43-11e8-8d4c-28adfd6d57dc.png) |


### External links

- GitHub repo. mentioned in the paper (https://github.com/jaiabhayk/stableGAN)
  - Empty by the date of this README.md update.
- Another impl. for PyTorch (https://github.com/shahsohil/stableGAN)
  - From the name of the repository owner, I guess it's written by one of the paper authors. (not 100% sure)
  - Currently supports ADAM only.

 
### TODOs
 
 - [x] : Impl. as an optimizer
 - [ ] : Support pip install
 - [x] : Add some experimental results 
