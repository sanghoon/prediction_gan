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
    
    
    for i, data in enumerate(dataloader, 0):
        # (1) Training D with samples from predicted generator
    
        with pred_G.lookahead(step=1.0):            # in the 'with' block, the model works as a 'predicted' model
            fake_predicted = netG(Z)                           
        
        # Compute loss 
        
        optim_D.step()
        
        
        # (2) Training G
        
        fake = netG(Z)                              # Draw samples from the real model. (not predicted one)
            
        # Compute loss
        
        optim_G.step()
        pred_G.step()                               # You should call PredOpt.step() after each update
    ``` 
    
    
 ### TODOs
 
 - [x] : Impl. as an optimizer
 - [ ] : Support pip install
 - [ ] : Add some experimental results 