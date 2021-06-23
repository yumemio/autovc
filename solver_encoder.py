from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.loss_type_id = config.loss_type_id   # Loss function for ident, 
        self.loss_type_id_psnt = config.loss_type_id_psnt # Loss function for id_psnt
        self.loss_type_cd = config.loss_type_cd # Loss function for cd

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.save_as = config.save_as

        # Build the model and tensorboard.
        self.build_model()

        # Load ckpt if exists.
        if self.save_as.is_file():
            self.load_weight(self.save_as)


    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.G.to(self.device)


    def load_weight(self, ckpt_path):
        """Loads model weights.
        @param  ckpt_path   `Path` to the checkpoint
        """
        print(f'Found a checkpoint at {ckpt_path}. Loading...')
        g_checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.G.load_state_dict(g_checkpoint['model'])
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #==========================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        # Map loss function names to PyTorch function
        loss_types_to_function = {
                'mse_loss': F.mse_loss,
                'l1_loss': F.l1_loss,
        }
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # ================================================================ #
            #                             1. Preprocess input data             #
            # ================================================================ #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.float().to(self.device) 
                        
       
            # ================================================================ #
            #                               2. Train the generator             #
            # ================================================================ #
            
            self.G = self.G.train()
                        
            # Users can choose loss function to apply.
            loss_function_id = loss_types_to_function[self.loss_type_id]
            loss_function_id_psnt = loss_types_to_function[self.loss_type_id_psnt]
            loss_function_cd = loss_types_to_function[self.loss_type_cd]

            # Identity mapping loss.
            x_identic, x_identic_psnt, code_real = self.G(
                x_real, emb_org, emb_org)
            x_identic = x_identic.squeeze(1)
            x_identic_psnt = x_identic_psnt.squeeze(1)
            g_loss_id = loss_function_id(x_real, x_identic)   
            g_loss_id_psnt = loss_function_id_psnt(x_real, x_identic_psnt)   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = loss_function_cd(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # ================================================================ #
            #                                 4. Miscellaneous                 #
            # ================================================================ #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

            # Save the model at every save_step
            if (i+1) % self.save_step == 0:
                print("Saving the model at step {}".format(i+1))
                torch.save(
                    {'model': self.G.state_dict(), 'optimizer': self.g_optimizer.state_dict()},
                    self.save_as
                )
